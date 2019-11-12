import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
from gensim.models import KeyedVectors
# from utilsXML import GEDoc
import utils
from collections import OrderedDict
import pdb
import argparse
import tqdm
from multitask_LSTM import BiLSTM
from predictor import Predictor
from evaluator import Evaluator
from sklearn.metrics import f1_score, precision_recall_fscore_support

def read_w2v_emb(word2idx, wv_file):
    word_emb = []
    wv_from_bin = KeyedVectors.load_word2vec_format(wv_file, binary=True)
    for word in word2idx:
        if word in wv_from_bin:
            word_emb.append(wv_from_bin[word])
        else:
            word_emb.append(wv_from_bin['UNK'])
    return np.array(word_emb)

def save_model(epoch, args, model, scores, optimizer, path):
    torch.save({'epoch': epoch,
                'args': args,
                'state_dict': model.state_dict(),
                'scores': scores,
                'optimizer' : optimizer.state_dict()
                }, path)

def remove_neg_data(data):
    """
    This is problematic, since the idx has been changed during iteration
    Not used any more
    """
    print "Original data len: {}".format(len(data))
    n_rm = 0
    for i, d in enumerate(data):
        trigger_labels = d[3]
        pair_idxs = d[4]
        if len(set(trigger_labels).intersection(set(["Gene_expression", "Transcription", "Protain_catabolism", "Localization",
                                                     "Phosphorylation", "Binding", "Regulation", "Positive_regulation", "Negative_regulation"]))) == 0:
            # meaning that this sentence does not have triggers
            # remove
            del data[i]
            n_rm += 1
            # continue
        # if len(pair_idxs) == 0:
        #     # meaning that this sentence does not have pairs to predict interaction
        #     # skip
        #     continue
    print "Removed number: {}".format(n_rm)
    return data
def construct_pairs(scores_trigger, gold_pair_idxs, gold_int_labels, gold_trigger_labels=None, args=None):
    """
    when gold_trigger_labels is fed in, then it is used to find the Protein entities
    when gold_trigger_labels is not fed in, then Protein is assumed to be unknown
    """
    def is_gold(pair_idx):
        return pair_idx in gold_pair_idxs
    res_pair_idxs = []
    res_int_labels = []
    entity_idxs = []
    trigger_idxs = []

    y_preds = scores_trigger.max(dim=1, keepdim=False)[1].tolist()
    for i in range(len(y_preds)):
        if gold_trigger_labels != None:
            # assume Protein  is given
            if gold_trigger_labels[i] in ['Protein']:
                entity_idxs.append(i)
        if args.idx2triggerlabel[y_preds[i]] == 'None':
            continue
        elif args.idx2triggerlabel[y_preds[i]] == 'Entity':
            entity_idxs.append(i)
        else:
            # trigger
            trigger_idxs.append(i)
    te_pair_idxs = [(i, j) for i in trigger_idxs for j in entity_idxs]
    tt_pair_idxs = [(i, j) for i in trigger_idxs for j in trigger_idxs if i != j]

    for pair in te_pair_idxs + tt_pair_idxs:
        # for all TE + TT pairs
        if not is_gold(pair):
            res_int_labels.append('None')
        else:
            res_int_labels.append(gold_int_labels[gold_pair_idxs.index(pair)])
        res_pair_idxs.append(pair)
    assert len(res_pair_idxs) == len(res_int_labels)
    # pdb.set_trace()

    return res_pair_idxs, res_int_labels


def train_epoch(data_train, model, optimizer, criterion, args):


    model.train()
    epoch_loss = 0
    # data_train = remove_neg_data(data_train)
    # pdb.set_trace()
    # all_data: corpus_ids, corpus_tokens, corpus_pos_tags, corpus_trigger_labels, corpus_interaction_idxs, corpus_interaction_labels
    for d in tqdm.tqdm(data_train):

        model.zero_grad()

        tokens = d[1]
        pos_tags = d[2]
        trigger_labels = d[3]
        assert len(tokens) == len(trigger_labels)

        tokens = [args.word2idx[i] for i in tokens]
        pos_tags = [args.pos2idx[i] for i in pos_tags]
        trigger_labels = [args.triggerlabel2idx[i] for i in trigger_labels]

        tokens = Variable(torch.LongTensor(np.array([tokens]).transpose()))
        pos_tags = Variable(torch.LongTensor(np.array([pos_tags]).transpose()))
        trigger_labels = Variable(torch.LongTensor(np.array(trigger_labels).transpose()))     # labels have to be one-dim for NLL loss


        if args.cuda:
            tokens, pos_tags, trigger_labels = [tokens.cuda(), pos_tags.cuda(), trigger_labels.cuda()]

        # first predict for triggers
        scores_trigger = model(tokens, pos_tags, pair_idxs=None, task='trigger')
        loss_trigger = criterion(scores_trigger, trigger_labels)


        # second predict edges, there are two cases
        if args.pred_edge_with_gold:
            # in this case, just use the gold pairs and predict the edge
            pair_idxs = d[4]
            interaction_labels = d[5]

        elif args.pred_edge_with_pred:
            # in this case, first construct the pairs with predicted triggers, pairs:(T, E), (T, T)
            # returned pair_idxs and ineteraction_labels can be empty
            pair_idxs, interaction_labels = construct_pairs(scores_trigger=scores_trigger, gold_pair_idxs=d[4], gold_int_labels=d[5], gold_trigger_labels=d[3], args=args)

        interaction_labels = [args.interactionlabel2idx[i] for i in interaction_labels]
        interaction_labels = Variable(torch.LongTensor(np.array(interaction_labels).transpose()))
        if args.cuda:
            interaction_labels = interaction_labels.cuda()

        loss_interaction = 0
        if len(pair_idxs) > 0:
            # Only compute loss for those sentences which have interactions
            scores_interaction = model(tokens, pos_tags, pair_idxs, task='interaction')
            loss_interaction = criterion(scores_interaction, interaction_labels)

        loss = args.trigger_w * loss_trigger + args.interaction_w * loss_interaction
        loss.backward()
        optimizer.step()
        epoch_loss += utils.to_scalar(loss)
    print('epoch loss:{}'.format(epoch_loss))

def predict(data, model, args):
    model.eval()
    y_trues_trigger = []
    y_preds_trigger = []
    y_trues_int = []
    y_preds_int = []

    for d in tqdm.tqdm(data):
        tokens = d[1]
        pos_tags = d[2]
        trigger_labels = d[3]
        assert len(tokens) == len(trigger_labels)
        tokens = [args.word2idx[i] for i in tokens]
        pos_tags = [args.pos2idx[i] for i in pos_tags]
        trigger_labels = [args.triggerlabel2idx[i] for i in trigger_labels]

        y_trues_trigger.extend(trigger_labels)


        tokens = Variable(torch.LongTensor(np.array([tokens]).transpose()))
        pos_tags = Variable(torch.LongTensor(np.array([pos_tags]).transpose()))
        trigger_labels = Variable(torch.LongTensor(np.array(trigger_labels).transpose()))     # labels have to be one-dim for NLL loss

        if args.cuda:
            model.cuda()
            tokens, pos_tags, trigger_labels = [tokens.cuda(), pos_tags.cuda(), trigger_labels.cuda()]

        # first predict for triggers
        scores_trigger = model(tokens, pos_tags, pair_idxs=None, task='trigger')
        loss_trigger = criterion(scores_trigger, trigger_labels)
        y_pred_trigger = scores_trigger.max(dim=1, keepdim=False)[1].tolist()
        y_preds_trigger.extend(y_pred_trigger)


        # second predict edges, there are two cases
        if args.pred_edge_with_gold:
            # in this case, just use the gold pairs and predict the edge
            pair_idxs = d[4]
            interaction_labels = d[5]

        elif args.pred_edge_with_pred:
            # in this case, first construct the pairs with predicted triggers, pairs:(T, E), (T, T)
            # returned pair_idxs and ineteraction_labels can be empty
            pair_idxs, interaction_labels = construct_pairs(scores_trigger=scores_trigger, gold_pair_idxs=d[4], gold_int_labels=d[5], gold_trigger_labels=d[3], args=args)


        interaction_labels = [args.interactionlabel2idx[i] for i in interaction_labels]
        y_trues_int.extend(interaction_labels)

        interaction_labels = Variable(torch.LongTensor(np.array(interaction_labels).transpose()))
        if args.cuda:
            interaction_labels = interaction_labels.cuda()

        loss_interaction = 0
        if len(pair_idxs) > 0:
            # Only compute loss for those sentences which have interactions
            scores_interaction = model(tokens, pos_tags, pair_idxs, task='interaction')
            y_pred_int = scores_interaction.max(dim=1, keepdim=False)[1].tolist()
            y_preds_int.extend(y_pred_int)

    return y_trues_trigger, y_preds_trigger, y_trues_int, y_preds_int
# def predict_noisy(data, model, args):
#     assert args.pred_edge_with_gold == False
#     assert args.pred_edge_with_pred == True
#     model.eval()
#     y_trues_trigger = []
#     y_preds_trigger = []
#     y_trues_int = []
#     y_preds_int = []
#
#
#
#     for d in tqdm.tqdm(data):
#
#
#         tokens = d[1]
#         pos_tags = d[2]
#         trigger_labels = d[3]
#         assert len(tokens) == len(trigger_labels)
#
#         tokens = [args.word2idx[i] for i in tokens]
#         pos_tags = [args.pos2idx[i] for i in pos_tags]
#         trigger_labels = [args.triggerlabel2idx[i] for i in trigger_labels]
#
#         y_trues_trigger.extend(trigger_labels)
#
#         tokens = Variable(torch.LongTensor(np.array([tokens]).transpose()))
#         pos_tags = Variable(torch.LongTensor(np.array([pos_tags]).transpose()))
#         trigger_labels = Variable(torch.LongTensor(np.array(trigger_labels).transpose()))     # labels have to be one-dim for NLL loss
#
#
#         if args.cuda:
#             tokens, pos_tags, trigger_labels = [tokens.cuda(), pos_tags.cuda(), trigger_labels.cuda()]
#
#         # first predict for triggers
#         scores_trigger = model(tokens, pos_tags, pair_idxs=None, task='trigger')
#         y_pred_trigger = scores_trigger.max(dim=1, keepdim=False)[1].tolist()
#         y_preds_trigger.extend(y_pred_trigger)
#
#
#         # second predict edges, there are two cases
#         if args.pred_edge_with_gold:
#             # in this case, just use the gold pairs and predict the edge
#             pair_idxs = d[4]
#             interaction_labels = d[5]
#
#         elif args.pred_edge_with_pred:
#             # in this case, first construct the pairs with predicted triggers, pairs:(T, E), (T, T)
#             # returned pair_idxs and ineteraction_labels can be empty
#             pair_idxs, interaction_labels = construct_pairs(scores_trigger=scores_trigger, gold_pair_idxs=d[4], gold_int_labels=d[5], gold_trigger_labels=d[3], args=args)
#
#         interaction_labels = [args.interactionlabel2idx[i] for i in interaction_labels]
#         interaction_labels = Variable(torch.LongTensor(np.array(interaction_labels).transpose()))
#         if args.cuda:
#             interaction_labels = interaction_labels.cuda()
#
#         loss_interaction = 0
#         if len(pair_idxs) > 0:
#             # Only compute loss for those sentences which have interactions
#             scores_interaction = model(tokens, pos_tags, pair_idxs, task='interaction')

def evaluate(y_trues_trigger, y_preds_trigger, y_trues_int, y_preds_int, final_eval):
    if not str2bool(final_eval):
        f1_trigger = f1_score(y_true=y_trues_trigger, y_pred=y_preds_trigger, average='micro')
        f1_int = f1_score(y_true=y_trues_int, y_pred=y_preds_int, average='micro')
        # pdb.set_trace()
        return f1_trigger, f1_int
    elif str2bool(final_eval):
        # for final eval, return a full report for prec, recall, f1, support
        prec_trigger, recall_trigger, f1_trigger, support_trigger = precision_recall_fscore_support(y_trues_trigger, y_preds_trigger, average=None)
        prec_int, recall_int, f1_int, support_int = precision_recall_fscore_support(y_trues_int, y_preds_int, average=None)
        return prec_trigger, recall_trigger, f1_trigger, support_trigger, prec_int, recall_int, f1_int, support_int


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hid_dim', type=int, default=60)
    p.add_argument('--n_layers', type=int, default=1)
    p.add_argument('--n_epoch', type=int, default=20)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--trainable_emb', type=str2bool, default=False)
    p.add_argument('--cuda', type=str2bool, default=True)
    p.add_argument('--lr', type=float, default=0.0005)
    p.add_argument('--opt', choices=['sgd', 'adam'], default='adam')
    p.add_argument('--random_seed', type=int, default=27)
    p.add_argument('--save_model_dir', type=str, default='./joint_models')
    p.add_argument('--patience', type=int, default=1)
    p.add_argument('--exclude_trigger', type=str2bool, default=True)
    p.add_argument('--trigger_w', type=float, default=1.0)
    p.add_argument('--interaction_w', type=float, default=1.0)
    p.add_argument('--pred_edge_with_gold', type=str2bool, default=True, help='When predicting interactions, using gold pair idx')
    p.add_argument('--pred_edge_with_pred', type=str2bool, default=True, help='When predicting interactions, using predicted pair idx, after some warmup epochs')
    p.add_argument('--n_warmup_epoch', type=int, default=20, help='warmup epochs for predicting interactions with gold pair idx')
    args = p.parse_args()


    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    data_train = pickle.load(open('./unmerging/GE09_train_flat.pkl', 'rb'))
    data_dev = pickle.load(open('./unmerging/GE09_dev_flat.pkl', 'rb'))

    # all_data: corpus_ids, corpus_tokens, corpus_pos_tags, corpus_trigger_labels, corpus_interaction_idxs, corpus_interaction_labels
    all_data = data_train + data_dev

    all_tokens = np.concatenate([d[1] for d in all_data])
    all_tokens = list(set(all_tokens))
    word_list = ['UNK'] + all_tokens
    word2idx = OrderedDict(zip(word_list, range(len(word_list))))
    args.word2idx = word2idx
    print "Loading w2v embeddings..."
    # w2v_emb = read_w2v_emb(word2idx, 'wikipedia-pubmed-and-PMC-w2v.bin')
    # np.save(open('w2v_emb.npy', 'wb'), w2v_emb)
    w2v_emb = np.load('w2v_emb.npy')

    all_pos = np.concatenate([d[2] for d in all_data])
    all_pos = list(set(all_pos))
    pos_list = ['UNK'] + all_pos
    pos2idx = OrderedDict(zip(pos_list, range(len(pos_list))))
    args.pos2idx = pos2idx
    pos_emb= np.zeros((len(pos2idx), len(pos2idx)))
    for i in range(pos_emb.shape[0]):
        pos_emb[i, i] = 1.0
    np.save(open('pos_emb.npy', 'wb'), pos_emb)
    pos_emb = np.load('pos_emb.npy')

    trigger_types = ["Gene_expression", "Transcription", "Protain_catabolism", "Localization",
                     "Phosphorylation", "Binding", "Regulation", "Positive_regulation", "Negative_regulation"]
    trigger_ignore_types = ['Protein']
    interaction_ignore_types = ['Site', 'ToLoc', 'AtLoc', 'SiteParent']


    all_trigger_labels = np.concatenate([d[3] for d in all_data])
    all_trigger_labels = list(set(all_trigger_labels))
    if args.exclude_trigger:
        for i in trigger_ignore_types:
            all_trigger_labels.remove(i)
    triggerlabel2idx = OrderedDict(zip(all_trigger_labels, range(len(all_trigger_labels))))
    if args.exclude_trigger:
        for i in trigger_ignore_types:
            # Do not see Protein  as triggers
            # All Protein  will be mapped to 'None'
            triggerlabel2idx[i] = triggerlabel2idx['None']
            # args.triggerlabel2idx['Entity'] = args.triggerlabel2idx['None']

    args.triggerlabel2idx = triggerlabel2idx

    # TODO: check this later
    # now the interaction labels exclude 'None', later when using predicted triggers,
    # 'None' should be included.
    all_interaction_labels = np.concatenate([d[5] for d in all_data])
    all_interaction_labels = list(set(all_interaction_labels))
    if args.pred_edge_with_pred:
        all_interaction_labels += ['None']
    interactionlabel2idx = OrderedDict(zip(all_interaction_labels, range(len(all_interaction_labels))))
    args.interactionlabel2idx = interactionlabel2idx


    idx2triggerlabel = {}
    for k, v in args.triggerlabel2idx.items():
        if k in trigger_ignore_types:
            # since Protein will both be mapped to None
            idx2triggerlabel[args.triggerlabel2idx['None']] = 'None'
        else:
            idx2triggerlabel[v] = k
    args.idx2triggerlabel = idx2triggerlabel

    idx2intlabel = {}
    for k, v in args.interactionlabel2idx.items():
        idx2intlabel[v] = k
    args.idx2intlabel = idx2intlabel
    # pdb.set_trace()



    model = BiLSTM(w2v_emb, pos_emb, args.hid_dim, args.dropout, args.batch, trainable_emb = args.trainable_emb,
                   n_layers = args.n_layers,
                   triggerset_dim=max(args.triggerlabel2idx.values())+1,
                   interactionset_dim=max(args.interactionlabel2idx.values())+1) # since the label2idx dict now have duplicated values, have to use max intsead of len
    model.rand_init()
    criterion = nn.NLLLoss()
    if args.cuda:
        model.cuda()
    if args.opt == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum)
    predictor = Predictor()
    # evaluator = Evaluator(predictor, args.label2idx, args.word2idx, args.pos2idx, args)

    print "Raw sentence number:{}".format(len(data_train))
    # Only use sentences which have triggers for training
    data_train = [i for i in data_train if len(set(i[3]).intersection(set(trigger_types)))!=0]
    print "Sentences have triggers number:{}".format(len(data_train))
    patience = 0
    best_score = 0
    best_epoch = 0
    for epoch in range(1, args.n_epoch+1):
        print('*'*10 + 'epoch {}'.format(epoch) + '*'*10)
        if epoch > args.n_warmup_epoch:
            args.pred_edge_with_gold = False
            args.pred_edge_with_pred = True
        train_epoch(data_train, model, optimizer, criterion, args)
        y_trues_trigger, y_preds_trigger, y_trues_int, y_preds_int = predict(data_dev, model, args)

        # f1_trigger, f1_int = evaluate(y_trues_trigger, y_preds_trigger, y_trues_int, y_preds_int, final_eval=False)
        # print "trigger f1 {}".format(f1_trigger)
        # print "interaction f1 {}".format(f1_int)
        prec_trigger, recall_trigger, f1_trigger, support_trigger, \
        prec_int, recall_int, f1_int, support_int = evaluate(y_trues_trigger, y_preds_trigger, y_trues_int, y_preds_int, final_eval=True)
        for k, v in idx2triggerlabel.items():
            print "trigger {}, prec {}, recall {}, f1 {}, support {}".format(v,
                                                                             prec_trigger[k],
                                                                             recall_trigger[k],
                                                                             f1_trigger[k],
                                                                             support_trigger[k])
        for k, v in idx2intlabel.items():
            # if args.pred_edge_with_gold:
                # if v == 'None':
                    # since when pred_edge_with_gold, no None edges will be constructed
                    # continue
            if k in y_trues_int:
                # dirty hack, precision_recall_fscore_support return labels in sorted order
                idx = sorted(list(set(y_trues_int))).index(k)
                print "interaction {}, prec {}, recall {}, f1 {}, support {}".format(v,
                                                                             prec_int[idx],
                                                                             recall_int[idx],
                                                                             f1_int[idx],
                                                                             support_int[idx])
        # for i in range(len(prec_int)):

        # elif args.pred_edge_with_pred:



        # args.final_eval = 1

        # f1_train = evaluator.evaluate(data_train+data_dev, model, args, cuda=True )
        # print('train micro f1: {}'.format(f1_train))

        # f1_dev = evaluator.evaluate(data_train+data_dev, model, args, cuda=True )
        # print('dev micro f1: {}'.format(f1_dev))
        # prec, recall, f1, support = evaluator.evaluate(data_train+data_dev, model, args, cuda=True )
        # idx2label = OrderedDict([(v, k) for k, v in args.label2idx.items()])
        # for i in range(len(prec)):
        #     print("Trigger type {}, prec {}, recall {}, f1 {}, support {}".format(idx2label[i], prec[i], recall[i], f1[i], support[i]))
        # if f1_dev > best_score:
        #     patience = 0
        #     best_score = f1_dev
        #     best_epoch = epoch
        #     save_model(best_epoch, args, model, best_score, optimizer, os.path.join(args.save_model_dir, 'DevF1{}_Epoch{}.model'.format(best_score, best_epoch)))
        #     print('Model saved as {}'.format(os.path.join(args.save_model_dir, 'DevF1{}_Epoch{}.model'.format(best_score, best_epoch))))
        # else:
        #     patience += 1
        # if patience > args.patience:
        #     print('Generating Final Eval...')
        #     args.final_eval = 1
        #     prec, recall, f1, support = evaluator.evaluate(data_dev, model, args, cuda=True )
        #     idx2label = OrderedDict([(v, k) for k, v in args.label2idx.items()])
        #     for i in range(len(prec)):
        #         print("Trigger type {}, prec {}, recall {}, f1 {}, support {}".format(idx2label[i], prec[i], recall[i], f1[i], support[i]))
        #     break
