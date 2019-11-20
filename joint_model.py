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


trigger_types = ["Gene_expression", "Transcription", "Protein_catabolism", "Localization",
                 "Phosphorylation", "Binding", "Regulation", "Positive_regulation", "Negative_regulation"]
trigger_ignore_types = ['Protein', 'Entity']
interaction_ignore_types = ['Site', 'ToLoc', 'AtLoc', 'SiteParent']

SIMPLE = ['Gene_expression', 'Transcription', 'Protain_catabolism', 'Localization', 'Phosphorylation']
REG = ['Negative_regulation', 'Positive_regulation', 'Regulation']
BIND = ['Binding']

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
def construct_pairs(y_preds, gold_pair_idxs, gold_int_labels, gold_trigger_labels=None, args=None, test=False):
    """
    when gold_trigger_labels is fed in, then it is used to find the Protein entities
    when gold_trigger_labels is not fed in, then Protein is assumed to be unknown
    when test is False, will return the constructed pairs and its label (both can be empty though)
    when test is True, will return the constructed pairs and an empty list
    """
    def is_gold(pair_idx, gold_pair_idxs, gold_int_labels):
        if not pair_idx in gold_pair_idxs:
            return False
        if gold_int_labels[gold_pair_idxs.index(pair_idx)] in interaction_ignore_types:
            # this is to exclude the Site ... args
            return False
        return True
        # return pair_idx in gold_pair_idxs
    if test:
        gold_int_labels = None
    else:
        assert gold_int_labels is not None

    # y_preds = scores_trigger.max(dim=1, keepdim=False)[1].tolist()
    res_pair_idxs = []
    res_int_labels = []
    entity_idxs = []
    trigger_idxs = []

    for i in range(len(y_preds)):
        if gold_trigger_labels != None:
            # assume Protein  is given
            if gold_trigger_labels[i] in ['Protein']:
                entity_idxs.append(i)
        if args.idx2triggerlabel[y_preds[i]] in ['None','Entity']:
            # Entity is also ignored, i.e. not considered as event trigger or Protein target
            continue
        # elif args.idx2triggerlabel[y_preds[i]] == 'Entity':
        #     entity_idxs.append(i)
        else:
            # trigger
            trigger_idxs.append(i)
    te_pair_idxs = [(i, j) for i in trigger_idxs for j in entity_idxs]
    # NOTE: here we only construct TT pairs if the source trigger is a REG event - the definition of dataset
    tt_pair_idxs = [(i, j) for i in trigger_idxs for j in trigger_idxs if i != j and args.idx2triggerlabel[y_preds[i]] in REG]

    for pair in te_pair_idxs + tt_pair_idxs:
        if not test:
            # for all TE + TT pairs
            if not is_gold(pair, gold_pair_idxs, gold_int_labels):
                res_int_labels.append('None')
            else:
                res_int_labels.append(gold_int_labels[gold_pair_idxs.index(pair)])
            # res_pair_idxs.append(pair)
        else:
            # test case, the gold_int_labels do not exist
            # just append the constructed pairs and leave the res_int_labels empty
            # pass
            assert gold_int_labels is None
        res_pair_idxs.append(pair)
    if test is False:
        assert len(res_pair_idxs) == len(res_int_labels)
    # pdb.set_trace()

    return res_pair_idxs, res_int_labels


def train_epoch(data_train, model, optimizer, criterion_t, criterion_i, args):


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
        loss_trigger = criterion_t(scores_trigger, trigger_labels)


        # second predict edges, there are two cases
        if args.pred_edge_with_gold:
            # in this case, just use the gold pairs and predict the edge
            pair_idxs = d[4]
            interaction_labels = d[5]
            assert len(pair_idxs) == len(interaction_labels)
            # only select Theme and Cause edges
            # this is to exclude the Site ... args
            # pair_idxs = [pair_idxs[i] for i in range(len(pair_idxs)) if interaction_labels[i] not in interaction_ignore_types]
            # interaction_labels = [interaction_labels[i] for i in range(len(interaction_labels)) if interaction_labels[i] not in interaction_ignore_types]

            # we construct the pairs using gold trigger labels
            # note that there can be None pairs
            pair_idxs, interaction_labels = construct_pairs(y_preds=[args.triggerlabel2idx[i] for i in d[3]],
                                                            gold_pair_idxs=d[4],
                                                            gold_int_labels=d[5],
                                                            gold_trigger_labels=d[3],
                                                            args=args,
                                                            test=False)

        elif args.pred_edge_with_pred:
            # in this case, first construct the pairs with predicted triggers, pairs:(T, E), (T, T)
            # returned pair_idxs and ineteraction_labels can be empty

            y_preds = scores_trigger.max(dim=1, keepdim=False)[1].tolist()
            # we construct the pairs using predicted triggers
            pair_idxs, interaction_labels = construct_pairs(y_preds=y_preds,
                                                            gold_pair_idxs=d[4],
                                                            gold_int_labels=d[5],
                                                            gold_trigger_labels=d[3],
                                                            args=args,
                                                            test=False)
        assert len(pair_idxs) == len(interaction_labels)
        assert set(interaction_labels).intersection(set(interaction_ignore_types)) == set([]), pdb.set_trace()#print(interaction_labels)

        interaction_labels = [args.interactionlabel2idx[i] for i in interaction_labels]
        interaction_labels = Variable(torch.LongTensor(np.array(interaction_labels).transpose()))
        if args.cuda:
            interaction_labels = interaction_labels.cuda()

        loss_interaction = 0
        if len(pair_idxs) > 0:
            # Only compute loss for those sentences which have interactions
            scores_interaction = model(tokens, pos_tags, pair_idxs, task='interaction')
            loss_interaction = criterion_i(scores_interaction, interaction_labels)

        loss = args.trigger_w * loss_trigger + args.interaction_w * loss_interaction
        loss.backward()
        optimizer.step()
        epoch_loss += utils.to_scalar(loss)
    print('epoch loss:{}'.format(epoch_loss))

def predict(data, model, args, test=False):
    '''
    When test is set to False, then the construct_pairs function will return the semi-gold int labels according to the real gold int labels
    When test is set to True, then the construct_pairs function will return empty int labels
    Note that the data_out always append the predicted int labels
    When dev:
        set test=False/True doesn't matter, pred_edge_with_gold, pred_edge_with_pred has to be (False, True)
    When test:
        set test=False/True doesn't matter, pred_edge_with_gold, pred_edge_with_pred has to be (False, True)
    '''
    model.eval()
    y_trues_trigger = []
    y_preds_trigger = []
    y_trues_int = []
    y_preds_int = []

    # save the predicted data structure for final event evaluation
    # the difference of data_out and gold data is in d[3], d[4], d[5], which contain
    # predicted trigger_types(No Protein No Entity), pair_idxs, int_labels(Only Theme and Cause)
    data_out = []
    for d in tqdm.tqdm(data):

        d_out = []
        d_out.extend([d[0], d[1], d[2]])

        tokens = d[1]
        pos_tags = d[2]
        trigger_labels = d[3]
        # if 'Protein' in trigger_labels:
        #     pdb.set_trace()
        assert len(tokens) == len(trigger_labels)
        tokens = [args.word2idx.get(i, args.word2idx['UNK']) for i in tokens]
        pos_tags = [args.pos2idx.get(i,args.word2idx['UNK']) for i in pos_tags]
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
        # loss_trigger = criterion(scores_trigger, trigger_labels)
        y_pred_trigger = scores_trigger.max(dim=1, keepdim=False)[1].tolist()
        y_preds_trigger.extend(y_pred_trigger)

        d_out.append([args.idx2triggerlabel[i] for i in y_pred_trigger])

        # second predict edges, there are two cases
        if args.pred_edge_with_gold:
            # in this case, just use the gold pairs and predict the edge
            pair_idxs = d[4]
            interaction_labels = d[5]
            assert len(pair_idxs) == len(interaction_labels)
            # only select Theme and Cause edges
            # this is to exclude the Site ... args
            # pair_idxs = [pair_idxs[i] for i in range(len(pair_idxs)) if interaction_labels[i] not in interaction_ignore_types]
            # interaction_labels = [interaction_labels[i] for i in range(len(interaction_labels)) if interaction_labels[i] not in interaction_ignore_types]

            # we construct the pairs using gold trigger labels
            # note that there can be None pairs
            pair_idxs, interaction_labels = construct_pairs(y_preds=[args.triggerlabel2idx[i] for i in d[3]],
                                                            gold_pair_idxs=d[4],
                                                            gold_int_labels=d[5],
                                                            gold_trigger_labels=d[3],
                                                            args=args,
                                                            test=test)

        elif args.pred_edge_with_pred:
            # in this case, first construct the pairs with predicted triggers, pairs:(T, E), (T, T)
            # returned pair_idxs and ineteraction_labels can be empty
            y_preds = scores_trigger.max(dim=1, keepdim=False)[1].tolist()
            pair_idxs, interaction_labels = construct_pairs(y_preds=y_preds,
                                                            gold_pair_idxs=d[4],
                                                            gold_int_labels=d[5],
                                                            gold_trigger_labels=d[3],
                                                            args=args,
                                                            test=test)

        # assert len(pair_idxs) == len(interaction_labels)
        assert set(interaction_labels).intersection(set(interaction_ignore_types)) == set([]), pdb.set_trace()#print(interaction_labels)

        d_out.append(pair_idxs) # d[4]

        # TODO: Need to think about this !!!!!!!!!!!!!!!!!!!
        # When test, the interaction_labels will be for sure []
        # When not test, the interaction_labels will constructed by the constructed_pairs function, but can be empty though
        interaction_labels = [args.interactionlabel2idx[i] for i in interaction_labels]
        y_trues_int.extend(interaction_labels)

        # interaction_labels = Variable(torch.LongTensor(np.array(interaction_labels).transpose()))
        # if args.cuda:
        #     interaction_labels = interaction_labels.cuda()

        # loss_interaction = 0
        if len(pair_idxs) > 0:
            # Only compute loss for those sentences which have interactions
            scores_interaction = model(tokens, pos_tags, pair_idxs, task='interaction')
            y_pred_int = scores_interaction.max(dim=1, keepdim=False)[1].tolist()
            int_pred = [args.idx2intlabel[y_pred_int[i]] for i in range(len(y_pred_int))]
            d_out.append(int_pred)  # d[5]
            y_preds_int.extend(y_pred_int)
        else:
            d_out.append([])

        d_out.append(d[6])  # d[6]


        data_out.append(d_out)

    return y_trues_trigger, y_preds_trigger, y_trues_int, y_preds_int, data_out

def safe_division(num, den, on_err=0.0):
    return on_err if den == 0.0 else float(num)/float(den)

def cal_prec_rec_f1(n_corr, n_pred, n_true):
    prec = safe_division(n_corr, n_pred)
    recall = safe_division(n_corr, n_true)
    f1 = safe_division(2*prec*recall, prec+recall)
    return prec, recall, f1

def evaluate(y_trues_trigger, y_preds_trigger, y_trues_int, y_preds_int, final_eval, args):
    if not str2bool(final_eval):
        f1_trigger = f1_score(y_true=y_trues_trigger, y_pred=y_preds_trigger, average='micro')
        f1_int = f1_score(y_true=y_trues_int, y_pred=y_preds_int, average='micro')
        # pdb.set_trace()
        return f1_trigger, f1_int
    elif str2bool(final_eval):
        # for final eval, return a full report for prec, recall, f1, support
        prec_trigger, recall_trigger, f1_trigger, support_trigger = precision_recall_fscore_support(y_trues_trigger, y_preds_trigger, average=None)
        prec_int, recall_int, f1_int, support_int = precision_recall_fscore_support(y_trues_int, y_preds_int, average=None)

        n_corr_trigger = 0
        n_true_trigger = len([i for i in y_trues_trigger if args.idx2triggerlabel[i] != 'None'])
        n_pred_trigger = len([i for i in y_preds_trigger if args.idx2triggerlabel[i] != 'None'])

        for t_true, t_pred in zip(y_trues_trigger, y_preds_trigger):
            if t_true == t_pred and args.idx2triggerlabel[t_true] != 'None':
                n_corr_trigger += 1

        n_corr_int = 0
        n_true_int = len([i for i in y_trues_int if args.idx2intlabel[i] != 'None'])
        n_pred_int = len([i for i in y_preds_int if args.idx2intlabel[i] != 'None'])

        for t_true, t_pred in zip(y_trues_int, y_preds_int):
            if t_true == t_pred and args.idx2intlabel[t_true] != 'None':
                n_corr_int += 1

        avg_prec_trigger, avg_recall_trigger, avg_f1_trigger = cal_prec_rec_f1(n_corr_trigger, n_pred_trigger, n_true_trigger)
        avg_prec_int, avg_recall_int, avg_f1_int = cal_prec_rec_f1(n_corr_int, n_pred_int, n_true_int)
        return prec_trigger, recall_trigger, f1_trigger, support_trigger, prec_int, recall_int, f1_int, support_int, \
                avg_prec_trigger, avg_recall_trigger, avg_f1_trigger, avg_prec_int, avg_recall_int, avg_f1_int


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
    p.add_argument('--exp_id', type=str, default='test')
    p.add_argument('--hid_dim', type=int, default=70)
    p.add_argument('--n_layers', type=int, default=1)
    p.add_argument('--n_epoch', type=int, default=30)
    p.add_argument('--dropout', type=float, default=0.4)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--trainable_emb', type=str2bool, default=False)
    p.add_argument('--cuda', type=str2bool, default=True)
    p.add_argument('--lr', type=float, default=0.002)   # 0.005 is a very good choice
    p.add_argument('--opt', choices=['sgd', 'adam'], default='adam')
    p.add_argument('--random_seed', type=int, default=42)
    p.add_argument('--save_model_dir', type=str, default='./joint_models_1119')
    p.add_argument('--out_pkl_dir', type=str, default='./out_pkl_1119')
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--exclude_trigger', type=str2bool, default=True)
    p.add_argument('--exclude_interaction', type=str2bool, default=True)
    p.add_argument('--trigger_w', type=float, default=1.0)
    p.add_argument('--interaction_w', type=float, default=1.0)
    p.add_argument('--pred_edge_with_gold', type=str2bool, default=True, help='When predicting interactions, using gold pair idx')
    p.add_argument('--pred_edge_with_pred', type=str2bool, default=False, help='When predicting interactions, using predicted pair idx, after some warmup epochs')
    p.add_argument('--tw_none', type=str2bool, default=False, help='weights of trigger label to deal with the highly imbalance of triggers')
    p.add_argument('--n_warmup_epoch', type=int, default=5, help='warmup epochs for predicting interactions with gold pair idx')
    # p.add_argument('--out_pkl_dev', type=str, default='GE11_dev-pred-w-gold')
    # p.add_argument('--out_pkl_test', type=str, default='GE11_test-pred-w-gold')
    # p.add_argument('--model_name', type=str, default='joint_model_pred-w-gold.pth')
    p.add_argument('--task', type=str, default='GE11')
    args = p.parse_args()


    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for path in [args.save_model_dir, args.out_pkl_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

    print args

    data_train = pickle.load(open('./unmerging/{}_train_flat_w-span.pkl'.format(args.task), 'rb'))
    data_dev = pickle.load(open('./unmerging/{}_dev_flat_w-span.pkl'.format(args.task), 'rb'))
    data_test = pickle.load(open('./unmerging/{}_test_flat_w-span.pkl'.format(args.task), 'rb'))

    # all_data: corpus_ids, corpus_tokens, corpus_pos_tags, corpus_trigger_labels, corpus_interaction_idxs, corpus_interaction_labels, corpus_spans
    all_data = data_train + data_dev + data_test

    all_tokens = np.concatenate([d[1] for d in all_data])
    all_tokens = list(set(all_tokens))
    word_list = ['UNK'] + all_tokens
    word2idx = OrderedDict(zip(word_list, range(len(word_list))))
    args.word2idx = word2idx
    print "Loading w2v embeddings..."
    # w2v_emb = read_w2v_emb(word2idx, 'wikipedia-pubmed-and-PMC-w2v.bin')
    # np.save(open('w2v_emb_GE11.npy', 'wb'), w2v_emb)
    w2v_emb = np.load('w2v_emb_GE11.npy')

    all_pos = np.concatenate([d[2] for d in all_data])
    all_pos = list(set(all_pos))
    pos_list = ['UNK'] + all_pos
    pos2idx = OrderedDict(zip(pos_list, range(len(pos_list))))
    args.pos2idx = pos2idx
    # pos_emb= np.zeros((len(pos2idx), len(pos2idx)))
    # for i in range(pos_emb.shape[0]):
    #     pos_emb[i, i] = 1.0
    # np.save(open('pos_emb.npy', 'wb'), pos_emb)
    pos_emb = np.load('pos_emb.npy')


    all_trigger_labels = np.concatenate([d[3] for d in all_data])
    all_trigger_labels = list(set(all_trigger_labels))
    if args.exclude_trigger:
        for i in trigger_ignore_types:
            all_trigger_labels.remove(i)
        assert set(all_trigger_labels) == set(trigger_types + ['None']), pdb.set_trace()
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
    all_interaction_labels = list(set(all_interaction_labels))   # there is NO None in all_interaction_labels
    # if args.pred_edge_with_pred:
    # 'None' will be added anyway, because the theme and cause ints are sort-of sparse in the original annotation
    # if no 'None', the predicted ints will generate too many false positives
    all_interaction_labels += ['None']
    if args.exclude_interaction:
        for i in interaction_ignore_types:
            all_interaction_labels.remove(i)
    interactionlabel2idx = OrderedDict(zip(all_interaction_labels, range(len(all_interaction_labels))))
    # NOTE: the 'Site', 'SiteParent' will not be used as training samples anyway
    # if args.exclude_interaction:
    #     for i in interaction_ignore_types:
    #         # Map the 'Site', 'SiteParent'... to None
    #         interactionlabel2idx[i] = interactionlabel2idx['None']
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


    if args.tw_none:
        trigger_weights = torch.Tensor([0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        if args.cuda:
            trigger_weights = trigger_weights.cuda()
    else:
        trigger_weights = None
    criterion_t = nn.NLLLoss(weight=trigger_weights)
    criterion_i = nn.NLLLoss()
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
        if args.pred_edge_with_pred:
            if epoch > args.n_warmup_epoch:
                args.pred_edge_with_gold = False
                args.pred_edge_with_pred = True
                # best_score = 0
                # best_epoch = 0
        else:
            assert args.pred_edge_with_gold == True
        train_epoch(data_train, model, optimizer, criterion_t,criterion_i, args)
        y_trues_trigger, y_preds_trigger, y_trues_int, y_preds_int, data_out = predict(data_dev, model, args, test=False)

        # f1_trigger, f1_int = evaluate(y_trues_trigger, y_preds_trigger, y_trues_int, y_preds_int, final_eval=False)
        # print "trigger f1 {}".format(f1_trigger)
        # print "interaction f1 {}".format(f1_int)
        prec_trigger, recall_trigger, f1_trigger, support_trigger, \
        prec_int, recall_int, f1_int, support_int, \
        avg_prec_trigger, avg_recall_trigger, avg_f1_trigger, \
        avg_prec_int, avg_recall_int, avg_f1_int = evaluate(y_trues_trigger, y_preds_trigger, y_trues_int, y_preds_int, final_eval=True, args=args)
        print 'Average trigger: prec {}, recall {}, f1 {}'.format(avg_prec_trigger, avg_recall_trigger, avg_f1_trigger)
        print 'Average int: prec {}, recall {}, f1 {}'.format(avg_prec_int, avg_recall_int, avg_f1_int)
        for k, v in idx2triggerlabel.items():
            print "trigger {}, prec {}, recall {}, f1 {}, support {}".format(v,
                                                                             prec_trigger[k],
                                                                             recall_trigger[k],
                                                                             f1_trigger[k],
                                                                             support_trigger[k])
        for k, v in idx2intlabel.items():
            # if args.pred_edge_with_gold:
            #     if v == 'None':
            #         since when pred_edge_with_gold, no None edges will be constructed
            #         continue
            if k in y_trues_int:
                # dirty hack, precision_recall_fscore_support return labels in sorted order
                idx = sorted(list(set(y_trues_int))).index(k)
                print "interaction {}, prec {}, recall {}, f1 {}, support {}".format(v,
                                                                             prec_int[idx],
                                                                             recall_int[idx],
                                                                             f1_int[idx],
                                                                             support_int[idx])

        if avg_f1_trigger + avg_f1_int < best_score:
            patience += 1
        else:
            patience = 0
            best_epoch = epoch
            best_score = avg_f1_trigger + avg_f1_int

            meta_name = 'EXP{}_Tf1{:.4f}_If1{:.4f}_pred-w-gold{}_pred-w-pred{}_lr{}_drp{}_ep{}_hd{}_seed{}_tw{}_iw{}_trEmb{}_twNone{}'.format(args.exp_id,
                                                                                             avg_f1_trigger,
                                                                                             avg_f1_int,
                                                                                             args.pred_edge_with_gold,
                                                                                             args.pred_edge_with_pred,
                                                                                             args.lr,
                                                                                             args.dropout,
                                                                                             best_epoch,
                                                                                             args.hid_dim,
                                                                                             args.random_seed,
                                                                                             args.trigger_w,
                                                                                             args.interaction_w,
                                                                                             args.trainable_emb,
                                                                                             args.tw_none)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                   }, '{}/{}.pth'.format(args.save_model_dir, meta_name))
            print 'model saved as ' + '{}/{}'.format(args.save_model_dir, meta_name)

        if patience > args.patience:
            print  '='*10 + 'early stopped at epoch {}'.format(best_epoch) + '='*10
            break
    # pdb.set_trace()

    # meta_name = 'EXP007_Tf10.62026295437_If10.78624813154_pred-w-goldTrue_pred-w-predFalse_lr0.002_drp0.4_ep15_hd70_seed42_tw1.0_iw0.5'


    checkpoint = torch.load('{}/{}.pth'.format(args.save_model_dir, meta_name))
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    args.pred_edge_with_pred = True  # This is important for testing !!!
    args.pred_edge_with_gold = False
    y_trues_trigger, y_preds_trigger, y_trues_int, y_preds_int, data_out = predict(data_dev, model, args, test=True)
    with open('{}/{}_dev_{}.pkl'.format(args.out_pkl_dir, args.task, 'fortest'), 'wb') as f:
        pickle.dump(data_out, f)



    model.eval()
    args.pred_edge_with_pred = True  # This is important for testing !!!
    args.pred_edge_with_gold = False
    y_trues_trigger, y_preds_trigger, y_trues_int, y_preds_int, data_out = predict(data_test, model, args, test=True)
    with open('{}/{}_test_{}.pkl'.format(args.out_pkl_dir, args.task, 'fortest'), 'wb') as f:
        pickle.dump(data_out, f)



