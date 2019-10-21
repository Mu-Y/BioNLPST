import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import model.utils as utils
from model.LSTM import BiLSTM
import argparse
import pdb
import pickle
from testAnn import DataProcessed
from collections import OrderedDict
import os
import numpy as np

from model.predictor import Predictor
from model.evaluator import Evaluator



def train_epoch(train_data, ner_model, optimizer, criterion, args):
    ner_model.train()
    # if args.cuda:
      #  ner_model = ner_model.cuda()
    # print(len(train_data))
    epoch_loss = 0
    for d in train_data:
        input_d = d[1]
        for tokens, pos_tags, labels in zip(input_d[0], input_d[1], input_d[2]):
            y_true = labels
            
            tokens = Variable(torch.LongTensor(np.array([tokens]).transpose()))
            pos_tags = Variable(torch.LongTensor(np.array([pos_tags]).transpose()))
            labels = Variable(torch.LongTensor(np.array(labels).transpose()))     # labels have to be one-dim for NLL loss
            
            if args.cuda:
                tokens, pos_tags, labels = [tokens.cuda(), pos_tags.cuda(), labels.cuda()]
            
            scores = ner_model(tokens, pos_tags)
            # print(scores)
            # print(labels)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += utils.to_scalar(loss)
    
    return epoch_loss
def print_scores(eval_result, args):
    if args.eval_separate_VN:
        print("[Entities]: ")
        print("Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(eval_result[1][4], eval_result[2][4], eval_result[0][4]))
        print("[Attr]: ")
        print("Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}, Acc: {:.3f}".format(eval_result[1][5], eval_result[2][5], eval_result[0][5], eval_result[0][6]))
        print("[TIMEX]: ")
        print("Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(eval_result[1][0], eval_result[2][0], eval_result[0][0]))
        print("[EVENT]: ")
        print("Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(eval_result[1][1], eval_result[2][1], eval_result[0][1]))
        print("[EVENT_V]: ")
        print("Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(eval_result[1][2], eval_result[2][2], eval_result[0][2]))
        print("[EVENT_N]: ")
        print("Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(eval_result[1][3], eval_result[2][3], eval_result[0][3]))
    elif args.eval_separate_all:
        print("[TIMEX]: ")
        print("Precision: {}, Recall: {}, F1: {}".format(eval_result[1][0], eval_result[2][0], eval_result[0][0]))
        print("[EVENT]: ")
        print("Precision: {}, Recall: {}, F1: {}".format(eval_result[1][1], eval_result[2][1], eval_result[0][1]))
        print("[EVENT_VERB]: ")
        print("Precision: {}, Recall: {}, F1: {}".format(eval_result[1][2], eval_result[2][2], eval_result[0][2]))
        print("[EVENT_NOUN]: ")
        print("Precision: {}, Recall: {}, F1: {}".format(eval_result[1][3], eval_result[2][3], eval_result[0][3]))
        print("[EVENT_OTHER]: ")
        print("Precision: {}, Recall: {}, F1: {}".format(eval_result[1][4], eval_result[2][4], eval_result[0][4]))
        print("[EVENT_ADJ]: ")
        print("Precision: {}, Recall: {}, F1: {}".format(eval_result[1][5], eval_result[2][5], eval_result[0][5]))
        print("[EVENT_PREP]: ")
        print("Precision: {}, Recall: {}, F1: {}".format(eval_result[1][6], eval_result[2][6], eval_result[0][6]))
    else:
        print("[TIMEX]: ")
        print("Precision: {}, Recall: {}, F1: {}".format(eval_result[1][0], eval_result[2][0], eval_result[0][0]))
        print("[EVENT]: ")
        print("Precision: {}, Recall: {}, F1: {}".format(eval_result[1][1], eval_result[2][1], eval_result[0][1]))
def save_model(epoch, args, ner_model, scores, optimizer, path):
    torch.save({'epoch': epoch,
                'args': args,
                'state_dict': ner_model.state_dict(),
                'scores': scores,
                'optimizer' : optimizer.state_dict()
                }, path)

if __name__ == '__main__':
    # TODO:
#*0.    procData + testAnn pipeline in single file.
#*1.    Add batch_norm
#*2.    Add more linear layer for MLP
#**3.    Learning rate decay
# 3.    Add shuffle
# 4.    Linguistic feats
# 5.    Criterion: Loss sum or Loss mean?


    p = argparse.ArgumentParser()
    
    p.add_argument('-load_data', help='path to processed data (pickle file)')
    p.add_argument('-word_emb', help='path of word emb (glove)')
    p.add_argument('-word_emb_dim', help='dim of word emb (glove)')

    # arguments for RNN model
    p.add_argument('-word_hid_dim', type=int, default=20)
    p.add_argument('-batch', type=int, default=1, help='batch size')
    p.add_argument('-epochs', type=int, default=100)
    p.add_argument('-seed', type=int, default=123)
    p.add_argument('-lr', type=float, default=0.0005)
    p.add_argument('-dropout', type=float, default=0.1)
    p.add_argument('-cuda', action='store_true')
    p.add_argument('-ignore_TIMEX', action='store_true')
    p.add_argument('-trainable_emb', action='store_true', help='Make word embedding and pos embedding trainable.')
    p.add_argument('-non_trainable_emb', action='store_true', help='Make word embedding and pos embedding non_trainable.')
    p.add_argument('-eval_on_dev', action='store_true', help='When set, first train on train_data and eval on dev_data for args.epochs epoch \
                    to look for best_epoch. Then re-train on train_dev + dev_data for best_epoch epochs and test on test_data.\
                    When not set, directly train on train_data + dev_data for args.epochs epochs and then test on test_data.')
    p.add_argument('-train_separate_all', action='store_true', help='eval EVENT of all different pos separately')
    p.add_argument('-eval_separate_all', action='store_true', help='eval EVENT of all different pos separately')
    p.add_argument('-train_separate_VN', action='store_true', help='train EVENT of VERB and all other pos separately')
    p.add_argument('-eval_separate_VN', action='store_true', help='eval EVENT of VERB and all other pos separately')
    # p.add_argument('-target_label', type=str, default=None, help='when args.eval_separate_all: {TIMEX, EVENT_VERB, EVENT_NOUN, EVENT_ADJ, EVENT_PREP, EVENT_OTHER},\
    #                   when args.eval_separate_VN: {TIMEX, EVENT_V, EVENT_N}')
    p.add_argument('-opt', choices=['sgd', 'adam'], default='adam')
    p.add_argument('-save_model', type=str, help='path to save the model checkpoint')
    args = p.parse_args()
    
    print('\n', args, '\n')
    torch.manual_seed(args.seed)

    # args.load_data = '/Users/muyang/Desktop/event/dataset/data'
    # args.word_emb = '/Users/muyang/Desktop/event/dataset/glove.6B/glove.6B.50d.txt'
    # args.save_model = '/Users/muyang/Desktop/event/event-detector/LSTM/checkpoints'

    # load data
    data = pickle.load(open(args.load_data + '/all_data_3.pkl', 'rb'))

    if args.train_separate_all:
        data.label2idx = OrderedDict([('O', 0),
                                     ('B_EVENT_VERB', 1), ('I_EVENT_VERB', 2),
                                     ('B_EVENT_NOUN', 3), ('I_EVENT_NOUN', 4),
                                     ('B_EVENT_OTHER', 5),('I_EVENT_OTHER', 6),
                                     ('B_EVENT_ADJ', 7), ('I_EVENT_ADJ', 8),
                                     ('B_EVENT_PREP', 9), ('I_EVENT_PREP', 10),
                                     ('B_TIMEX', 11), ('I_TIMEX', 12)])
    elif args.train_separate_VN:
        data.label2idx = OrderedDict([('O', 0),
                                     ('B_EVENT_VERB', 1), ('I_EVENT_VERB', 2),
                                     ('B_EVENT_NOUN', 3), ('I_EVENT_NOUN', 4),
                                     ('B_EVENT_OTHER', 3),('I_EVENT_OTHER',4),
                                     ('B_EVENT_ADJ', 3), ('I_EVENT_ADJ',4),
                                     ('B_EVENT_PREP', 3), ('I_EVENT_PREP', 4),
                                     ('B_TIMEX', 5), ('I_TIMEX', 6)])

    else:
        data.label2idx = OrderedDict([('O', 0),
                                     ('B_EVENT_VERB', 1), ('I_EVENT_VERB', 2),
                                     ('B_EVENT_NOUN', 1), ('I_EVENT_NOUN',2),
                                     ('B_EVENT_OTHER', 1),('I_EVENT_OTHER',2),
                                     ('B_EVENT_ADJ', 1), ('I_EVENT_ADJ',2),
                                     ('B_EVENT_PREP', 1), ('I_EVENT_PREP', 2),
                                     ('B_TIMEX', 3), ('I_TIMEX', 4)])

    if args.ignore_TIMEX:
        data.label2idx['B_TIMEX']=0
        data.label2idx['I_TIMEX']=0

    data.label_idx = [[[data.label2idx[t] for t in sent] for sent in doc ] for doc in data.labels]
        # data_tokens, data_poss, data_labels = [], [], []
        # for doc_id in range(len(data.tokens)):
            # doc_tokens, doc_poss, doc_labels = [], [], []
            # for tokens, pos_tags, labels in zip(data.token_idx[doc_id], data.pos_idx[doc_id], data.label_idx[doc_id]):
                # if len(tokens) > 5:
                    # doc_tokens.append(tokens)
                    # doc_poss.append(pos_tags)
                    # doc_labels.append(labels)
            # data_tokens.append(doc_tokens)
            # data_poss.append(doc_poss)
            # data_labels.append(doc_labels)
        
        # data.token_idx = data_tokens
        # data.pos_idx = data_poss
        # data.label_idx = data_labels
    # pdb.set_trace() 
    
    filenames = os.listdir(args.load_data + '/tbdense/timeml')
    dev_files = ["APW19980227.0487",
                 "CNN19980223.1130.0960",
                 "NYT19980212.0019",
                 "PRI19980216.2000.0170",
                 "ed980111.1130.0089"]

    test_files = ["APW19980227.0489",
                  "APW19980227.0494",
                  "APW19980308.0201",
                  "APW19980418.0210",
                  "CNN19980126.1600.1104",
                  "CNN19980213.2130.0155",
                  "NYT19980402.0453",
                  "PRI19980115.2000.0186",
                  "PRI19980306.2000.1675"]
    
    train_files = [r for r in data.ids if not r in dev_files + test_files]
    
    word2idx, pos2idx, label2idx = data.word2idx, data.pos2idx, data.label2idx
    
    train_data, dev_data, test_data = [], [], []
    for i in range(len(data.ids)):
        tmp = []
        if data.ids[i] in train_files:
            tmp = train_data
        elif data.ids[i] in dev_files:
            tmp = dev_data
        elif data.ids[i] in test_files:
            tmp = test_data
        tmp.append([[data.ids[i], data.tokens[i], data.pos_tags[i], data.labels[i]], [data.token_idx[i], data.pos_idx[i], data.label_idx[i]]])
    
    print("Data Loaded!")
    # load word embedding
    word_emb = utils.read_glove(args.word_emb + '/glove.6B.{}d.txt'.format(args.word_emb_dim), word2idx)
    word_emb = np.array(word_emb)
    
    # pdb.set_trace()
    # add start tag for crf
    # label2idx['<start>'] = len(label2idx)
    # label2idx['<end>'] = len(label2idx)
    # word2idx['<start>'] = len(word2idx)
    # word2idx['<end>'] = len(word2idx)
    # pos2idx['<start>'] = len(pos2idx)+1 # +1 to hold the place for unk pos_tags
    # pos2idx['<end>'] = len(pos2idx)+1
    # word_emb = np.vstack((np.random.uniform(0, 1, (2, word_emb.shape[1])), word_emb))


    pos_emb= np.zeros((len(pos2idx) + 1, len(pos2idx) + 1))
    for i in range(pos_emb.shape[0]):
        pos_emb[i, i] = 1.0
    
    if args.eval_on_dev:
 
        ner_model = BiLSTM(word_emb, pos_emb, args.word_hid_dim, max(list(label2idx.values()))+1, args.dropout, args.batch, trainable_emb = args.trainable_emb)
        ner_model.rand_init()
        criterion = nn.NLLLoss()
        if args.cuda:
            ner_model.cuda()
        if args.opt == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, ner_model.parameters()), lr=args.lr)
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, ner_model.parameters()), lr=args.lr, momentum=args.momentum)
        
        predictor = Predictor()
        evaluator = Evaluator(predictor, label2idx, word2idx, pos2idx, args)

        best_scores = []
        best_dev_f1_sum = 0
        patience = 0

        print('\n'*2)
        print('='*10 + 'Phase1, train on train_data, epoch=args.epochs' + '='*10)
        print('\n'*2)
        for epoch in range(1, args.epochs+1):
            
            loss = train_epoch(train_data, ner_model, optimizer, criterion, args)
            print("*"*10 + "epoch:{}, loss:{}".format(epoch, loss) + "*"*10)
            eval_result_train = evaluator.evaluate(train_data, ner_model, args, cuda = args.cuda)
            print("On train_data: ")
            print_scores(eval_result_train, args)
            eval_result_dev = evaluator.evaluate(dev_data, ner_model, args, cuda = args.cuda)
            print("On dev_data: ")
            print_scores(eval_result_dev, args)

            if eval_result_dev[0][0] + eval_result_dev[0][1] >= best_dev_f1_sum:
                best_scores = [loss, eval_result_train, eval_result_dev]
                best_dev_f1_sum = eval_result_dev[0][0] + eval_result_dev[0][1]
                
                save_model(epoch, args, ner_model, best_scores, optimizer,
                           args.save_model + '/epoch{}_dev_TF1_{}_EF1_{}.model'.format(epoch, eval_result_dev[0][0], eval_result_dev[0][1]))
                print('Model saved as:', args.save_model + '/epoch{}_dev_TF1_{}_EF1_{}.model'.format(epoch, eval_result_dev[0][0], eval_result_dev[0][1]))
                best_epoch = epoch
                patience = 0

            patience += 1
            if patience > 10: # Early stopping, best_f1_sum does not change after successive 10 epochs.
                print("****We should Early stopped at epoch{}.****".format(best_epoch))
                break
    else:
        best_epoch = args.epochs

    print('\n'*2)
    print('='*10 + 'Phase2, train on train_data + dev_data, n_epoch={}'.format(best_epoch) + '='*10)
    print('\n'*2)


    # Re-initialize the model for training on phase 2
    ner_model = BiLSTM(word_emb, pos_emb, args.word_hid_dim, max(list(label2idx.values()))+1, args.dropout, args.batch, trainable_emb = args.trainable_emb)
    ner_model.rand_init()
    criterion = nn.NLLLoss()
    if args.cuda:
        ner_model.cuda()
    if args.opt == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, ner_model.parameters()), lr=args.lr)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, ner_model.parameters()), lr=args.lr, momentum=args.momentum)
    predictor = Predictor()
    evaluator = Evaluator(predictor, label2idx, word2idx, pos2idx, args)
    # begin phase 2    
    for epoch in range(1, best_epoch+1):
        
        loss = train_epoch(train_data + dev_data, ner_model, optimizer, criterion, args)
        print("*"*10 + "epoch:{}, loss:{}".format(epoch, loss) + "*"*10)
        eval_result_train_dev = evaluator.evaluate(train_data + dev_data, ner_model, args, cuda = args.cuda)
        print("On train_data + dev_data: ")
        print_scores(eval_result_train_dev, args)
    print('\n'*2)
    eval_result_test = evaluator.evaluate(test_data, ner_model, args, cuda = args.cuda)
    print("On test_data: ")
    print_scores(eval_result_test, args)
    best_scores = [loss, eval_result_train_dev, eval_result_test]
    save_model(epoch, args, ner_model, best_scores, optimizer,
                       args.save_model + '/epoch{}_test_TF1_{}_EF1_{}.model'.format(epoch, eval_result_test[0][0], eval_result_test[0][1]))
    print('Model saved as:', args.save_model + '/epoch{}_test_TF1_{}_EF1_{}.model'.format(epoch, eval_result_test[0][0], eval_result_test[0][1]))
        # print("Testing after epoch{}...".format(epoch))
        # tokens = Variable(torch.LongTensor(np.array([train_data[0][1][0][7]]).transpose()))
        # pos_tags = Variable(torch.LongTensor(np.array([train_data[0][1][1][7]]).transpose()))
        # score, pred_list = predictor.predict([tokens, pos_tags], ner_model)
        # print(score)
        # print("doc id: ", train_data[0][0][0])
        # print("tokens: ", train_data[0][0][1][7])
        # print("y_true: ", train_data[0][1][2][7])
        # print("y_preds: ", pred_list)

    # decoder = CRFDecode_vb(len(label2idx), label2idx['<start>'], label2idx['<end>'])
    # predictor = Predictor(decoder)
    # evaluator = Evaluator(predictor, label2idx, word2idx, pos2idx)
    # trainer = Trainer(criterion, optimizer, evaluator, label2idx, word2idx, pos2idx)
    
    
    # print("\n\n"+"*"*40+" Start Training: EXP1 "+"*"*40+"\n")
    # best_epoch = trainer.train([train_data, dev_data, test_data], ner_model, args.epochs, True, args)
    
    # ner_model = LSTM_CRF(word_emb, pos_emb, len(label2idx), args.word_hid, args.dropout, args.batch)
    # ner_model.rand_init()
    # if args.opt == 'adam':
    #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, ner_model.parameters()), lr=args.lr)
    # else:
    #     optimizer = optim.SGD(filter(lambda p: p.requires_grad, ner_model.parameters()), lr=args.lr, momentum=args.momentum)
    
    
    # trainer = Trainer(criterion, optimizer, evaluator, label2idx, word2idx, pos2idx)
    # print("\n\n"+"*"*40+" Start Training: EXP2 "+"*"*40+"\n")
    # best_epoch = trainer.train([train_data+dev_data, None, test_data], ner_model, best_epoch, False, args)
