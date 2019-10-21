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
from utilsXML import GEDoc
import utils
from collections import OrderedDict
import pdb
import argparse
import tqdm
from LSTM import BiLSTM
from predictor import Predictor
from evaluator import Evaluator

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


def train_epoch(data_train, model, optimizer, criterion, args):
    model.train()
    epoch_loss = 0
    for doc_train in tqdm.tqdm(data_train):
        for tokens, pos_tags, labels in zip(doc_train.sents, doc_train.pos_tags, doc_train.token_labels):
            assert len(tokens) == len(labels)
            # pdb.set_trace()
            tokens = [args.word2idx[i] for i in tokens]
            pos_tags = [args.pos2idx[i] for i in pos_tags]
            labels = [args.label2idx[i] for i in labels]
            y_true = labels

            # pdb.set_trace()
            tokens = Variable(torch.LongTensor(np.array([tokens]).transpose()))
            pos_tags = Variable(torch.LongTensor(np.array([pos_tags]).transpose()))
            labels = Variable(torch.LongTensor(np.array(labels).transpose()))     # labels have to be one-dim for NLL loss
            if args.cuda:
                tokens, pos_tags, labels = [tokens.cuda(), pos_tags.cuda(), labels.cuda()]
            scores = model(tokens, pos_tags)
            # print(scores)
            # print(labels)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += utils.to_scalar(loss)
    print('epoch loss:{}'.format(epoch_loss))



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hid_dim', type=int, default=60)
    p.add_argument('--n_epoch', type=int, default=20)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--trainable_emb', action='store_true')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--lr', type=float, default=0.0005)
    p.add_argument('--opt', choices=['sgd', 'adam'], default='adam')
    p.add_argument('--random_seed', type=int, default=27)
    p.add_argument('--save_model_dir', type=str, default='./trigger_models')
    p.add_argument('--patience', type=int, default=1)

    args = p.parse_args()


    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    data_train = pickle.load(open('GE09-train.pkl', 'rb'))
    data_dev = pickle.load(open('GE09-devel.pkl', 'rb'))

    all_data = data_train + data_dev

    all_tokens = np.concatenate(np.concatenate([doc.sents for doc in all_data]))
    all_tokens = list(set(all_tokens))
    word_list = ['UNK'] + all_tokens
    word2idx = OrderedDict(zip(word_list, range(len(word_list))))
    args.word2idx = word2idx
    # w2v_emb = read_w2v_emb(word2idx, 'wikipedia-pubmed-and-PMC-w2v.bin')
    w2v_emb = np.load('w2v_emb.npy')

    all_pos = np.concatenate(np.concatenate([doc.pos_tags for doc in all_data]))
    all_pos = list(set(all_pos))
    pos_list = ['UNK'] + all_pos
    pos2idx = OrderedDict(zip(pos_list, range(len(pos_list))))
    args.pos2idx = pos2idx
    # pos_emb= np.zeros((len(pos2idx), len(pos2idx)))
    # for i in range(pos_emb.shape[0]):
    #     pos_emb[i, i] = 1.0
    # pdb.set_trace()
    pos_emb = np.load('pos_emb.npy')
    all_labels = np.concatenate(np.concatenate([doc.token_labels for doc in all_data]))
    all_labels = list(set(all_labels))
    label2idx = OrderedDict(zip(all_labels, range(len(all_labels))))
    args.label2idx = label2idx

    model = BiLSTM(w2v_emb, pos_emb, args.hid_dim, len(label2idx), args.dropout, args.batch, trainable_emb = args.trainable_emb)
    model.rand_init()
    criterion = nn.NLLLoss()
    if args.cuda:
        model.cuda()
    if args.opt == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum)
    predictor = Predictor()
    evaluator = Evaluator(predictor, args.label2idx, args.word2idx, args.pos2idx, args)

    patience = 0
    best_score = 0
    best_epoch = 0
    for epoch in range(1, args.n_epoch+1):
        print('*'*10 + 'epoch {}'.format(epoch) + '*'*10)
        train_epoch(data_train+data_dev, model, optimizer, criterion, args)

        args.final_eval = 1

        # f1_train = evaluator.evaluate(data_train+data_dev, model, args, cuda=True )
        # print('train micro f1: {}'.format(f1_train))

        # f1_dev = evaluator.evaluate(data_train+data_dev, model, args, cuda=True )
        # print('dev micro f1: {}'.format(f1_dev))
        prec, recall, f1, support = evaluator.evaluate(data_train+data_dev, model, args, cuda=True )
        idx2label = OrderedDict([(v, k) for k, v in args.label2idx.items()])
        for i in range(len(prec)):
            print("Trigger type {}, prec {}, recall {}, f1 {}, support {}".format(idx2label[i], prec[i], recall[i], f1[i], support[i]))
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
