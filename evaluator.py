import torch
import numpy as np
import itertools
from torch.autograd import Variable
from copy import deepcopy
from collections import OrderedDict
from sklearn.metrics import f1_score, precision_recall_fscore_support
import pdb


class Evaluator:

    def __init__(self, predictor, label2idx, word2idx, pos2idx, args):
        self.predictor = predictor
        label2idx = deepcopy(label2idx)
        self.label2idx = label2idx
        self.word2idx = word2idx
        self.pos2idx = pos2idx
        self.idx2label = OrderedDict([(v, k) for k, v in self.label2idx.items()])
        # self.O = self.label2idx['O']
        # if args.ignore_TIMEX:
        #     types = [r[2:] for r in self.label2idx if r.startswith('B_EVENT') ]
        #     label2idx['B_TIMEX'] = 0
        #     label2idx['I_TIMEX'] = 0
        # else:
        #     types = [r[2:] for r in self.label2idx if r.startswith('B') ]
        # self.B2I = {self.label2idx['B_'+r]: self.label2idx['I_'+r] for r in types}

    def calc_prec_recall_f1(self, TP_time, num_pred_time, num_true_time):
        prec_time = TP_time / num_pred_time if num_pred_time != 0 else 0
        recall_time = TP_time / num_true_time if num_true_time != 0 else 0
        f1_time = 2 * prec_time * recall_time / (prec_time + recall_time) if prec_time + recall_time != 0 else 0
        return f1_time, prec_time, recall_time

    def evaluate(self, data, model, args, cuda=True):
        """
        target_label: when args.eval_separate_all, {TIMEX, EVENT_VERB, EVENT_NOUN, EVENT_ADJ, EVENT_PREP, EVENT_OTHER}
                      when args.eval_separate_VN, {TIMEX, EVENT_V, EVENT_N}
        """
        model.eval()

        y_trues = []
        y_preds = []
        for d in data:
            # input_d = d[1]
            for tokens, pos_tags, labels in zip(d.sents, d.pos_tags, d.token_labels):
            # for tokens, pos_tags, labels, GT_labels, words, pos in zip(input_d[0], input_d[1], input_d[2], d[0][3], d[0][1], d[0][2]):
                tokens = [self.word2idx[i] for i in tokens]
                pos_tags = [self.pos2idx[i] for i in pos_tags]
                y_true = [self.label2idx[i] for i in labels]
                y_trues.append(y_true)
                tokens = Variable(torch.LongTensor(np.array([tokens]).transpose()))
                pos_tags = Variable(torch.LongTensor(np.array([pos_tags]).transpose()))

                if cuda:
                    tokens, pos_tags = [tokens.cuda(), pos_tags.cuda()]
                _, y_pred = self.predictor.predict([tokens, pos_tags], model)

                # pdb.set_trace()
                assert len(y_true) == len(y_pred)
                y_preds.append(y_pred)
        y_trues = np.concatenate(y_trues)
        y_preds = np.concatenate(y_preds)

        if args.final_eval == 0:
            f1 = f1_score(y_true=y_trues, y_pred=y_preds, average='micro')
            # pdb.set_trace()
            return f1
        elif args.final_eval == 1:
            # for final eval, return a full report for prec, recall, f1, support
            prec, recall, f1, support = precision_recall_fscore_support(y_trues, y_preds, average=None)
            return prec, recall, f1, support


