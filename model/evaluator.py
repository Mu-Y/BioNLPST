import torch
import numpy as np
import itertools
from torch.autograd import Variable
from copy import deepcopy
from collections import OrderedDict
import pdb


class Evaluator:
    
    def __init__(self, predictor, label2idx, word2idx, pos2idx, args):
        self.predictor = predictor
        label2idx = deepcopy(label2idx)
        
        if args.eval_separate_VN:
            label2idx = OrderedDict([('O', 0),
                                  ('B_EVENT_VERB', 1), ('I_EVENT_VERB', 2),
                                  ('B_EVENT_NOUN', 3), ('I_EVENT_NOUN', 4),
                                  ('B_EVENT_OTHER', 3),('I_EVENT_OTHER',4),
                                  ('B_EVENT_ADJ', 3), ('I_EVENT_ADJ',4),
                                  ('B_EVENT_PREP', 3), ('I_EVENT_PREP', 4),
                                  ('B_TIMEX', 5), ('I_TIMEX', 6)])
        elif args.eval_separate_all:
            label2idx = OrderedDict([('O', 0),
                                     ('B_EVENT_VERB', 1), ('I_EVENT_VERB', 2),
                                     ('B_EVENT_NOUN', 3), ('I_EVENT_NOUN', 4),
                                     ('B_EVENT_OTHER', 5),('I_EVENT_OTHER', 6),
                                     ('B_EVENT_ADJ', 7), ('I_EVENT_ADJ', 8),
                                     ('B_EVENT_PREP', 9), ('I_EVENT_PREP', 10),
                                     ('B_TIMEX', 11), ('I_TIMEX', 12)])
        else:
            label2idx = OrderedDict([('O', 0),
                                     ('B_EVENT_VERB', 1), ('I_EVENT_VERB', 2),
                                     ('B_EVENT_NOUN', 1), ('I_EVENT_NOUN',2),
                                     ('B_EVENT_OTHER', 1),('I_EVENT_OTHER',2),
                                     ('B_EVENT_ADJ', 1), ('I_EVENT_ADJ',2),
                                     ('B_EVENT_PREP', 1), ('I_EVENT_PREP', 2),
                                     ('B_TIMEX', 3), ('I_TIMEX', 4)])

        self.label2idx = label2idx
        self.word2idx = word2idx
        self.pos2idx = pos2idx
        self.O = self.label2idx['O']
        if args.ignore_TIMEX:
            types = [r[2:] for r in self.label2idx if r.startswith('B_EVENT') ]
            label2idx['B_TIMEX'] = 0
            label2idx['I_TIMEX'] = 0
        else:
            types = [r[2:] for r in self.label2idx if r.startswith('B') ]
        self.B2I = {self.label2idx['B_'+r]: self.label2idx['I_'+r] for r in types}
    
    def iob_to_obj(self, y):
        obj = []
        in_obj = False
        curr_obj = []
        curr_I = None
        for i in range(len(y)):
            # end of obj
            if in_obj and y[i] != curr_I:
                obj.append(tuple(curr_obj + [i-1]))
                curr_obj = []
                curr_I = None
                in_obj = False
            # beginning of obj
            if y[i] in self.B2I:
                assert y[i] in [self.label2idx[r] for r in self.label2idx if r[0]=='B']
                curr_obj = [y[i], i]
                curr_I = self.B2I[y[i]]
                in_obj = True
        
        return obj
    def calc_prec_recall_f1(self, TP_time, num_pred_time, num_true_time):
        prec_time = TP_time / num_pred_time if num_pred_time != 0 else 0
        recall_time = TP_time / num_true_time if num_true_time != 0 else 0
        f1_time = 2 * prec_time * recall_time / (prec_time + recall_time) if prec_time + recall_time != 0 else 0
        return f1_time, prec_time, recall_time
    
    def evaluate(self, data, ner_model, args, cuda=True):
        """
        target_label: when args.eval_separate_all, {TIMEX, EVENT_VERB, EVENT_NOUN, EVENT_ADJ, EVENT_PREP, EVENT_OTHER}
                      when args.eval_separate_VN, {TIMEX, EVENT_V, EVENT_N}
        """
        ner_model.eval()
        num_pred_time_entities, num_true_time_entities, num_TP_time_entities = 0, 0, 0
        num_TP_entities, num_pred_entities, num_true_entities = 0, 0, 0
        TP_time, num_pred_time, num_true_time = 0, 0, 0
        TP_event, num_pred_event, num_true_event = 0, 0, 0
        TP_event_V, num_pred_event_V, num_true_event_V = 0, 0, 0
        TP_event_N, num_pred_event_N, num_true_event_N = 0, 0, 0
        TP_event_VERB, num_pred_event_VERB, num_true_event_VERB = 0, 0, 0
        TP_event_NOUN, num_pred_event_NOUN, num_true_event_NOUN = 0, 0, 0
        TP_event_OTHER, num_pred_event_OTHER, num_true_event_OTHER = 0, 0, 0
        TP_event_ADJ, num_pred_event_ADJ, num_true_event_ADJ = 0, 0, 0
        TP_event_PREP, num_pred_event_PREP, num_true_event_PREP = 0, 0, 0
        # pdb.set_trace()

        for d in data:
            input_d = d[1]
            for tokens, pos_tags, labels, GT_labels, words, pos in zip(input_d[0], input_d[1], input_d[2], d[0][3], d[0][1], d[0][2]):
                
                # labels are used just for training, e.g. [0,1,2] when not train_separate_VN and eval_separate_VN
                # GT_labels are GLOBALLY true label from raw data, e.g. [0,1,2,3,4] when not train_separate_VN and eval_separate_VN
                # y_true = labels
                y_true = [self.label2idx[i] for i in GT_labels]
                #  assert y_true == labels  # check that labels used for training is the same as labels used for evaluating
                # label_for_crf = [self.label2idx['<start>']] + labels + [self.label2idx['<end>']]
                # label_for_crf = [label_for_crf[i] * len(self.label2idx) + label_for_crf[i+1] for i in range(len(label_for_crf)-1)]
                # tokens = tokens + [self.word2idx['<end>']]
                # pos_tags = pos_tags + [self.pos2idx['<end>']]
                tokens = Variable(torch.LongTensor(np.array([tokens]).transpose()))
                pos_tags = Variable(torch.LongTensor(np.array([pos_tags]).transpose()))
                labels = Variable(torch.LongTensor(np.array(labels).transpose()))     # labels have to be one-dim for NLL loss
                
                if cuda:
                    tokens, pos_tags, labels = [tokens.cuda(), pos_tags.cuda(), labels.cuda()]
                
                _, y_pred = self.predictor.predict([tokens, pos_tags], ner_model)

                if (not args.train_separate_VN) and args.eval_separate_VN:
                    # Meaning that predicted label space is ['O', 'B/I_EVENT']([0, 1, 2]), 
                    # but GT label space is ['0', 'B/I_EVENT_V', 'B/I_EVENT_N']([0, 1, 2, 3, 4])
                    #post process y_pred with the token pos_tags

                    for i in range(len(y_pred)):
                        if y_pred[i] == 0:
                            # label at this position is 'O', don't modify
                            continue
                        # Handle TIMEX labels
                        elif y_pred[i] == 3:
                            # print('handled B_TIMEX')
                            y_pred[i] = 5 
                        elif y_pred[i] == 4:
                            # print('handled I_TIMEX')
                            y_pred[i] = 6
                        else:
                            # Handle EVENT_V labels 
                            if pos_tags[i].item() in [self.pos2idx[r] for r in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]:
                                # print('set to verb event')
                                # Verb EVENTs, set to 1 or 2
                                y_pred[i] = 1 if y_pred[i] == 1 else 2
                            # Handle EVENT_N labels
                            else: 
                                # print('set to NOMINAL event')
                                # Nominal EVENTS, set to 3 or 4
                                y_pred[i] = 3 if y_pred[i] == 1 else 4
                pred_obj = self.iob_to_obj(y_pred)
                true_obj = self.iob_to_obj(y_true)
                TP = [r for r in pred_obj if r in true_obj]

                pred_entities = [(r[1], r[2]) for r in pred_obj]
                true_entities = [(r[1], r[2]) for r in true_obj]
                TP_entities = [r for r in pred_entities if r in true_entities] 
                
                pred_time_entities = [(r[1], r[2]) for r in pred_obj if r[0]==self.label2idx['B_TIMEX']]
                true_time_entities = [(r[1], r[2]) for r in true_obj if r[0]==self.label2idx['B_TIMEX']]
                TP_time_entities = [r for r in pred_time_entities if r in true_time_entities] 


                TP_time += len([r for r in TP if r[0]==self.label2idx['B_TIMEX']])
                num_pred_time += len([r for r in pred_obj if r[0]==self.label2idx['B_TIMEX']])
                num_true_time += len([r for r in true_obj if r[0]==self.label2idx['B_TIMEX']])
                if args.eval_separate_all:
                    TP_event_VERB += len([r for r in TP if r[0]==self.label2idx['B_EVENT_VERB']]) 
                    num_pred_event_VERB += len([r for r in pred_obj if r[0]==self.label2idx['B_EVENT_VERB']])
                    num_true_event_VERB += len([r for r in true_obj if r[0]==self.label2idx['B_EVENT_VERB']])
                    TP_event_NOUN += len([r for r in TP if r[0]==self.label2idx['B_EVENT_NOUN']]) 
                    num_pred_event_NOUN += len([r for r in pred_obj if r[0]==self.label2idx['B_EVENT_NOUN']])
                    num_true_event_NOUN += len([r for r in true_obj if r[0]==self.label2idx['B_EVENT_NOUN']])
                    TP_event_ADJ += len([r for r in TP if r[0]==self.label2idx['B_EVENT_ADJ']]) 
                    num_pred_event_ADJ += len([r for r in pred_obj if r[0]==self.label2idx['B_EVENT_ADJ']])
                    num_true_event_ADJ += len([r for r in true_obj if r[0]==self.label2idx['B_EVENT_ADJ']])
                    TP_event_OTHER += len([r for r in TP if r[0]==self.label2idx['B_EVENT_OTHER']]) 
                    num_pred_event_OTHER += len([r for r in pred_obj if r[0]==self.label2idx['B_EVENT_OTHER']])
                    num_true_event_OTHER += len([r for r in true_obj if r[0]==self.label2idx['B_EVENT_OTHER']])
                    TP_event_PREP += len([r for r in TP if r[0]==self.label2idx['B_EVENT_PREP']]) 
                    num_pred_event_PREP += len([r for r in pred_obj if r[0]==self.label2idx['B_EVENT_PREP']])
                    num_true_event_PREP += len([r for r in true_obj if r[0]==self.label2idx['B_EVENT_PREP']])
                elif args.eval_separate_VN:
                    
                    # TP_event_V += len([r for r in TP if r[0]==self.label2idx['B_EVENT_VERB']]) 
                    # num_pred_event_V += len([r for r in pred_obj if r[0]==self.label2idx['B_EVENT_VERB']])
                    # num_true_event_V += len([r for r in true_obj if r[0]==self.label2idx['B_EVENT_VERB']])
                    # # Note: here EVENT_NOUN, ADJ, OTHER, PREP are all mapped to the same index
                    # # So only counting B_EVENT_NOUN will be sufficient
                    # TP_event_N += len([r for r in TP if r[0]==self.label2idx['B_EVENT_NOUN']]) 
                    # num_pred_event_N += len([r for r in pred_obj if r[0]==self.label2idx['B_EVENT_NOUN']])
                    # num_true_event_N += len([r for r in true_obj if r[0]==self.label2idx['B_EVENT_NOUN']])

                    TP_event_V += len([r for r in TP if r[0]==self.label2idx['B_EVENT_VERB']]) 
                    TP_event_N += len([r for r in TP if r[0]==self.label2idx['B_EVENT_NOUN']]) 
                   

                    # Note: here EVENT_NOUN, ADJ, OTHER, PREP are all mapped to the same index
                    # So only counting B_EVENT_NOUN will be sufficient
                    num_pred_event_V += len([r for r in pred_obj if r[0]==self.label2idx['B_EVENT_VERB']])
                    num_pred_event_N += len([r for r in pred_obj if r[0]==self.label2idx['B_EVENT_NOUN']])
                    
                    num_true_event_V += len([r for r in true_obj if r[0]==self.label2idx['B_EVENT_VERB']])
                    num_true_event_N += len([r for r in true_obj if r[0]==self.label2idx['B_EVENT_NOUN']])
                    
                    assert  len([r for r in pred_obj if r[0]==self.label2idx['B_EVENT_VERB']]) + \
                            len([r for r in pred_obj if r[0]==self.label2idx['B_EVENT_NOUN']]) + \
                            len([r for r in pred_obj if r[0]==self.label2idx['B_TIMEX']])    == \
                            len(pred_obj)
                    assert  len([r for r in true_obj if r[0]==self.label2idx['B_EVENT_VERB']]) + \
                            len([r for r in true_obj if r[0]==self.label2idx['B_EVENT_NOUN']]) + \
                            len([r for r in true_obj if r[0]==self.label2idx['B_TIMEX']])== \
                            len(true_obj)
                    assert  len([r for r in TP if r[0]==self.label2idx['B_EVENT_VERB']]) + \
                            len([r for r in TP if r[0]==self.label2idx['B_EVENT_NOUN']]) + \
                            len([r for r in TP if r[0]==self.label2idx['B_TIMEX']])== \
                            len(TP)

                    num_pred_entities += len(pred_entities)
                    num_true_entities += len(true_entities)
                    num_TP_entities += len(TP_entities)

                    num_pred_time_entities += len(pred_time_entities)
                    num_true_time_entities += len(true_time_entities)
                    num_TP_time_entities += len(TP_time_entities)
                else:
                    TP_event += len([r for r in TP if r[0]==self.label2idx['B_EVENT_VERB']])
                    num_pred_event += len([r for r in pred_obj if r[0]==self.label2idx['B_EVENT_VERB']])
                    num_true_event += len([r for r in true_obj if r[0]==self.label2idx['B_EVENT_VERB']])

                # pdb.set_trace()   ############# For error analysis
                # print('tokens: ', words)
                # print('y_true: ', y_true)
                # print('y_pred: ', y_pred)
        f1_time, prec_time, recall_time= self.calc_prec_recall_f1(TP_time, num_pred_time, num_true_time)

        if args.eval_separate_all:
            f1_event_VERB, prec_event_VERB, recall_event_VERB = self.calc_prec_recall_f1(TP_event_VERB, num_pred_event_VERB, num_true_event_VERB)
            f1_event_NOUN, prec_event_NOUN, recall_event_NOUN = self.calc_prec_recall_f1(TP_event_NOUN, num_pred_event_NOUN, num_true_event_NOUN)
            f1_event_OTHER, prec_event_OTHER, recall_event_OTHER = self.calc_prec_recall_f1(TP_event_OTHER, num_pred_event_OTHER, num_true_event_OTHER)
            f1_event_ADJ, prec_event_ADJ, recall_event_ADJ = self.calc_prec_recall_f1(TP_event_ADJ, num_pred_event_ADJ, num_true_event_ADJ)
            f1_event_PREP, prec_event_PREP, recall_event_PREP = self.calc_prec_recall_f1(TP_event_PREP, num_pred_event_PREP, num_true_event_PREP)
            f1_event, prec_event, recall_event = self.calc_prec_recall_f1(TP_event_VERB + TP_event_NOUN + TP_event_ADJ + TP_event_OTHER + TP_event_PREP, \
                                                                     num_pred_event_VERB + num_pred_event_NOUN + num_pred_event_ADJ + num_pred_event_OTHER + num_pred_event_PREP, \
                                                                     num_true_event_VERB + num_true_event_NOUN + num_true_event_ADJ + num_true_event_OTHER + num_true_event_PREP)
            return ([f1_time, f1_event, f1_event_VERB, f1_event_NOUN, f1_event_OTHER, f1_event_ADJ, f1_event_PREP], \
                    [prec_time, prec_event, prec_event_VERB, prec_event_NOUN, prec_event_OTHER, f1_event_ADJ, f1_event_PREP], \
                    [recall_time, recall_event, recall_event_VERB, recall_event_NOUN, recall_event_OTHER, recall_event_ADJ, recall_event_PREP])
                    
        elif args.eval_separate_VN:
            TP_event = TP_event_V + TP_event_N
            num_pred_event = num_pred_event_V + num_pred_event_N
            num_true_event = num_true_event_V + num_true_event_N

            f1_entities,prec_entities, recall_entities = self.calc_prec_recall_f1(num_TP_entities, \
                                                                                  num_pred_entities, \
                                                                                  num_true_entities)
            f1_Attr, prec_Attr, recall_Attr = self.calc_prec_recall_f1(TP_event + TP_time, \
                                                                     num_pred_entities, \
                                                                     num_true_entities)
            print('TP_event:{}, TP_time:{}, num_pred_entities{}, num_true_entities:{}'.format(TP_event, TP_time, num_pred_entities, num_true_entities))
            print('num_pred_time:{}, num_true_time:{}, num_pred_event{}, num_true_event:{}'.format(num_pred_time, num_true_time, num_pred_event, num_true_event))     
            print('pred_time_entities:{}, true_time_entities:{}, TP_time_entities:{}'.format(num_pred_time_entities, num_true_time_entities, num_TP_time_entities))
            f1_event, prec_event, recall_event = self.calc_prec_recall_f1(TP_event, \
                                                                     num_pred_event, \
                                                                     num_true_event)
            f1_event_V, prec_event_V, recall_event_V = self.calc_prec_recall_f1(TP_event_V, num_pred_event_V, num_true_event_V)
            f1_event_N, prec_event_N, recall_event_N = self.calc_prec_recall_f1(TP_event_N, num_pred_event_N, num_true_event_N)

            
            Acc_Attr = f1_Attr / f1_entities if f1_entities != 0 else 0
            return ([f1_time, f1_event, f1_event_V, f1_event_N, f1_entities, f1_Attr, Acc_Attr], \
                    [prec_time, prec_event, prec_event_V, prec_event_N, prec_entities, prec_Attr], \
                    [recall_time, recall_event, recall_event_V, recall_event_N, recall_entities, recall_Attr])         
           
        else:
            f1_event, prec_event, recall_event = self.calc_prec_recall_f1(TP_event , \
                                                                     num_pred_event, \
                                                                     num_true_event)
            return ([f1_time, f1_event], \
                    [prec_time, prec_event], \
                    [recall_time, recall_event])  

            
                


        
        

