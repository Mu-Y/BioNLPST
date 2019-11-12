import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import utils
import argparse
import pdb
import pickle
from collections import OrderedDict
import os
import numpy as np
# from model.predictor import Predictor
# from model.evaluator import Evaluator

def create_emb_layer(weights_matrix, trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.weight = Parameter(torch.FloatTensor(weights_matrix))
    if not trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer

class BiLSTM(nn.Module):

    def __init__(self, word_emb, pos_emb, hid_dim, dropout, batch_size, trainable_emb = False,
                 n_layers=1,  triggerset_dim=None, interactionset_dim=None):
        super(BiLSTM, self).__init__()

        vocab_size = len(word_emb)
        word_emb_dim = len(word_emb[0])
        pos_emb_dim = len(pos_emb)

        self.hid_dim = hid_dim

        self.word_emb = create_emb_layer(word_emb, trainable = trainable_emb)
        self.pos_emb = create_emb_layer(pos_emb, trainable = trainable_emb)


        self.word_lstm = nn.LSTM(word_emb_dim + pos_emb_dim, hid_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        # self.MLP = MLP
        # self.tagset_dim = tagset_dim

        # if self.MLP:
        #     self.linear_layer = nn.Linear(hid_dim*2, tagset_dim)  # bi-directional
        # if self.multitask:

        # MLP layer for trigger classificatoin
        self.linear_layer_trigger = nn.Linear(hid_dim*2, triggerset_dim)
        # MLP layer for interaction classificaiton
        self.linear_layer_interaction = nn.Linear(hid_dim*4, interactionset_dim)

    def create_emb_layer(self, weights_matrix, trainable=False):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if not trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, num_embeddings, embedding_dim

    def rand_init(self):
        """
        random initialization
        args:
            init_char_embedding: random initialize char embedding or not
            init_word_embedding: random initialize word embedding or not
        """
        utils.init_lstm(self.word_lstm)
        utils.init_linear(self.linear_layer_trigger)
        utils.init_linear(self.linear_layer_interaction)
        # self.crf.rand_init()
    def forward(self, tokens, pos_tags, pair_idxs, task):
        '''
        args:
            tokens: A sentence.(token_idx?)
            pos_tags: The pos_tags for the sentence.(pos_idx?)
            pair_idxs: list of pair index tuple (l_idx, r_idx)
        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''

        #word
        word_emb = self.word_emb(tokens)
        d_word_emb = self.dropout(word_emb)

        #pos
        pos_emb = self.pos_emb(pos_tags)

        #word level lstm
        lstm_out, _ = self.word_lstm(torch.cat((d_word_emb, pos_emb), dim=2))
        d_lstm_out = self.dropout(lstm_out) # (word_seq_len, batch_size, hid_size)

        # if self.MLP:
        if task == 'trigger':
            out = self.linear_layer_trigger(F.relu(d_lstm_out.view(len(tokens), -1)))
            scores = F.log_softmax(out, dim=1)
            # pdb.set_trace()
        if task == 'interaction':
            # TODO: finish the interaction constructer, then finish this
            # assert len(l_idx) == len(r_idx)
            # unsqueeze is to retain the batch_size dim, o.w. this dim will be lost by splicing
            ltar_f = torch.cat([d_lstm_out[l, :, :self.hid_dim].unsqueeze(1) for (l, r) in pair_idxs], dim=0)
            ltar_b = torch.cat([d_lstm_out[l, :, self.hid_dim:].unsqueeze(1) for (l, r) in pair_idxs], dim=0)
            rtar_f = torch.cat([d_lstm_out[r, :, :self.hid_dim].unsqueeze(1) for (l, r) in pair_idxs], dim=0)
            rtar_b = torch.cat([d_lstm_out[r, :, self.hid_dim:].unsqueeze(1) for (l, r) in pair_idxs], dim=0)

            # pdb.set_trace()

            out = torch.cat((ltar_f, ltar_b, rtar_f, rtar_b), dim=2)
            out = self.dropout(out)



            # ltar_f = torch.cat([d_lstm_out[b, lidx_start[b][r], :self.hid_size].unsqueeze(0) for b,r in rel_idxs], dim=0)
            # ltar_b = torch.cat([d_lstm_out[b, lidx_end[b][r], self.hid_size:].unsqueeze(0) for b,r in rel_idxs], dim=0)
            # rtar_f = torch.cat([d_lstm_out[b, ridx_start[b][r], :self.hid_size].unsqueeze(0) for b,r in rel_idxs], dim=0)
            # rtar_b = torch.cat([d_lstm_out[b, ridx_end[b][r], self.hid_size:].unsqueeze(0) for b,r in rel_idxs], dim=0)
            assert out.shape[0] == len(pair_idxs)
            out = self.linear_layer_interaction(F.relu(out.view(len(pair_idxs), -1)))
            # pdb.set_trace()
            scores = F.log_softmax(out, dim=1)
            # pdb.set_trace()
        # print(tag_scores)


        return scores


