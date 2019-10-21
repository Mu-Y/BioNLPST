import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import model.utils as utils
import argparse
import pdb
import pickle
from testAnn import DataProcessed
from collections import OrderedDict
import os
import numpy as np
from model.predictor import Predictor
from model.evaluator import Evaluator

def create_emb_layer(weights_matrix, trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.weight = Parameter(torch.FloatTensor(weights_matrix))
    if not trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer

class BiLSTM(nn.Module):

    def __init__(self, word_emb, pos_emb, hid_dim, tagset_dim, dropout, batch_size, trainable_emb = False, MLP = True):
        super(BiLSTM, self).__init__()

        vocab_size = len(word_emb)
        word_emb_dim = len(word_emb[0])
        pos_emb_dim = len(pos_emb)

        self.word_emb = create_emb_layer(word_emb, trainable = trainable_emb)
        self.pos_emb = create_emb_layer(pos_emb, trainable = trainable_emb)


        self.word_lstm = nn.LSTM(word_emb_dim + pos_emb_dim, hid_dim, num_layers=1, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.MLP = MLP
        self.tagset_dim = tagset_dim

        if self.MLP:
            self.linear_layer = nn.Linear(hid_dim*2, tagset_dim)  # bi-directional

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
        utils.init_linear(self.linear_layer)
        # self.crf.rand_init()
    def forward(self, tokens, pos_tags):
        '''
        args:
            tokens: A sentence.(token_idx?)
            pos_tags: The pos_tags for the sentence.(pos_idx?)
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
        d_lstm_out = self.dropout(lstm_out)

        if self.MLP:
            d_lstm_out = self.linear_layer(F.relu(d_lstm_out.view(len(tokens), -1)))

        tag_scores = F.log_softmax(d_lstm_out, dim=1)
        # print(tag_scores)


        return tag_scores


