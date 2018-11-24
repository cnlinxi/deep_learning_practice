# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 16:02
# @Author  : MengnanChen
# @FileName: text.py
# @Software: PyCharm

import re

import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from . import hparams


class TextProcess:
    def __init__(self, w2v_path):
        self._w2v_path = w2v_path

        self.w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=False,
                                                           unicode_errors='ignore',
                                                           limit=10000)  # only read 10000 word
        self.un_vec = np.random.uniform(-0.1, 0.1, size=hparams.embedding_size)
        self.un_index = self.w2v_model.vocab[list(self.w2v_model.vocab.keys())[-1]].index + 1
        self.vocab_size = self.un_index + 1  # index start from 0

        self.word2index = {token: token_index for token_index, token in enumerate(self.w2v_model.index2word)}
        self.index2word = self.w2v_model.index2word
        self.w2v_array = np.concatenate((self.w2v_model.wv.syn0, [self.un_vec]))

    def _text_to_ids(self, text):
        ids = []
        for word in text:
            if word in self.w2v_model.vocab:
                ids.append(self.word2index[word])
            else:
                ids.append(self.un_index)
        return ids

    def text_to_sequence(self, text, cleaner_names=''):
        ### clean data
        return self._text_to_ids(text)

    def sequence_to_text(self, sequence):
        text = [self.index2word[x] for x in sequence]
        return ''.join(text)
