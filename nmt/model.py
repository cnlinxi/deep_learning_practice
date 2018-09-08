# -*- coding: utf-8 -*-
# @Time    : 2018/8/29 11:54
# @Author  : MengnanChen
# @FileName: model.py
# @Software: PyCharm Community Edition

from keras.models import Sequential
from keras.layers import Embedding,LSTM,RepeatVector,TimeDistributed,Dense

class NMT:
    def __init__(self,src_vocab,tar_vocab,src_timesteps,tar_timesteps,n_units):
        self._src_vocab=src_vocab
        self._tar_vocab=tar_vocab
        self._src_timesteps=src_timesteps
        self._tar_timesteps=tar_timesteps
        self._n_units=n_units

    def create_model(self):
        model=Sequential()
        model.add(Embedding(self._src_vocab,self._n_units,input_length=self._src_timesteps,mask_zero=True))
        model.add(LSTM(self._n_units))
        model.add(RepeatVector(self._tar_timesteps))
        model.add(LSTM(self._n_units,return_sequences=True))
        model.add(TimeDistributed(Dense(self._tar_vocab,activation='softmax')))
        return model