#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/18 22:20
# @Author  : MengnanChen
# @Site    : 
# @File    : tacotron.py
# @Software: PyCharm Community Edition


import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from text.symbols import symbols

class Tacotron():
    def __init__(self,hparam):
        self._hparam=hparam

    def initialize(self,inputs,input_lengths,mel_targets=None,linear_targets=None):
        '''
        初始化模型以进行inference
        :param inputs:[N,T_in], 其中N为batch_size, T_in是输入时间序列的步数，张量中的值为字符的id
        :param input_lengths:[N], 其中N为batch_size, 张量中的值为每个输入序列的长度
        :param mel_targets:[N,T_out,M]， 其中N为batch_size,T_out为输出序列的步数,M为num_mels，张量中的值为
        :param linear_targets:
        :return:
        '''
        with tf.variable_scope('inference') as scope:
            is_training=linear_targets is not None
            batch_size=tf.shape(inputs)[0]
            hp=self._hparam

            embedding_table=tf.get_variable(
                'embedding',[len(symbols),hp.embed_depth],dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.5)
            )
            embedded_inputs=tf.nn.embedding_lookup(embedding_table,inputs) # [N,T_in,embed_depth=256]