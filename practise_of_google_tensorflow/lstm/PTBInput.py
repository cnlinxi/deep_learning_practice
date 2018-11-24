# -*- coding: utf-8 -*-
# @Time    : 2018/10/22 9:22
# @Author  : MengnanChen
# @FileName: PTBInput.py
# @Software: PyCharm


import time

import numpy as np
import tensorflow as tf

class PTBInput(object):
    def __init__(self,config,data,name=None):
        self.batch_size=batch_size=config.batch_size
        self.num_steps=num_steps=config.num_steps
        self.epoch_size=((len(data)//batch_size)-1)//num_steps

        self.input_data,self.targets=[],[]


class PTBModel(object):
    def __init__(self,is_training,config,input_):
        self._input=input_
        batch_size=input_.batch_size
        num_steps=input_.num_steps
        size=config.hidden_size
        vocab_size=config.vocab_size

        def lstm_cell():
            return tf.nn.rnn_cell.BasicLSTMCell(size,forget_bias=0.,state_is_tuple=True)
        attn_cell=lstm_cell
        if is_training and config.keep_prob<1:
            def attn_cell():
                return tf.nn.rnn_cell.DropoutWrapper(lstm_cell(),output_keep_prob=config.keep_prob)

        cell=tf.nn.rnn_cell.MultiRNNCell([attn_cell() for _ in range(config.num_layers)])
        self.initial_state=cell.zero_state(batch_size,tf.float32)

