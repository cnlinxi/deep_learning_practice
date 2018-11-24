# -*- coding: utf-8 -*-
# @Time    : 2018/9/21 11:37
# @Author  : MengnanChen
# @FileName: masked.py
# @Software: PyCharm

import tensorflow as tf
from functools import partial


WN_INIT_SCALE=1.0

def get_unsample_act(act_str):
    if act_str=='tanh':
        return tf.nn.tanh
    elif act_str=='relu':
        return tf.nn.relu
    elif act_str=='leaky_relu':
        return partial(tf.nn.leaky_relu,alpha=0.4)
    else:
        raise ValueError('Unsupported activation function for unsample layer')