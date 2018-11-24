# -*- coding: utf-8 -*-
# @Time    : 2018/10/18 17:53
# @Author  : MengnanChen
# @FileName: resnet.py
# @Software: PyCharm

import collections
import tensorflow as tf
slim=tf.contrib.slim


class Block(collections.namedtuple('Block',['scope','unit_fn','args'])):

    def subsample(self,inputs,factor,scope=None):
        if factor==1:
            return inputs
        else:
            return slim.max_pool2d(inputs,[1,1],stride=factor,scope=scope)

    def conv2d_same(self,inputs,num_outputs,kernel_size,stride,scope=None):
        if stride==1:
            return