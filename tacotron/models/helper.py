# -*- coding: utf-8 -*-
# @Time    : 2018/8/19 20:53
# @Author  : MengnanChen
# @FileName: helper.py
# @Software: PyCharm Community Edition

import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper

class TacoTestHelper(Helper):
    def __init__(self,batch_size,output_dim,r):
        with tf.name_scope('TacoTestHelper'):
            self._batch_size=batch_size
            self._output_dim=output_dim
            self._end_token=tf.tile([0.0],[output_dim*r])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        return (tf.tile([False],[self._batch_size]),_go_frames([self._batch_size,self._output_dim]))

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0],[self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        '''
        在EOS处停止，否则传入最后的output作为下一个input，pass through state
        :param time:
        :param outputs:
        :param state:
        :param sample_ids:
        :param name:
        :return:
        '''
        with tf.name_scope('TacoTestHelper'):
            finished=tf.reduce_all(tf.equal(outputs,self._end_token),axis=1)
            # 将最后一帧的output作为下一个input。output: [N,output_dim*r]
            next_inputs=outputs[:,-self._output_dim:]
            return (finished,next_inputs,state)


class TacoTrainingHelper(Helper):
    def __init__(self,inputs,targets,output_dim,r):
        # input: [N,T_in]
        # output: [N,T_in,D]
        with tf.name_scope('TacoTrainingHelper'):
            self._batch_size=tf.shape(inputs)[0]
            self._output_dim=output_dim

            # 输入每个r-th的target帧作为输入
            self._targets=targets[:,r-1::r,:]

            # 对于每个target,使用全长，因为我们不希望mask填充的帧
            num_steps=tf.shape(self._targets)[1]
            self._lengths=tf.tile([num_steps],[self._batch_size])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        return (tf.tile([False],[self._batch_size]),_go_frames(self._batch_size,self._output_dim))

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0],[self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope(name or 'TacoTrainingHelper'):
            finished=(time+1>=self._lengths)
            next_inputs=self._targets[:,time,:]
            return (finished,next_inputs,state)


def _go_frames(batch_size,output_dim):
    '''
    对于一个给定的batch_size和output_dim，返回全零的<GO>帧
    :param batch_size:
    :param output_dim:
    :return:
    '''
    return tf.tile([[0.0]],[batch_size,output_dim])