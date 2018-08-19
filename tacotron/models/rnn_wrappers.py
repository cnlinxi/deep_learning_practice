# -*- coding: utf-8 -*-
# @Time    : 2018/8/19 14:06
# @Author  : MengnanChen
# @FileName: rnn_wrappers.py
# @Software: PyCharm Community Edition

'''
这个文件非常多的不懂
'''


import collections
import tensorflow as tf
from .modules import prenet
from .attention import _compute_attention
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops import array_ops, check_ops, rnn_cell_impl, tensor_array_ops
from tensorflow.python.util import nest
from hparams import hparams as hp

class TacotronDecoderCellState(collections.namedtuple('TacotronDecoderCellState',
                                                      ('cell_state','attention','time','alignments','alignment_history'))):
    '''
    'namedtuple'存储'TacotronDecoderCell'的状态'state'
    包括:
    - cell_state: 在前一个时间步上被wrapped的RNNCell的state
    - attention: 在前一个时间步上输出的attention
    - alignments: 对于每个对齐机制在之前的时间步上输出的包含“对齐”信息的single或tuple的Tensor
    - alignment_history: 对于每个对齐机制在所有时间步上输出的包含“对齐”信息的single或tuple的TensorArray.
    调用stack()可将其转化为Tensor
    '''
    def replace(self,**kwargs):
        '''
        当重写kwargs提供的components时，克隆当前状态 ####所以这个类到底是干嘛的？
        :param kwargs:
        :return:
        '''
        return super(TacotronDecoderCellState, self)._replace(**kwargs)

class TacotronDecoderWrapper(RNNCell):
    def __init__(self, is_training,attention_mechanism,rnn_cell,frame_projection=None,
                 stop_projection=None):
        super(TacotronDecoderWrapper,self).__init__()
        self._training=is_training
        self._attention_mechanism=attention_mechanism
        self._cell=rnn_cell
        self._frame_projection=frame_projection
        self._stop_projection=stop_projection

        self._attention_layer_size=self._attention_mechanism.values.get_shape()[-1].value

    def _batch_size_checks(self,batch_size,error_message):
        return [check_ops.assert_equal(batch_size,self._attention_mechanism.batch_size,
                                       message=error_message)]

    def output_size(self):
        return self._cell.output_size+self._cell.state_size.attention

    def state_size(self):
        '''
        TacotronDecoderWrapper的state_size属性
        :return:
        '''
        return TacotronDecoderCellState(
            cell_state=self._cell._cell.state_size,
            time=tensor_shape.TensorShape([]),
            attention=self._attention_mechanism.alignment_size,
            alignment_history=()
        )

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__+'ZeroState',values=[batch_size]):  ####这句话什么意思？
            cell_state=self._cell.zero_state(batch_size,dtype=dtype)
            error_message=(
                'When calling zero_state of TacotronDecoderCell %s'%self._base_name+
                'Non-matching batch sizes between the memory '
                '(encoder output) and the requested batch size.'
            )
            with ops.control_dependencies(
                self._batch_size_checks(batch_size,error_message)
            ):
                cell_state = nest.map_structure(
                    lambda s: array_ops.identity(s, name="checked_cell_state"),
                    cell_state)
                return TacotronDecoderCellState(
                    cell_state=cell_state,
                    time=array_ops.zeros([], dtype=tf.int32),
                    attention=rnn_cell_impl._zero_state_tensors(self._attention_layer_size, batch_size, dtype),
                    alignments=self._attention_mechanism.initial_alignments(batch_size, dtype),
                    alignment_history=tensor_array_ops.TensorArray(dtype=dtype, size=0,
                                                                   dynamic_size=True))

    def __call__(self, inputs, state):
        # Information bottleneck (essential for learning attention)
        prenet_output = prenet(inputs, self._training, hp.prenet_depths, scope='decoder_prenet')

        # Concat context vector and prenet output to form RNN cells input (input feeding)
        rnn_input = tf.concat([prenet_output, state.attention], axis=-1)

        # Unidirectional RNN layers
        rnn_output, next_cell_state = self._cell(tf.layers.dense(rnn_input, hp.decoder_depth), state.cell_state)

        # Compute the attention (context) vector and alignments using
        # the new decoder cell hidden state as query vector
        # and cumulative alignments to extract location features
        # The choice of the new cell hidden state (s_{i}) of the last
        # decoder RNN Cell is based on Luong et Al. (2015):
        # https://arxiv.org/pdf/1508.04025.pdf
        previous_alignments = state.alignments
        previous_alignment_history = state.alignment_history
        context_vector, alignments, cumulated_alignments = _compute_attention(self._attention_mechanism,
                                                                              rnn_output,
                                                                              previous_alignments,
                                                                              attention_layer=None)

        # Concat RNN outputs and context vector to form projections inputs
        # projections_input = tf.concat([rnn_output, context_vector], axis=-1)
        cell_outputs = tf.concat([rnn_output, context_vector], axis=-1)

        # Compute predicted frames and predicted <stop_token>
        # cell_outputs = self._frame_projection(projections_input)
        # stop_tokens = self._stop_projection(projections_input)

        # Save alignment history
        alignment_history = previous_alignment_history.write(state.time, alignments)

        # Prepare next decoder state
        next_state = TacotronDecoderCellState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=context_vector,
            alignments=cumulated_alignments,
            alignment_history=alignment_history)

        # return (cell_outputs, stop_tokens), next_state
        return cell_outputs, next_state