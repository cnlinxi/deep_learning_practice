# -*- coding: utf-8 -*-
# @Time    : 2018/8/28 11:05
# @Author  : MengnanChen
# @FileName: architecture_wrappers.py
# @Software: PyCharm Community Edition

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import check_ops,array_ops,math_ops
import collections

def _compute_attention(attention_mechanism,cell_output,attention_state,attention_layer):
    # cell_output -> LSTM_output
    # attention_state -> previous_alignments
    # compute the attention and alignments for a given attention mechanism

    # context vector -> c_i: c_i=\sum_{j=1}^{T_x}\alpha_{ij}h_j
    # alignments or attention weights -> \alpha_{ij}: softmax(e_{ij})
    # energy -> e_{ij}
    alignments,next_attention_state=attention_mechanism(
        cell_output,state=attention_state
    )
    # reshape from [batch_size, memory_time] to [batch_size,1,memory_time]
    expanded_alignments=array_ops.expand_dims(alignments,1)

    # context: inner product of alignments & values along the memory time dimension
    # alignments: [batch_size,1,memory_time]
    # attention_mechanism.values: [batch_size,memory_time,memory_size]
    # context: [batch_size,1,memory_size]
    context=math_ops.matmul(expanded_alignments,attention_mechanism.values)
    # context: [batch_size,memory_size]
    context=array_ops.squeeze(context,[1])

    if attention_layer is not None:
        attention=attention_layer(array_ops.concat([cell_output,context],axis=1))
    else:
        attention=context
    # context_vector, alignments, cumulated_alignments
    return attention,alignments,next_attention_state

class TacotronEncoderCell(RNNCell):
    # Tacotron2 encoder:
    # input_text -> char_embedding -> 3_conv_layer -> Bi-LSTM
    def __init__(self,convolutional_layers,lstm_layer):
        super(TacotronEncoderCell,self).__init__()
        self._convolutions=convolutional_layers
        self._cell=lstm_layer

    def __call__(self, inputs,input_lengths=None):
        # 3 conv layer
        conv_output=self._convolutions(inputs)
        # bi-lstm
        hidden_representation=self._cell(conv_output,input_lengths)
        self.conv_output_shape=conv_output.shape
        return hidden_representation

class TacotronDecoderCellState(
    collections.namedtuple('TacotronDecoderCellState',
                           ('cell_state','attention','time','alignments','alignment_history')
                           )):
    '''
    nametuple storing the state of a TacotronDecoderCell
    '''
    def replace(self,**kwargs):
        return super(TacotronDecoderCellState,self)._replace(**kwargs)

class TacotronDecoderCell(RNNCell):
    def __init__(self,prenet,attention_mechanism,rnn_cell,frame_projection,stop_projection):
        super(TacotronDecoderCell,self).__init__()
        self._attention_mechanism=attention_mechanism
        self._cell=rnn_cell
        self._frame_projection=frame_projection
        self._stop_projection=stop_projection

        self._attention_layer_size=self._attention_mechanism.values.get_shape()[-1].value

    def _batch_size_checks(self,batch_size,error_message):
        return [check_ops.assert_equal(batch_size,
                                       self._attention_mechanism.batch_size,
                                       message=error_message)]
    @property
    def output_size(self):
        return self._frame_projection.shape

    def __call__(self, inputs, state):
        # Information bottleneck
        # prenet: compress last output information
        prenet_output=self._prenet(inputs)

        # concat compressed inputs & previous context vector
        LSTM_input=tf.concat([prenet_output,state.attention],axis=-1)

        # decoder RNN (actual decoding) to predict current state s_{i}
        LSTM_output,next_cell_state=self._cell(LSTM_input,state.cell_state)

        previous_alignments=state.alignments
        previous_alignment_history=state.alignment_history

        # compute new context vector c_{i} based on s_{i} and a cumulative sum of previous alignments
        context_vector,alignments,cumulated_alignments=_compute_attention(self._attention_mechanism,
                                                                          LSTM_output,
                                                                          previous_alignments,
                                                                          attention_layer=None)

        projections_input=tf.concat([LSTM_output,context_vector],axis=-1)

        # predict new output
        cell_outputs=self._frame_projection(projections_input)
        # predict <stop_token>
        stop_tokens=self._stop_projection(projections_input)

        alignment_history=previous_alignment_history.write(state.time,alignments)

        next_state=TacotronDecoderCellState(
            time=state.time+1,
            cell_state=next_cell_state,
            attention=context_vector,
            alignments=cumulated_alignments,
            alignment_history=alignment_history
        )

        return (cell_outputs,stop_tokens),next_state