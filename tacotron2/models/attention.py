# -*- coding: utf-8 -*-
# @Time    : 2018/8/28 15:23
# @Author  : MengnanChen
# @FileName: attention.py
# @Software: PyCharm Community Edition

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
from tensorflow.python.ops import variable_scope,array_ops

class LocationSensitiveAttention(BahdanauAttention):
    def __init__(self,num_units,memory,hparams,mask_encoder=True,
                 memory_sequence_length=None,smoothing=False,cumulate_weights=True,
                 name='LocationSensitiveAttention'):
        normalization_function=_smoothing_normalization if (smoothing == True) else None
        memory_length=memory_sequence_length if (mask_encoder==True) else None
        super(LocationSensitiveAttention,self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_length,
            probability_fn=normalization_function,
            name=name
        )
        self.location_convolution=tf.layers.Conv1D(filters=hparams.attention_filters,
                                                   kernel_size=hparams.attention_kernel,
                                                   padding='same',
                                                   use_bias=True,
                                                   bias_initializer=tf.zeros_initializer(),
                                                   name='location_features_convolution')
        self.location_layer=tf.layers.Dense(units=num_units,
                                            use_bias=False,
                                            dtype=tf.float32,
                                            name='location_features_layer')
        self._cumulate=cumulate_weights

    def __call__(self, query, state):
        previous_alignments=state
        with variable_scope.variable_scope(None,'Location_Sensitive_Attention',[query]):
            # query
            # [batch_size, query_depth] -> [batch_size,attention_dim]
            processed_query=self.query_layer(query) if self.query_layer else query
            # [batch_size,attention_dim] -> [batch_size,1,attention_dim]
            processed_query=tf.expand_dims(processed_query,axis=1)

            # previous_alignments(state)
            # [batch_size,max_time] -> [batch_size,max_time,1]
            expanded_alignments=tf.expand_dims(previous_alignments,axis=2)
            # [batch_size,max_time,1] -> [batch_size,max_time,filters]
            f=self.location_convolution(expanded_alignments)
            # [batch_size,max_time,filters] -> [batch_size,max_time,attention_dim]
            processed_location_features=self.location_layer(f)

            energy=_location_sensitive_score(processed_query,processed_location_features,self.keys)

        alignments=self._probability_fn(energy,previous_alignments)

        if self._cumulate:
            next_state=alignments+previous_alignments
        else:
            next_state=alignments

        return alignments,next_state


def _smoothing_normalization(e):
    '''
    a_{i,j}=sigmoid(e_{i,j})/sum_j(sigmoid(e_{i,j}))
    :param e:
    :return:
    '''
    return tf.nn.sigmoid(e)/tf.reduce_sum(tf.nn.sigmoid(e),axis=-1,keepdims=True)

def _location_sensitive_score(W_query,W_fil,W_keys):
    '''
    energy=dot(v_a,tanh(W_query+W_fil+W_keys+b_a))
    :param W_query:
    :param W_fil:
    :param W_keys:
    :return:
    '''
    dtype=W_query.dtype
    num_units=W_keys.shape[-1].value or array_ops.shape(W_keys)[-1]

    v_a=tf.get_variable('attention_variable',shape=[num_units],dtype=dtype,
                        initializer=tf.contrib.layers.xavier_initializer())
    b_a=tf.get_variable('attention_bias',shape=[num_units],dtype=dtype,
                        initializer=tf.zeros_initializer())
    # W_query: [batch_size,1,attention_dim]
    # W_location: [batch_size,max_time,attention_dim]
    # W_keys: [batch_size,max_time,attention_dim]
    # return: [batch_size,max_time], attention score(energy)
    return tf.reduce_sum(v_a*tf.tanh(W_keys+W_query+W_fil+b_a),axis=[2])