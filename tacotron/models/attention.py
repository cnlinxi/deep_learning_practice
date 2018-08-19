# -*- coding: utf-8 -*-
# @Time    : 2018/8/19 16:20
# @Author  : MengnanChen
# @FileName: attention.py
# @Software: PyCharm Community Edition

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops,math_ops,nn_ops,variable_scope

def _compute_attention(attention_mechanism,cell_output,attention_state,attention_layer):
    # 对一个给定的attention机制，计算attention和alignments
    alignments,next_attention_state=attention_mechanism(
        cell_output,state=attention_state
    )

    # reshape, from: [batch_size, memory_time]->[batch_size,1,memory_time]
    expanded_alignments=array_ops.expand_dims(alignments,axis=1)

    # context是alignments和memory时间轴上值的内积
    # alignments: [batch_size,1,memory_time]
    # attention_mechanism.value: [batch_size,memory_time,memory_size]
    # 内积后结果: [batch_size,1,memory_size]
    # 然后对维度为1的轴进行squeeze
    context=math_ops.matmul(expanded_alignments,attention_mechanism.values)
    context=array_ops.squeeze(context,[1])

    if attention_layer is not None:
        attention=attention_layer(array_ops.concat([cell_output,context],axis=1))
    else:
        attention=context

    return attention,alignments,next_attention_state

def _location_sensitive_score(W_query,W_fil,W_keys):
    '''
    Impelements Bahdanau-style (cumulative) scoring function.
	This attention is described in:
	J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-gio,
	“Attention-based models for speech recognition,” in Advances in Neural
	Information Processing Systems, 2015, pp.577–585.
    hybrid attention (content-based + location-based)
    f = F*α_{i-1}
    #######
    energy=dot(v_a,tanh(W_keys(h_enc)+W_query(h_dec)+W_fil(f)+b_a))
    #######
    :param W_query: [batch_size,1,attention_dim] compare to location features
    :param W_fil: [batch_size,max_time,attention_dim], 处理之前的alignments到location features
    :param W_keys:[batch_size,max_time,attention_dim], 典型值为encoder的输出
    :return: [batch_size,max_time] attention score(energy)
    '''
    dtype=W_query.dtype
    num_units=W_keys.shape[-1].value or array_ops.shape(W_keys)[-1]

    v_a=tf.get_variable(
        'attention_variable',shape=[num_units],dtype=dtype,
        initializer=tf.contrib.layers.xavier_initializer()
    )
    b_a=tf.get_variable(
        'attention_bias',shape=[num_units],dtype=dtype,
        initializer=tf.zeros_initializer()
    )
    return tf.reduce_sum(v_a*tf.tanh(W_keys+W_query+W_fil+b_a),[2])

def _smoothing_normalization(e):
    '''施加平滑规范化而非softmax
    J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Bengio,
    “Attention-based models for speech recognition,”
    in Advances in Neural Information Processing Systems, 2015, pp.577–585.
    #######
    a_{i,j}=sigmoid(e_{i,j})/sum_j(sigmoid(e_{i,j}))
    #######
    :param e: [batch_size,max_time], attention机制的能量值
    :return: [batch_size,max_time], 规范化对齐，可能会出现多个memory时间步
    '''
    return tf.nn.sigmoid(e)/tf.reduce_sum(tf.nn.sigmoid(e),axis=-1,keepdims=True)

class LocationSensitiveAttention(BahdanauAttention):
    def __init__(self, num_units, memory, smoothing=False, cumulate_weights=True, name='LocationSensitiveAttention'):
        normalization_function=_smoothing_normalization if (smoothing==True) else None
        super(LocationSensitiveAttention,self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=None,
            probability_fn=normalization_function,
            name=name
        )

        self.location_convolution=tf.layers.Conv1D(
            filters=32,
            kernel_size=(31,),
            padding='same',
            use_bias=True,
            bias_initializer=tf.zeros_initializer(),
            name='location_features_convolution'
        )
        self.location_layer=tf.layers.Dense(
            units=num_units,
            use_bias=False,
            dtype=tf.float32,
            name='location_feature_layer'
        )
        self._cumulate=cumulate_weights

    def __call__(self, query, state):
        '''
        基于key和value为query打分
        :param query: [batch_size,query_depth], 用来与self.value做对比
        :param state: [batch_size,alignments_size], 用来与self.value做对比
        alignment_size即是memory的max_time
        :return: alignments: [batch_size,alignment_size], alignment_size即是memory的max_time
        '''
        previous_alignments=state
        with variable_scope.variable_scope(None,'Location_Sensitive_Attention',[query]):
            # processed_query: [batch_size,query_depth] -> [batch_size,attention_dim]
            processed_query=self.query_layer(query) if self.query_layer else query
            # -> [batch_size,1,attention_dim]
            processed_query=tf.expand_dims(processed_query,axis=1)

            # processed_location_features: [batch_size,max_time,attention_dim]
            # [batch_size,max_time] -> [batch_size,max_time,1]
            expanded_alignments=tf.expand_dims(previous_alignments,axis=2)
            # location features: [batch_size,max_time,filters]
            f=self.location_convolution(expanded_alignments)
            # projected location features: [batch_size,max_time,attention_dim]
            processed_location_features=self.location_layer(f)

            # energy: [batch_size,max_time]
            energy=_location_sensitive_score(processed_query,processed_location_features,self.keys)

        # alignments shape=energy shape=[batch_size,max_time]
        alignments=self._probability_fn(energy,previous_alignments)

        if self._cumulate:
            next_state=alignments+previous_alignments
        else:
            next_state=alignments

        return alignments,next_state