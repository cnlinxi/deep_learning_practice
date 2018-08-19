# -*- coding: utf-8 -*-
# @Time    : 2018/8/19 10:23
# @Author  : MengnanChen
# @FileName: modules.py
# @Software: PyCharm Community Edition

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

def prenet(inputs,is_training,layer_sizes,scope=None):
    '''
    prenet_outputs = prenet(embedded_inputs, is_training, hp.prenet_depth)  # [N,T_in,prenet_depths[-1]=128]
    embedded_inputs: [N,T_in,embed_depth=256]
    :param inputs: [N,T_in,embed_depth=256]
    :param is_training:
    :param layer_sizes:
    :param scope:
    :return: [N,T_in,prenet_depths[-1]=128]
    '''
    x=inputs
    drop_rate=0.5 if is_training else 0.
    with tf.variable_scope(scope or 'prenet'):
        for i,size in enumerate(layer_sizes):
            dense=tf.layers.dense(x,units=size,activation=tf.nn.relu,name='dense_%d'%(i+1))
            x=tf.layers.dropout(dense,rate=drop_rate,name='dropout_%d'%(i+1))
    return x

def encoder_cbhg(inputs,input_lengths,is_training,depth):
    '''
    encoder_outputs=encoder_cbhg(prenet_outputs,input_lengths,is_training,hp.encoder_depth)
    prenet_outputs: [N,T_in,prenet_depths[-1]=128]
    :param inputs: [N,T_in,prenet_depths[-1]=128]
    :param input_lengths: 用于双向RNN的sequence_length
    :param is_training:
    :param depth: 该值的一半用作highway中H&T、双向RNN中GRUCell的输出维度 ####为何一半？
    :return: [N,T_in,encoder_depth=256]
    '''
    input_channels=inputs.get_shape()[2]
    return cbhg(
        inputs,
        input_lengths,
        is_training,
        scope='encoder_cbhg',
        K=16,  # convolution bank的kernel_size, 当K=16时，其中的conv1d的卷积核大小为1~16
        projections=[128,input_channels],  # conv1d-maxpooling之后，highway之前的残差连接的投影层大小
        depth=depth
    )

def post_cbhg(inputs,input_dim,is_training,depth):
    return cbhg(
        inputs,
        None,
        is_training,
        scope='post_cbhg',
        K=8,
        projections=[256,input_dim],
        depth=depth
    )

def cbhg(inputs,input_lengths,is_training,scope,K,projections,depth):
    # CBHG: 对序列数据特征提取
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_bank'):
            # 卷积银行(convolution bank): 在最后的axis上做concatenate，从所有的卷积中堆叠通道
            # 注意，此处的conv1d为自定义的conv1d，其预定义了一些参数且自带batch normalization
            conv_outputs = tf.concat(
                [conv1d(inputs, k, 128, tf.nn.relu, is_training, 'conv1d_%d' % k) for k in range(1, K + 1)],
                axis=-1
            )

        # maxpooling
        maxpool_output = tf.layers.max_pooling1d(
            conv_outputs,
            pool_size=2,
            strides=1,
            padding='same'
        )

        # 两层投影层
        proj1_output =conv1d(maxpool_output,3,projections[0],tf.nn.relu,is_training,'proj_1')
        proj2_output=conv1d(proj1_output,3,projections[1],None,is_training,'proj_2')

        # 残差连接
        highway_input=proj2_output+inputs

        half_depth=depth//2
        assert half_depth*2==depth,'encoder和postnet的depths必须是偶数'

        # 处理维度不匹配: 直接一个全连接层dense到目标维度
        if highway_input.shape[2]!=half_depth:
            highway_input=tf.layers.dense(highway_input,half_depth)

        # 4层HighwayNet
        for i in range(4):
            highway_input=highwaynet(highway_input,half_depth)
        rnn_input=highway_input

        # 双向RNN
        outputs,states=tf.nn.bidirectional_dynamic_rnn(
            GRUCell(half_depth),
            GRUCell(half_depth),
            rnn_input,
            sequence_length=input_lengths,
            dtype=tf.float32
        )
        # 前向后向做concat
        return tf.concat(outputs,axis=2)

def highwaynet(inputs,scope,depth):
    with tf.variable_scope(scope):
        H=tf.layers.dense(
            inputs,
            units=depth,
            activation=tf.nn.relu,
            name='H'
        )
        T=tf.layers.dense(
            inputs,
            units=depth,
            activation=tf.nn.sigmoid,
            name='T',
            bias_initializer=tf.constant_initializer(-1.0)
        )
        # T为门函数, H为非线性变换
        return H*T+inputs*(1.0-T)

def conv1d(inputs,kernel_size,channels,activation,is_training,scope):
    with tf.variable_scope(scope):
        conv1d_output=tf.layers.conv1d(
            inputs,
            filters=channels,
            kernel_size=kernel_size,
            activation=activation,
            padding='same'
        )
        return tf.layers.batch_normalization(conv1d_output,training=is_training)