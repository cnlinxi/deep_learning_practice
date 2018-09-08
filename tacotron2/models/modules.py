# -*- coding: utf-8 -*-
# @Time    : 2018/8/28 15:15
# @Author  : MengnanChen
# @FileName: modules.py
# @Software: PyCharm Community Edition

import tensorflow as tf

class Prenet:
    def __init__(self,is_training,layers_sizes=(256,256),drop_rate=0.5,activation=tf.nn.relu,scope=None):
        super(Prenet,self).__init__()
        self.drop_rate=drop_rate
        self.layer_sizes=layers_sizes

        self.activation=activation
        self.is_training=is_training

        self.scope='prenet' if scope is None else scope

    def __call__(self, inputs):
        x=inputs

        with tf.variable_scope(self.scope):
            for i,size in enumerate(self.layer_sizes):
                dense=tf.layers.dense(x,units=size,activation=self.activation,
                                      name='dense_{}'.format(i+1))
                x=tf.layers.dropout(dense,rate=self.drop_rate,training=True,
                                    name='dropout_{}'.format(i+1)+self.scope)

        return x

class ZoneoutLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,num_units,is_training,zoneout_factor_cell=0.,zoneout_factor_output=0.,
                 state_is_tuple=True,name=None):
        zm = min(zoneout_factor_output, zoneout_factor_cell)
        zs = max(zoneout_factor_output, zoneout_factor_cell)

        if zm < 0. or zs > 1.:
            raise ValueError('One/both provided Zoneout factors are not in [0, 1]')

        self._cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=state_is_tuple, name=name)
        self._zoneout_cell = zoneout_factor_cell
        self._zoneout_outputs = zoneout_factor_output
        self.is_training = is_training
        self.state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        '''Runs vanilla LSTM Cell and applies zoneout.
        '''
        # Apply vanilla LSTM
        output, new_state = self._cell(inputs, state, scope)

        if self.state_is_tuple:
            (prev_c, prev_h) = state
            (new_c, new_h) = new_state
        else:
            num_proj = self._cell._num_units if self._cell._num_proj is None else self._cell._num_proj
            prev_c = tf.slice(state, [0, 0], [-1, self._cell._num_units])
            prev_h = tf.slice(state, [0, self._cell._num_units], [-1, num_proj])
            new_c = tf.slice(new_state, [0, 0], [-1, self._cell._num_units])
            new_h = tf.slice(new_state, [0, self._cell._num_units], [-1, num_proj])

        # Apply zoneout
        if self.is_training:
            # nn.dropout takes keep_prob (probability to keep activations) not drop_prob (probability to mask activations)!
            c = (1 - self._zoneout_cell) * tf.nn.dropout(new_c - prev_c, (1 - self._zoneout_cell)) + prev_c
            h = (1 - self._zoneout_outputs) * tf.nn.dropout(new_h - prev_h, (1 - self._zoneout_outputs)) + prev_h

        else:
            c = (1 - self._zoneout_cell) * new_c + self._zoneout_cell * prev_c
            h = (1 - self._zoneout_outputs) * new_h + self._zoneout_outputs * prev_h

        new_state = tf.nn.rnn_cell.LSTMStateTuple(c, h) if self.state_is_tuple else tf.concat(1, [c, h])

        return output, new_state

class DecoderRNN:
    def __init__(self,is_training,layers=2,size=1024,zoneout=0.1,scope=None):
        super(DecoderRNN,self).__init__()
        self.is_training=is_training

        self.layers=layers
        self.size=size
        self.zoneout=zoneout
        self.scope='decoder_rnn' if scope is None else scope

        self.rnn_layers=[ZoneoutLSTMCell(size,is_training,
                                         zoneout_factor_cell=zoneout,
                                         zoneout_factor_output=zoneout,
                                         name='decoder_LSTM_{}'.format(i+1)) for i in range(layers)]
        self._cell=tf.contrib.rnn.MultiRNNCell(self.rnn_layers,state_is_tuple=True)

    def __call__(self, inputs,states):
        with tf.variable_scope(self.scope):
            return self._cell(inputs,states)

class FrameProjection:
    def __init__(self,shape=80,activation=None,scope=None):
        super(FrameProjection,self).__init__()
        self.shape=shape
        self.activation=activation

        self.scope='Linear_projection' if scope is not None else scope
        self.dense=tf.layers.Dense(units=shape,activation=activation,name='projection_{}'.format(self.scope))

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            output=self.dense(inputs)
            return output

class StopProjection:
    def __init__(self,is_training,shape=1,activation=tf.nn.sigmoid,scope=None):
        super(StopProjection).__init__()
        self.is_training=is_training

        self.shape=shape
        self.activation=activation
        self.scope='stop_token_projection' if scope is not None else scope

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            output=tf.layers.dense(inputs,units=self.shape,activation=None,
                                   name='projection_{}'.format(self.scope))
            if self.is_training:
                return output
            return self.activation(output)

def conv1d(inputs,kernel_size,channels,activation,is_training,drop_rate,scope):
    with tf.variable_scope(scope):
        conv1d_output=tf.layers.conv1d(
            inputs,
            filters=channels,
            kernel_size=kernel_size,
            activation=None,
            padding='same'
        )
        batched=tf.layers.batch_normalization(conv1d_output,training=is_training)
        activated=activation(batched)
        return tf.layers.dropout(activated,rate=drop_rate,training=is_training,
                                 name='dropout_{}'.format(scope))


class PostNet:
    def __init__(self,is_training,hparams,activation=tf.nn.tanh,scope=None):
        super(PostNet,self).__init__()
        self.is_training=is_training

        self.kernel_size=hparams.postnet_kernel_size
        self.channels=hparams.postnet_channels
        self.activation=activation
        self.scope='postnet_convolutions' if scope is None else scope
        self.postnet_num_layers=hparams.postnet_num_layers
        self.drop_rate=hparams.tacotron_dropout_rate

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            x=inputs
            for i in range(self.postnet_num_layers-1):
                x=conv1d(x,self.kernel_size,self.channels,self.activation,
                         self.is_training,self.drop_rate,'conv_layer_{}_'.format(i+1)+self.scope)
            x=conv1d(x,self.kernel_size,self.channels,lambda _:_,self.is_training,self.drop_rate,
                     'conv_layer_{}_'.format('last_conv1')+self.scope)
            return x

def cbhg(inputs,input_lengths,is_training,scope,K,projections):
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_bank'):
            conv_outputs=tf.concat([conv1d(inputs,k,128,tf.nn.relu,is_training,0.,'conv1d_%d'%k) for k in range(1,K)],axis=-1)

        # maxpooling
        maxpool_output=tf.layers.max_pooling1d(
            conv_outputs,
            pool_size=2,
            strides=1,
            padding='same'
        )

        proj1_output=conv1d(maxpool_output, kernel_size=3,channels=projections[0],activation=tf.nn.relu,is_training=is_training,drop_rate=0.,scope='proj_1')
        proj2_output=conv1d(proj1_output,kernel_size=3,channels=projections[1],activation=lambda _:_,is_training=is_training,drop_rate=0.,scope='proj_2')

        # residual connection
        highway_input=proj2_output+inputs
        if highway_input.shape[2]!=128:
            highway_input=tf.layers.dense(highway_input,128)

        for i in range(4):
            highway_input=highwaynet(highway_input,'highway_%d'%(i+1))
        rnn_input=highway_input

        outputs,states=tf.nn.bidirectional_dynamic_rnn(
            tf.nn.rnn_cell.GRUCell(128),
            tf.nn.rnn_cell.GRUCell(128),
            rnn_input,
            sequence_length=input_lengths,
            dtype=tf.float32
        )
        # concat forward & backward
        return tf.concat(outputs,axis=2)

def post_cbhg(inputs,input_dim,is_training):
    return cbhg(
        inputs,
        None,
        is_training,
        scope='post_cbhg',
        K=8,
        projections=[256,input_dim]
    )

def highwaynet(inputs,scope):
    with tf.variable_scope(scope):
        H=tf.layers.dense(inputs,
                          units=128,
                          activation=tf.nn.relu,
                          name='H')
        T=tf.layers.dense(inputs,
                          units=128,
                          activation=tf.nn.sigmoid,
                          name='T',
                          bias_initializer=tf.constant_initializer(-1.0))
        return H*T+inputs*(1.0-T)

def post_cbhg(inputs,input_dim,is_training):
    return