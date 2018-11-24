# -*- coding: utf-8 -*-
# @Time    : 2018/11/24 11:03
# @Author  : MengnanChen
# @FileName: model.py
# @Software: PyCharm
# refer to: https://github.com/tensorflow/nmt
# https://github.com/NELSONZHAO/zhihu/blob/master/basic_seq2seq/Seq2seq_char.ipynb
# https://blog.csdn.net/weixin_42724775/article/details/81094429
# https://blog.csdn.net/thriving_fcl/article/details/74165062

import tensorflow as tf
from tensorflow.python.layers.core import Dense


class hparams:
    batch_size = 32
    embedding_size = 256
    rnn_size = 256  # num of lstm units, also the output dimension of lstm
    layer_size = 4  # the layer of lstm
    # the depth of the attention (output) layer(s). If None (default), use the context as attention at each time step.
    attention_layer_size = 2

    initial_learning_rate = 1e-3  # initial learning rate
    decay_rate = 0.96
    decay_steps = 2000
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_epsilon = 1e-6


class Seq2SeqModel:
    def __init__(self, mode='train'):
        self._training = (mode == 'train')

    def build_model(self, elements, feeder):
        # self.input_origin: [batch_size,seq_len]
        # self.input_input_lengths: [batch_size]
        # self.input_target: [batch_size,seq_len]
        self.input_origin, self.input_lengths, self.input_target = elements
        self.feeder = feeder

        # embedding layer
        with tf.variable_scope('embedding'), tf.device('/cpu:0'):
            encoder_lookup_table = tf.get_variable(
                initializer=tf.truncated_normal(shape=(feeder.origin_vocab_size, hparams.embedding_size), stddev=0.1)
            )
            decoder_lookup_table = tf.get_variable(
                initializer=tf.truncated_normal(shape=(feeder.target_vocab_size, hparams.embedding_size), stddev=0.1)
            )
            # input_origin: [batch_size,seq_len,embedding_size]
            input_origin = tf.nn.embedding_lookup(encoder_lookup_table, self.input_origin)

        def get_lstm_cell(rnn_size):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(rnn_size,
                                                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2018))
            return lstm_cell

        # encoder
        with tf.variable_scope('encoder'):
            cell_fw = get_lstm_cell(rnn_size=hparams.rnn_size)
            cell_bw = get_lstm_cell(rnn_size=hparams.rnn_size)

            # encoder_output: (output_fw,output_bw)
            # output_fw: [batch_size,max_time,output_size], forward rnn output of bidirectional rnn
            # output_bw: [batch_size,max_time,output_size], backward rnn output of bidirectional rnn
            # encoder_state: (output_state_fw,output_state_bw)
            # output_state_fw: [batch_size,output_size], forward final state of bidirectional rnn
            # output_state_bw: [batch_size,output_size], backward final state of bidirectional rnn
            # output_state_fw: (c_fw,h_fw), c_fw/h_fw: [batch_size,hparams.rnn_size]
            # output_state_bw: (c_bw,h_bw), c_bw/h_bw: [batch_size,hparams.rnn_size]
            encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                            cell_bw=cell_bw,
                                                                            inputs=input_origin,
                                                                            sequence_length=feeder.input_origin_lengths,
                                                                            dtype=tf.float32)

        # decoder
        with tf.variable_scope('decoder'):
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_lstm_cell(hparams.rnn_size) for _ in range(hparams.layer_size)])
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                # the depth of attention mechanism
                num_units=hparams.attention_layer_size,
                # the memory to query, usually the output of encoder
                # [batch_size,max_time,output_size]
                memory=encoder_output,
                # sequence_lengths of memory
                memory_sequence_length=feeder.input_origin_lengths
            )
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                               attention_layer_size=hparams.attention_layer_size)
            decoder_initial_state = tf.concat(encoder_state,
                                              axis=-1)  # encoder final state as the initial value of decoder
            # decoder_initial_state = decoder_cell.zero_state(batch_size=hparams.batch_size, dtype=tf.float32).clone(
            #                                                 cell_state=tf.concat(encoder_state,axis=-1))
            output_layer = Dense(feeder.target_vocab_size,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            # helper will be different when train and infer
            if self._training:
                with tf.device('/cpu:0'):
                    input_target = tf.nn.embedding_lookup(decoder_lookup_table, self.input_target)
                helper = tf.contrib.seq2seq.TrainingHelper(inputs=input_target,
                                                           sequence_length=feeder.input_target_lengths)
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=decoder_lookup_table,
                    start_tokens=feeder.start_tokens,  # [batch_size], tf.int32, can be [id_of_<go>] * batch_size
                    end_token=feeder.end_tokens  # scalar, tf.int32, the token that marks end of decoding, id of <eos>
                )
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                      helper,
                                                      decoder_initial_state,
                                                      output_layer)
            # perform dynamic decoding with decoder, call initialize() once and step() repeatedly on the decoder.
            # self.final_output.rnn_output: logits
            # self.final_output.sample_id: sample_ids
            self.final_output, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                                       impute_finished=True,
                                                                                                       maximum_iterations=feeder.max_target_sequence_length)

    def add_loss(self):
        self.logits = tf.identity(self.final_output.rnn_output, name='logits')
        masks = tf.sequence_mask(self.feeder.input_target_lengths, self.feeder.max_target_sequence_length,
                                 dtype=tf.float32)
        with tf.variable_scope('loss'):
            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.logits,  # [batch_size,seq_len,num_of_target_symbols]
                targets=self.input_target,  # [batch_size,seq_len]
                weights=masks
            )

    def add_optimizer(self, global_step):
        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=hparams.initial_learning_rate,
                decay_rate=hparams.decay_rate,
                decay_steps=hparams.decay_steps,
                global_step=global_step
            )
            self.optimizer = tf.train.AdamOptimizer(
                self.learning_rate, beta1=hparams.adam_beta1, beta2=hparams.adam_beta2, epsilon=hparams.adam_epsilon
            ).minimize(self.loss)
