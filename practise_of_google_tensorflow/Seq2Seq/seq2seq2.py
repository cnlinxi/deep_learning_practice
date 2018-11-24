# -*- coding: utf-8 -*-
# @Time    : 2018/10/22 13:57
# @Author  : MengnanChen
# @FileName: seq2seq2.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense


class Seq2SeqModel(object):
    def __init__(self, rnn_size, layer_size, encoder_vocab_size,
                 decoder_vocab_size, embed_size, grab_clip, is_inference=False):
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')

        # embedding layer
        with tf.variable_scope('embedding'):
            encoder_embedding = tf.Variable(tf.truncated_normal([encoder_vocab_size, embed_size], stddev=0.1),
                                            name='encoder_embedding')
            decoder_embedding = tf.Variable(tf.truncated_normal([decoder_vocab_size, embed_size], stddev=0.1),
                                            name='decoder_embedding')

        # encoder
        with tf.variable_scope('encoder_layer'):
            encoder = self._get_simple_lstm(rnn_size, layer_size)

        with tf.device('/cpu:0'):
            input_x_embedding = tf.nn.embedding_lookup(encoder_embedding, self.input_x)

        # dynamic_rnn input: encoder object & input embedding
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(encoder, input_x_embedding, dtype=tf.float32)

        if is_inference:
            self.start_tokens = tf.placeholder(tf.int32, shape=[None], name='start_tokens')
            self.stop_tokens = tf.placeholder(tf.int32, name='stop_tokens')
            # inference_helper inputs: decoder_embedding & start_tokens(placeholder) & stop_token(placeholder)
            helper = GreedyEmbeddingHelper(decoder_embedding, self.start_tokens, self.stop_tokens)
        else:
            self.target_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='target_ids')
            self.decoder_seq_length = tf.placeholder(dtype=tf.int32, shape=[None], name='batch_seq_length')
            with tf.device('/cpu:0'):
                target_embedding = tf.nn.embedding_lookup(decoder_embedding, self.target_ids)
            # train_helper inputs: target_embedding & decoder_seq_length(placeholder)
            helper = TrainingHelper(target_embedding, self.decoder_seq_length)

        with tf.variable_scope('decoder'):
            fc_layer = Dense(decoder_vocab_size)
            decoder_cell = self._get_simple_lstm(rnn_size, layer_size)
            decoder = BasicDecoder(decoder_cell, helper, encoder_states, fc_layer)

        logits, final_states, final_sequence_length = dynamic_decode(decoder)

        if not is_inference:
            targets = tf.reshape(self.target_ids, [-1])
            logits_flat = tf.reshape(logits.rnn_output, [-1, decoder_vocab_size])

            self.loss = tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)

            # forward, get gradient
            tvars = tf.trainable_variables()
            # get gradients need loss
            grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grab_clip)

            # backward, update parameters
            optimizer = tf.train.AdamOptimizer(1e-3)
            self.train_op = optimizer.apply_gradients(zip(grad, tvars))
        else:
            self.prob = tf.nn.softmax(logits)

    def _get_simple_lstm(self, rnn_size, layer_size):
        lstm_layers = [tf.nn.rnn_cell.LSTMCell(rnn_size) for _ in range(layer_size)]
        return tf.nn.rnn_cell.MultiRNNCell(lstm_layers)
