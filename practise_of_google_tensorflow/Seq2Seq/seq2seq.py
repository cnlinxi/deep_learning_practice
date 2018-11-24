# -*- coding: utf-8 -*-
# @Time    : 2018/10/22 10:53
# @Author  : MengnanChen
# @FileName: seq2seq.py
# @Software: PyCharm
# refer to: https://blog.csdn.net/thriving_fcl/article/details/74165062

import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense


class Seq2SeqModel(object):

    def __init__(self, rnn_size, layer_size, encoder_vocab_size,
                 decoder_vocab_size, embed_size, grad_clip, is_inference=False):
        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')

        # embedding layer
        with tf.variable_scope('embedding'):
            encoder_embedding = tf.Variable(tf.truncated_normal(shape=[encoder_vocab_size, embed_size], stddev=0.1),
                                            name='encoder_embedding')
            decoder_embedding = tf.Variable(tf.truncated_normal(shape=[decoder_vocab_size, embed_size], stddev=0.1),
                                            name='decoder_embedding')
        # encoder
        with tf.variable_scope('encoder'):
            encoder = self._get_simple_lstm(rnn_size, layer_size)

        with tf.device('/cpu:0'):
            input_x_embedded = tf.nn.embedding_lookup(encoder_embedding, self.input_x)

        # dynamic_rnn输入encoder实例、input embedding向量
        # 输出包含每一时刻的hidden_state(encoder_outputs)、最后时刻的hidden state(encoder_state)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder, input_x_embedded, dtype=tf.float32)

        # define helper for decoder
        if is_inference:
            self.start_tokens = tf.placeholder(tf.int32, shape=[None], name='start_tokens')
            self.end_tokens = tf.placeholder(tf.int32, name='end_tokens')
            # 定义用于inference阶段的helper：将output输出后的logits使用argmax获取id后再经过embedding layer获取下一时刻的输出
            helper = GreedyEmbeddingHelper(decoder_embedding, self.start_tokens, self.end_tokens)
        else:
            self.target_ids = tf.placeholder(tf.int32, shape=[None,None], name='target_ids')
            self.decoder_seq_length = tf.placeholder(tf.int32, shape=[None], name='batch_seq_length')
            with tf.device('/cpu:0'):
                target_embeddings = tf.nn.embedding_lookup(decoder_embedding, self.target_ids)
            helper = TrainingHelper(target_embeddings, self.decoder_seq_length)

        with tf.variable_scope('decoder'):
            fc_layer = Dense(decoder_vocab_size)
            decoder_cell = self._get_simple_lstm(rnn_size, layer_size)
            decoder = BasicDecoder(decoder_cell, helper, encoder_state, fc_layer)

        logits, final_state, final_sequence_lengths = dynamic_decode(decoder)

        if not is_inference:
            targets = tf.reshape(self.target_ids, [-1])
            logits_flat = tf.reshape(logits.rnn_output, [-1, decoder_vocab_size])
            print('logits_flat shape:{}'.format(logits_flat.shape))
            print('logits shape:{}'.format(logits.rnn_output.shape))

            self.cost = tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)

            # train op
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)

            optimizer = tf.train.AdamOptimizer(1e-3)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        else:
            self.prob = tf.nn.softmax(logits)

    def _get_simple_lstm(self, rnn_size, layer_size):
        lstm_layers = [tf.nn.rnn_cell.LSTMCell(rnn_size) for _ in range(layer_size)]
        return tf.nn.rnn_cell.MultiRNNCell(lstm_layers)
