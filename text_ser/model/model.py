# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 18:40
# @Author  : MengnanChen
# @FileName: model.py
# @Software: PyCharm

import tensorflow as tf

from .transformer_modules import *
from . import hparams as hp


class TextTransformer:
    def __init__(self, mode='train'):
        self._training = (mode == 'train')

    def build_model(self, elements, feeder):
        # input_x: [batch_size,seq_len]
        # input_lengths: [batch_size,]
        # input_y: [batch_size,]
        self.input_x, self.input_lengths, self.input_y = elements
        x = self.input_x

        vocab_size = feeder.vocab_size
        # [batch_size,seq_len,embedding_size]
        x = embedding(x, vocab_size, initial_table=feeder.w2v_table.astype(np.float32),
                      zero_pad=False)
        # [batch_size,seq_len,embedding_resize]
        x = tf.layers.dense(x, units=hp.embedding_resize, activation=tf.nn.relu)

        with tf.variable_scope('pos_enc'):
            # [batch_size,seq_len,embedding_resize]
            pos_enc = positional_encoding(
                batch_size=hp.batch_size,
                seq_len=feeder.max_seq_len,
                num_units=hp.embedding_resize,
                zero_pad=False,
                scale=False
            )
            x = x + pos_enc

            x = tf.layers.dropout(
                inputs=x,
                rate=hp.drop_rate,
                training=self._training
            )

        with tf.variable_scope('multi_head'):
            for i in range(hp.transformer_num_blocks):
                with tf.variable_scope('num_blocks_{}'.format(i)):
                    # [batch_size,seq_len,transformer_hidden_units]
                    # outputs += queries in <multihead_attention> function, so we must set num_units to query.shape[-1]?
                    x = multihead_attention(
                        queries=x,
                        keys=x,
                        num_units=hp.embedding_resize,
                        num_heads=hp.transformer_num_heads,
                        dropout_rate=hp.drop_rate,
                        is_training=self._training,
                        causality=False
                    )
                    x = feedforward(
                        inputs=x,
                        num_units=(4 * hp.embedding_resize, hp.embedding_resize)
                    )

        with tf.variable_scope('global_pool'):
            x = tf.reduce_max(x, reduction_indices=[1])
            x = tf.reshape(x, (-1, hp.embedding_resize))  # [batch_size,embedding_resize]

        with tf.variable_scope('final_op'):
            self.logits = tf.layers.dense(x, hp.n_classes)

    def add_loss(self):
        # Rank mismatch: Rank of labels should equal rank of logits minus 1.
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                       logits=self.logits)
        all_vars = tf.trainable_variables()
        regularization = tf.add_n([tf.nn.l2_loss(v) for v in all_vars
                                   if not ('bias' in v.name or 'Bias' in v.name)]) * hp.reg_weight
        self.loss = tf.reduce_mean(cross_entropy) + regularization

    def add_metric(self):
        with tf.variable_scope('metric'):
            pred = tf.argmax(tf.nn.softmax(self.logits), axis=-1,output_type=tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, self.input_y), dtype=tf.float32))

    def add_optimizer(self, global_step):
        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=hp.initial_learning_rate,
                decay_rate=hp.decay_rate,
                decay_steps=hp.decay_steps,
                global_step=global_step
            )
            optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                               beta1=hp.adam_beta1,
                                               beta2=hp.adam_beta2,
                                               epsilon=hp.adam_epsilon)

            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            if hp.clip_gradients:
                clipped_gradent, _ = tf.clip_by_global_norm(gradients, clip_norm=1.)
            else:
                clipped_gradent = gradients
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradent, variables),
                                                          global_step=global_step)
