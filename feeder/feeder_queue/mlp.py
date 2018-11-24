# -*- coding: utf-8 -*-
# @Time    : 2018/11/21 12:00
# @Author  : MengnanChen
# @FileName: mlp.py
# @Software: PyCharm

import tensorflow as tf


class MLP:
    def __init__(self, mode):
        self._training = (mode == 'train')
        self._drop_rate = 0.5
        self._layer_units = (256, 512, 128)
        self._layer_size = len(self._layer_units)
        self.n_classes=4
        self._reg_weight=1e-3
        self._initial_learning_rate=1e-3
        self._decay_rate=0.96
        self._decay_steps=10000
        self._adam_beta1=0.9
        self._adam_beta2=0.999
        self._adam_epsilon=1e-6
        self._clip_gradients=True  # in case of gradient explode

    def bulid_model(self,element):
        '''
        :param input_x: [batch_size,6553]
        :param input_y: [batch_size,]
        :return:
        '''
        input_x, input_y=element
        x=input_x
        with tf.variable_scope('mlp'):
            for i in range(self._layer_size):
                x=tf.layers.dense(x,units=self._layer_units[i],activation=tf.nn.relu)
                if self._training:
                    x=tf.layers.dropout(x,rate=self._drop_rate)

        with tf.variable_scope('logits'):
            self.logits=tf.layers.dense(x,units=self.n_classes,activation=None)
        self.labels=input_y

    def add_loss(self):
        with tf.variable_scope('loss'):
            all_vars=tf.trainable_variables()
            reg=tf.add_n([tf.nn.l2_loss(v) for v in all_vars if 'bias' not in v.name])*self._reg_weight
            cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,logits=self.logits)
            self.loss=tf.reduce_mean(cross_entropy)+reg

    def add_metric(self):
        with tf.variable_scope('metrics'):
            pred=tf.nn.softmax(self.logits)
            pred=tf.argmax(pred,axis=-1)
            correct_prediction=tf.equal(pred,self.labels)
            self.accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    def add_optimizer(self,global_step):
        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=self._initial_learning_rate,
                global_step=global_step,
                decay_rate=self._decay_rate,
                decay_steps=self._decay_steps
            )
            optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                               beta1=self._adam_beta1,
                                               beta2=self._adam_beta2,
                                               epsilon=self._adam_epsilon)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            if self._clip_gradients:
                clip_gradient, _ = tf.clip_by_global_norm(gradients, 1.)
            else:
                clip_gradient = gradients

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clip_gradient, variables),
                                                          global_step=global_step)



