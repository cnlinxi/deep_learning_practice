# -*- coding: utf-8 -*-
# @Time    : 2018/11/18 17:28
# @Author  : MengnanChen
# @FileName: simple_gan.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class SimpleGAN:
    def __init__(self):
        self.X = tf.placeholder(tf.float32, shape=(None, 784), name='X')
        self.Z = tf.placeholder(tf.float32, shape=(None, 100), name='Z')
        self.mnist = input_data.read_data_sets(train_dir='')

        self.mb_size = 28
        self.Z_dim = 1

        self.mn_size = 16

    def generator(self, z):
        with tf.variable_scope('Gen') as scope:
            G_h1 = tf.layers.dense(z, activation=tf.nn.relu)
            G_prob = tf.nn.sigmoid(tf.layers.dense(G_h1, activation=None))
            return G_prob

    def discriminator(self, x):
        '''
        discriminator output the probability that x is REAL
        '''
        with tf.variable_scope('Dis') as scope:
            D_h1 = tf.layers.dense(x, activation=tf.nn.relu)
            D_logits = tf.layers.dense(D_h1, activation=None)
            D_prob = tf.nn.sigmoid(D_logits)
            return D_prob, D_logits

    def sample_Z(self, m, n):
        return np.random.uniform(-1., 1., size=[m, n])

    def add_loss(self):
        G_sample = self.generator(self.Z)
        D_real, D_logit_real = self.discriminator(self.X)
        D_fake, D_logit_fake = self.discriminator(G_sample)

        self.D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
        self.G_loss = -tf.reduce_mean(tf.log(D_fake))

        # alternative losses
        # the raw article loss as followed, but I think this is a mistake...
        # refer to: https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
        # D_loss_real=tf.nn.sigmoid_cross_entropy_with_logits(labels=D_logit_real,logits=tf.ones_like(D_logit_real))
        # D_loss_fake=tf.nn.sigmoid_cross_entropy_with_logits(labels=D_logit_fake,logits=tf.zeros_like(D_logit_fake))
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        self.D_loss = D_loss_fake + D_loss_real
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_fake), logits=D_logit_fake))

    def train(self):
        all_variables = tf.trainable_variables()
        d_vars = [v for v in all_variables if 'Dis' in v.name]
        g_vars = [v for v in all_variables if 'Gen' in v.name]

        D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=d_vars)
        G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=g_vars)

        with tf.Session() as sess:
            for i in range(10000):
                X_mb, _ = self.mnist.train.next_batch(self.mn_size)

                _, cur_D_loss = sess.run([D_solver, self.D_loss],
                                         feed_dict={self.X: X_mb, self.Z: self.sample_Z(self.mb_size, self.Z_dim)})
                _, cur_G_loss = sess.run([G_solver, self.G_loss],
                                         feed_dict={self.Z: self.sample_Z(self.mb_size, self.Z_dim)})
