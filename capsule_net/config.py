#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/6 13:39
# @Author  : MengnanChen
# @Site    : 
# @File    : config.py
# @Software: PyCharm Community Edition

import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_value', 0.5, 'down weight of the loss for absent digit classes')

# for training
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('epoch', 5, 'epoch')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule or not')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.392, 'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')


############################
#   environment setting    #
############################
flags.DEFINE_string('dataset', 'mnist', 'The name of dataset [mnist, fashion-mnist')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing examples')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_integer('train_sum_freq', 100, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_freq', 500, 'the frequency of saving valuation summary(step)')
flags.DEFINE_integer('save_freq', 3, 'the frequency of saving model(epoch)')
flags.DEFINE_string('result', 'result', 'path for saving results')

############################
#   distributed setting    #
############################
flags.DEFINE_integer('num_gpu', 2, 'number of gpus for distributed training')
flags.DEFINE_integer('batch_size_per_gpu', 128, 'batch size on 1 gpu')
flags.DEFINE_integer('thread_per_gpu', 4, 'Number of preprocessing threads per tower.')

cfg = tf.app.flags.FLAGS