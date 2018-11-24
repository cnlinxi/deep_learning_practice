# -*- coding: utf-8 -*-
# @Time    : 2018/11/21 14:08
# @Author  : MengnanChen
# @FileName: train.py
# @Software: PyCharm

import sys

sys.path.append('..')
import os

import tensorflow as tf


def train(log_dir):
    save_dir = os.path.join(log_dir, 'ckpt')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
