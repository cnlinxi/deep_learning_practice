#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/6 13:37
# @Author  : MengnanChen
# @Site    : 
# @File    : capsule_net.py
# @Software: PyCharm Community Edition

import tensorflow as tf
from capsule_layer import capsule_layer
from config import cfg
from utils import get_batch_data

epsilon=1e-9

class capsule_net(object):
    def __init__(self,is_train=True):
        self.graph=tf.Graph()
        with self.graph.as_default():
            if is_train:
                self.X,self.labels=get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads)
                self.Y=tf.one_hot(self.labels,depth=10,axis=1,dtype=tf.float32)

                self.build_arch()
                self.loss()
                self._summary()

                self.global_step=tf.Variable(0,name='global_step',trainable=False)
                self.optimizer=tf.train.AdamOptimizer()
                self.train_op=self.optimizer.minimize(self.total_loss,global_step=self.global_step)
            else:
                self.X=tf.placeholder(dtype=tf.float32,shape=(cfg.batch_size,28,28,1))
                self.labels=tf.placeholder(dtype=tf.int32,shape=(cfg.batch_size,))
                self.Y=tf.reshape(self.labels,shape=(cfg.batch_size,10,1))
                self.build_arch()

        tf.logging.info('Setting up the main structure')

    def build_arch(self):
        with tf.variable_scope('conv1_layer'):
            conv1=tf.contrib.layers.conv2d(self.X,num_outputs=256,kernel_size=9,
                                           stride=1,padding='VALID')
            # print(conv1.get_shape()) # (?, 20, 20, 256)
            # assert conv1.get_shape()==[cfg.batch_size,20,20,256]

        with tf.variable_scope('primary_capsule_layer'):
            primary_capsule=capsule_layer(num_outputs=32,vec_length=8,with_routing=False,layer_type='CONV')
            caps1=primary_capsule(conv1,kernel_size=9,stride=2)
            # assert caps1.get_shape()==[cfg.batch_size,1152,8,1]

        with tf.variable_scope('digit_capsule_layer'):
            digit_caps=capsule_layer(num_outputs=10,vec_length=16,with_routing=True,layer_type='FC')
            self.caps2=digit_caps(caps1)

        # Decoder
        # Do masking
        with tf.variable_scope('masking'):
            # 计算||v||，然后对||v||的axis=1做softmax
            # [None,10,16,1] -> [None,10,1,1]
            self.v_length=tf.sqrt(tf.reduce_sum(tf.square(self.caps2),axis=2,keep_dims=True)+epsilon)
            self.softmax_v=tf.nn.softmax(self.v_length,dim=1)
            # assert self.softmax_v.get_shape()==[cfg.batch_size,10,1,1]

            # 找到softmax中最大的idx[[0.6,0.4],[0.3,0.7]]->[0,1]
            # [None,10,1,1]->[None]
            self.argmax_idx=tf.to_int32(tf.argmax(self.softmax_v,axis=1))
            # assert self.argmax_idx.get_shape()==[cfg.batch_size,1,1]
            self.argmax_idx=tf.reshape(self.argmax_idx,shape=(cfg.batch_size,))
            # 这段完全不能理解
            # Method 1.
            if not cfg.mask_with_y:
                # c). indexing
                # It's not easy to understand the indexing process with argmax_idx
                # as we are 3-dim animal
                masked_v = []
                for batch_size in range(cfg.batch_size):
                    v = self.caps2[batch_size][self.argmax_idx[batch_size], :]
                    masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

                self.masked_v = tf.concat(masked_v, axis=0)
                # assert self.masked_v.get_shape() == [cfg.batch_size, 1, 16, 1]
            # Method 2. masking with true label, default mode
            else:
                # self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)), transpose_a=True)
                self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)))
                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2,keep_dims=True) + epsilon)

        with tf.variable_scope('decoder'):
            # 使用3层全连接层重建图像
            # [None,1,16,1]=>[None,16]=>[None,512]=>[None,1024]=>[None,784]
            # 原MNIST数据集，每张Image大小:28x28=784
            masked_v=tf.reshape(self.masked_v,(cfg.batch_size,-1))
            # assert masked_v.get_shape()==[cfg.batch_size,16]
            fc1=tf.contrib.layers.fully_connected(masked_v,num_outputs=512)
            # assert fc1.get_shape()==[cfg.batch_size,512]
            fc2=tf.contrib.layers.fully_connected(fc1,num_outputs=1024)
            # assert fc2.get_shape()==[cfg.batch_size,1024]
            self.decoded=tf.contrib.layers.fully_connected(fc2,num_outputs=784,activation_fn=tf.sigmoid)

    def loss(self):
        # margin_loss
        # (max(0,m^{+} - ||v_k||))^2
        max_l=tf.square(tf.maximum(0.,cfg.m_plus-self.v_length))
        # (max(0,||v_k||-m^{-}))^2
        max_r=tf.square(tf.maximum(0.,self.v_length-cfg.m_minus))

        # T_k:[None,10]
        T_k=self.Y
        # L_k:[None,10]
        L_k=T_k*max_l+cfg.lambda_value*(1-T_k)*max_r
        # 公式(4)
        self.margin_loss=tf.reduce_mean(tf.reduce_sum(L_k, axis=1))

        # 重建损失
        origin=tf.reshape(self.X,shape=(cfg.batch_size,-1))
        reconstruction_loss_squared=tf.square(self.decoded-origin)
        self.reconstruction_loss=tf.reduce_mean(reconstruction_loss_squared)

        self.total_loss=self.margin_loss+self.reconstruction_loss

    def _summary(self):
        train_summary=[]
        train_summary.append(tf.summary.scalar('train/margin_loss',self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss',self.reconstruction_loss))
        train_summary.append(tf.summary.scalar('train/total_loss',self.total_loss))
        reconstruction_image=tf.reshape(self.decoded,shape=(cfg.batch_size,28,28,1)) # MNIST Image size:28x28x1
        train_summary.append(tf.summary.image('reconstruction_image',reconstruction_image))
        self.train_summary=tf.summary.merge(train_summary)

        correct_prediction=tf.equal(tf.to_int32(self.labels),self.argmax_idx)
        self.accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # raw repo 中此处为tf.reduce_sum