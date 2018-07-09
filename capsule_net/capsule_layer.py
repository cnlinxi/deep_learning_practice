#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/6 17:16
# @Author  : MengnanChen
# @Site    : 
# @File    : capsule_layer.py
# @Software: PyCharm Community Edition

import tensorflow as tf
from config import cfg
import numpy as np

epsilon=1e-9

class capsule_layer(object):
    def __init__(self,num_outputs,vec_length,with_routing,layer_type):
        self.num_outputs=num_outputs
        self.vec_length=vec_length
        self.with_routing=with_routing
        self.layer_type=layer_type

    def __call__(self, input,kernel_size=None,stride=None):
        if self.layer_type=='CONV': # 如果使用的layer是conv，那么就使用kernel_size & stride
            # PrimaryCaps，一个卷积层
            # input:[None,20,20,256]
            self.kenel_size=kernel_size
            self.stride=stride

            if not self.with_routing:
                # assert input.get_shape()==[cfg.batch_size,20,20,256] # 确保数据从第一层普通的卷积层过来
                # num_outputs->number of channels/number of filters, vec_length->capsule dimensions
                capsules=tf.contrib.layers.conv2d(inputs=input,num_outputs=self.num_outputs*self.vec_length,
                                           padding='VALID',kernel_size=self.kenel_size,stride=self.stride)
                capsules=tf.reshape(capsules,(cfg.batch_size,-1,self.vec_length,1))
                capsules=squash(capsules)
                # assert capsules.get_shape()==[cfg.batch_size,1152,8,1]
                return capsules

        if self.layer_type=='FC':
            # DigitCaps, 一个全连接层
            # input:[None,1152,8,1]
            self.input=tf.reshape(input,(cfg.batch_size,-1,1,input.shape[-2].value,1)) # reshape to [None,1152,1,8,1]
            with tf.variable_scope('routing'):
                # b_IJ:[batch_size, num_caps_l, num_caps_l_plus_1, 1, 1]->[batch_size,1152,32,1,1]
                # num_caps_l:number of capsule
                # len(u_i):number of conv filter in raw conv-layer
                # b_IJ=tf.constant(np.zeros([cfg.batch_size,input.shape[1].value,self.num_outputs,1,1],dtype=np.float32))
                b_IJ=tf.constant(np.zeros([cfg.batch_size,1152,self.num_outputs,1,1],dtype=np.float32))
                capsules=routing(self.input,b_IJ)
                capsules=tf.squeeze(capsules,axis=1)
            return capsules

def routing(input,b_IJ):
    # input: [None,num_caps_l=1152,1,len(u_i)=8,1]
    # W: [1,num_caps_j*len_v_j,len_u_j,1]
    W=tf.get_variable('Weight',shape=(1,1152,160,8,1),dtype=tf.float32,
                      initializer=tf.random_normal_initializer(stddev=cfg.stddev))
    biases=tf.get_variable('bias',shape=(1,1,10,16,1))

    # 公式（2），此处原库使用的是一种特殊的矩阵乘方法，以节省计算资源，不大看懂
    # \hat{u}_{j|i}=W_{ij}u_i
    # caculate u_hat
    # in raw repository, matmul is time-consuming
    # matmul [a,b]x[b,c] is equal to
    # element-wise multiply [a*c,b]*[a*c,b], reduce sum at axis=1 and reshape to [a,c]
    input=tf.tile(input,[1,1,160,1,1])
    # assert input.get_shape()==[cfg.batch_size,1152,160,8,1]

    u_hat=tf.reduce_sum(W*input,axis=3,keep_dims=True)
    u_hat=tf.reshape(u_hat,shape=[-1,1152,10,16,1])
    # assert u_hat.get_shape()==[cfg.batch_size,1152,10,16,1]

    # In forward, u_hat_stopped=u_hat, in backward, no gradient passed back from u_hat_stopped to u_hat
    # 在内部循环时，不反向传播，在循环最后一次才使用反向传播更新W
    u_hat_stopped=tf.stop_gradient(u_hat,name='stop_gradient')

    for r_iter in range(cfg.iter_routing):
        with tf.variable_scope('iter_'+str(r_iter)):
            c_IJ=tf.nn.softmax(b_IJ,dim=2) # c_j <- softmax(b_i)
            if r_iter==cfg.iter_routing-1:
                s_J=tf.multiply(c_IJ,u_hat) # s_j <- c_{ij}\hat{u_{j|i}}
                s_J=tf.reduce_sum(s_J,axis=1,keep_dims=True)+biases # s_j=\sum_j s_j
                # assert s_J.get_shape()==[cfg.batch_size,1,10,16,1]
                v_J=squash(s_J) # v_j <- squash(s_j)
                # assert v_J.get_shape()==[cfg.batch_size,1,10,16,1]
            elif r_iter<cfg.iter_routing-1: # 内部循环时，不使用反向传播
                s_J=tf.multiply(c_IJ,u_hat_stopped)
                s_J=tf.reduce_sum(s_J,axis=1,keep_dims=True)+biases
                v_J=squash(s_J)

                v_J_tiled=tf.tile(v_J,[1,1152,1,1,1])
                u_product_v=tf.reduce_sum(u_hat_stopped*v_J_tiled,axis=3,keep_dims=True) # \hat{u_{j|i}}v_j
                # assert u_product_v.get_shape()==[cfg.batch_size,1152,10,1,1]

                b_IJ+=u_product_v
    return v_J


def squash(vector):
    '''
    原文公式(1)
    '''
    vector_norm=tf.reduce_sum(tf.square(vector),axis=-2,keep_dims=True) # {||s_j||}^2
    scalar_factor=vector_norm/(vector_norm+1)/tf.sqrt(vector_norm+epsilon) # epsilon 为小数，防止除以0
    vec_squashed=scalar_factor*vector
    return vec_squashed