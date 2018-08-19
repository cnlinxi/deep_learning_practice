#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/16 13:08
# @Author  : MengnanChen
# @Site    : 
# @File    : capsule_layer.py
# @Software: PyCharm Community Edition

import keras.backend as K
import tensorflow as tf
from keras import layers,initializers

class Length(layers.Layer):
    '''
    求矩阵的模
    inputs: shape:[None,num_capsules,dim_capsules]
    outputs: shape:[None,num_capsules]
    '''
    def __call__(self, inputs, **kwargs):
        return K.sqrt(K.sum((K.square(inputs)),axis=-1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config=super(Length,self).get_config()
        return config

class Mask(layers.Layer):
    '''
    将（最后一层）的“非激活”的神经元capsule给mask掉，将“激活”的神经元（模长最大，也就是表示概率最大）过滤出来
    '''
    def call(self, inputs, **kwargs):
        if type(inputs) is list: # 使用label来mask
            # inputs[0]: [None,num_capsules,dim_capsules]
            # inputs[1]: [None,num_capsules]
            assert len(inputs)==2
            inputs,mask=inputs
        else: # 使用capsule的模长来mask
            # inputs: [None,num_capsules,dim_capsules]
            x=K.sqrt(K.sum(K.square(inputs),axis=-1)) # 求capsule的模长
            mask=K.one_hot(indices=K.argmax(x,axis=1),num_classes=x.get_shape().as_list()[1])

        masked=K.batch_flatten(inputs*tf.expand_dims(mask,-1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple: # 使用label来mask
            return tuple([None,input_shape[0][1]*input_shape[0][2]])
        else:
            return tuple([None,input_shape[1]*input_shape[2]])

    def get_config(self):
        config=super(Mask,self).get_config()
        return config

def squash(vectors,axis=-1):
    '''
    对vectors做squash 函数的激活
    squash(x)=\frac{||x||^2}{1+||x||^2}*\frac{x}{||x||}
    '''
    s_squared_norm=K.sum(K.square(vectors),axis,keepdims=True) # ||x||^2
    constant=s_squared_norm/(1.0+s_squared_norm) # \frac{||x||^2}{1+||x||^2}
    return vectors/K.sqrt(s_squared_norm+K.epsilon())*constant

class CapsuleLayer(layers.Layer):
    def __init__(self,output_num_capsules,output_dim_capsules,routings=3,
                 kernel_initializer='glorot_uniform',**kwargs):
        super(CapsuleLayer,self).__init__(kwargs)
        self.output_num_capsules=output_num_capsules
        self.output_dim_capsules=output_dim_capsules
        self.routings=routings
        self.kernel_initializer=initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape)==3 # input tensor:[None, num_capsules, dim_capsules]
        self.input_num_capsules=input_shape[1]
        self.input_dim_capsules=input_shape[2]
        # 仿射矩阵，在动态路由中通过BP学习到的，乘以上一层的capsule，以实现“多角度”看特征
        self.W=self.add_weight(shape=[self.output_num_capsules,self.input_num_capsules,
                                      self.output_dim_capsules,self.input_dim_capsules],
                               initializer=self.kernel_initializer,name='W')
        self.built=True

    def __call__(self, inputs,training=None):
        # inputs: [None,input_num_capsules,input_dim_capsules]
        # inputs_expand:[None,1,input_num_capsules,input_dim_capsules]
        inputs_expand=K.expand_dims(inputs,axis=1) # 主要为了后续计算，扩展axis=1位置为接下来output_dim_capsules的位置
        # 相当于8个卷积并行，同时乘以W，相当于共享了权值W
        input_tiled=K.tile(inputs_expand,[1,self.output_dim_capsules,1,1])
        # x:[None,output_num_capsules,input_num_capsules,input_dim_capsules]
        # W:[None,output_num_capsules,input_num_capsules,output_dim_capsules,input_dim_capsules]
        # K.batch_dot(x,self.W,[2,3]): [input_dim_capsules] x [output_dim_capsules,input_dim_capsules]
        # K.map_fn(elems=...): ignore 'batch' dimension, so the axis=2 of x is: input_dim_capsules and W is this too.
        # [input_dim_capsules] x [output_dim_capsules,input_dim_capsules]=[ouput_dim_capsules]
        # inputs_hat: [None,output_num_capsules,input_num_capsules,output_dim_capsules]
        inputs_hat=K.map_fn(lambda x:K.batch_dot(x,self.W,[2,3]),elems=input_tiled) # \hat{u_{ij}}

        b=tf.zeros(shape=[K.shape(inputs_hat)[0],self.output_num_capsules,self.input_num_capsules]) # b_{ij}

        assert self.routings>0 # 动态路由迭代次数应大于0
        for i in range(self.routings):
            # c_i, 在axis=1维度即output_num_capsules维度上做softmax,使得各个output_capsule的"概率"加和为1，capsule内部一样
            # samples=3, num_capsules=2, dim_capsules=2, softmax之后的结果类似于：
            # [[[0.11,0.11],[0.89,0.89]],
            #  [[0.11,0.11],[0.89,0.89]],
            #  [[0.11,0.11],[0.89,0.89]]]
            # c: [None,output_dim_capsules,input_dim_capsules]
            c=tf.nn.softmax(b,axis=1)
            # [input_dim_capsules] x [input_dim_capsules,output_dim_capsules]=[output_dim_capsules]
            # outputs:[None,output_num_capsules,output_dim_capsules]
            outputs=squash(K.batch_dot(c,inputs_hat,[2,2])) # v_j
            if i<self.routings-1:
                # [output_dim_capsules] x [input_num_capsules,output_dim_capsules]^T=[input_num_capsules]
                # b: [None,output_num_capsules,input_num_capsules]
                b+=K.batch_dot(outputs,inputs_hat,[2,3])

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple(None,self.output_num_capsules,self.output_dim_capsules)

    def get_config(self):
        config={
            'output_num_capsules':self.output_num_capsules,
            'output_dim_capsules':self.output_dim_capsules,
            'routings':self.routings
        }
        base_config=super(CapsuleLayer,self).get_config()
        return dict(list(base_config)+list(config))

def PrimaryCap(inputs,dim_capsule,n_channels,kernel_size,strides,padding):
    output=layers.Conv2D(filters=dim_capsule*n_channels,kernel_size=kernel_size,strides=strides,
                         padding=padding)(inputs)
    output=layers.Reshape(target_shape=[-1,dim_capsule],name='primary_capsule_reshape')(output)
    return layers.Lambda(squash,name='primary_capsule_squash')(output)