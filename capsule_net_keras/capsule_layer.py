#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/16 13:08
# @Author  : MengnanChen
# @Site    : 
# @File    : capsule_layer.py
# @Software: PyCharm Community Edition

import keras.backend as K
import tensorflow as tf
from keras import layers

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

