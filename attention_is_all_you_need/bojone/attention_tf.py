# -*- coding: utf-8 -*-
# @Time    : 2018/10/2 21:39
# @Author  : MengnanChen
# @FileName: attention_tf.py
# @Software: PyCharm

import tensorflow as tf


def position_embedding(inputs,position_size):
    batch_size,seq_len=tf.shape(inputs)[0],tf.shape(inputs)[1]
    position_j=1./tf.pow(1e4,2*tf.range(position_size/2.,dtype=tf.float32)/position_size)
    position_j=tf.expand_dims(position_j,axis=0)
    position_i=tf.range(tf.cast(seq_len,dtype=tf.float32),dtype=tf.float32)
    position_i=tf.expand_dims(position_i,axis=1)
    position_ij=tf.matmul(position_i,position_j)
    position_ij=tf.concat([tf.cos(position_ij),tf.sin(position_ij)],axis=1)
    position_embeddings=tf.expand_dims(position_ij,axis=0)+tf.zeros([batch_size,seq_len,position_size])
    return position_embeddings


'''
mode==mul: 全连接层之前，将超出部分全部置为0
mode==add: softmax之前，将超出部分减去一个很大的数，使得softmax之后值为0
'''
def mask(inputs,seq_len,mode='mul'):
    if seq_len==None:
        return inputs
    else:
        masks=tf.cast(tf.sequence_mask(seq_len),dtype=tf.float32)
        if mode=='mul':
            return inputs*masks
        if mode=='add':
            return inputs-(1-masks)*1e12


def dense(inputs,output_size,bias=True,seq_len=None):
    input_size=tf.shape(inputs)[-1]
    W=tf.Variable(tf.random_uniform([input_size,output_size],-0.05,0.05))
    if bias:
        b=tf.Variable(tf.random_uniform([output_size],-0.05,0.05))
    else:
        b=0.
    output=tf.matmul(inputs,tf.reshape(W,[-1,input_size]))
    output=tf.reshape(output,tf.concat(tf.shape(inputs)[-1:],[output_size]))
    if seq_len!=None:
        output=mask(output,seq_len,mode='mul')
    return output


def attention(Q,K,V,nb_head,size_per_head,Q_len=None,V_len=None):
    Q=dense(Q,nb_head*size_per_head,False)
    Q=tf.reshape(Q,[-1,tf.shape(Q)[1],nb_head,size_per_head])
    Q=tf.transpose(Q,[0,2,1,3])
    K=dense(K,nb_head*size_per_head,False)
    K=tf.reshape(K,[-1,tf.shape(K)[1],nb_head,size_per_head])
    K=tf.transpose(K,[0,2,1,3])
    V=dense(V,nb_head*size_per_head,False)
    V=tf.reshape(V,[-1,tf.shape(V)[-1],nb_head,size_per_head])
    V=tf.transpose(V,[0,2,1,3])
    A=tf.matmul(Q,K,transpose_b=True)/tf.sqrt(size_per_head)
    A=tf.transpose(A,[0,3,2,1])
    A=mask(A,V_len,mode='add')
    A=tf.transpose(A,[0.3,2,1])
    A=tf.nn.softmax(A)
    O=tf.matmul(A,V)
    O=tf.transpose(O,[0,2,1,3])
    O=tf.reshape(O,[-1,tf.shape(O)[1],nb_head*size_per_head])
    O=mask(O,Q_len,mode='mul')
    return O