# -*- coding: utf-8 -*-
# @Time    : 2018/10/17 16:38
# @Author  : MengnanChen
# @FileName: vgg16.py
# @Software: PyCharm


import time
from datetime import datetime
import tensorflow as tf

batch_size=8

def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(name=scope + 'w', shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, strides=[1, dh, dw, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[n_out]), trainable=True, name='b')
        bias = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(bias, name=scope)
        p += [kernel, biases]

    return activation, p


def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(name=scope + 'w', shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
    return activation, p


def mpool_op(input_op, kh, kw, dh, dw, name):
    return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)


def inference(images,keep_prob):
    p=[]

    conv1_1,p=conv_op(images,'conv1_1',3,3,64,1,1,p)
    conv1_2,p=conv_op(conv1_1,'conv1_2',3,3,64,1,1,p)
    pool_1=mpool_op(conv1_2,2,2,2,2,'pool_1')

    conv2_1,p=conv_op(pool_1,'conv2_1',3,3,128,1,1,p)
    conv2_2,p=conv_op(conv2_1,'conv2_2',3,3,128,1,1,p)
    pool_2=mpool_op(conv2_2,2,2,2,2,'pool_2')

    conv3_1,p=conv_op(pool_2,'conv3_1',3,3,256,1,1,p)
    conv3_2,p=conv_op(conv3_1,'conv3_2',3,3,256,1,1,p)
    pool_3=mpool_op(conv3_2,2,2,2,2,'pool_3')

    conv4_1,p=conv_op(pool_3,'conv4_1',3,3,512,1,1,p)
    conv4_2,p=conv_op(conv4_1,'conv4_2',3,3,512,1,1,p)
    pool_4=mpool_op(conv4_2,2,2,2,2,'pool_4')

    conv5_1,p=conv_op(pool_4,'conv5_1',3,3,512,1,1,p)
    conv5_2,p=conv_op(conv5_1,'conv5_2',3,3,512,1,1,p)
    pool_5=mpool_op(conv5_2,2,2,2,2,'pool_5')

    shp=pool_5.get_shape()
    flattened_shape=shp[0].value*shp[1].value*shp[2].value

    resh1=tf.reshape(pool_5,[-1,flattened_shape],name='resh1')

    fc6,p=fc_op(resh1,name='fc6',n_out=4096,p=p)
    fc6_drop=tf.nn.dropout(fc6,keep_prob=keep_prob,name='fc6_drop')

    fc7,p=fc_op(fc6_drop,name='fc7',n_out=4096,p=p)
    fc7_drop=tf.nn.dropout(fc7,keep_prob=keep_prob,name='fc7_drop')

    fc8,p=fc_op(fc7_drop,name='fc8',n_out=1000,p=p)
    prediction=tf.argmax(tf.nn.softmax(fc8),axis=1)

    return prediction,fc8,p


def time_tensorflow_run(sess,target,info_string,feed_dict):
    num_steps_burn_in=10
    num_steps=20
    total_duration=0.
    for i in range(num_steps+num_steps_burn_in):
        start_time=time.time()
        _=sess.run(target,feed_dict=feed_dict)
        duration=time.time()-start_time
        if i>=num_steps_burn_in:
            if not i%10:
                print(f'{datetime.now()}: step {i-num_steps_burn_in}, duration={duration}')
        total_duration+=duration
    print(f'{datetime.now()}: {info_string} across {num_steps} total_duration: {total_duration}')


def run_benchmark():
    with tf.Graph().as_default():
        image_size=128
        images=tf.Variable(tf.random_normal(
            shape=[batch_size,image_size,image_size,3],
            stddev=0.1
        ))

        keep_prob=tf.placeholder(tf.float32)
        predictions,fc8,p=inference(images,keep_prob)

        init=tf.global_variables_initializer()
        sess=tf.Session()
        sess.run(init)

        time_tensorflow_run(sess,predictions,'FORWARD',{keep_prob:1.0})

        objective=tf.nn.l2_loss(fc8)
        grad=tf.gradients(objective,p)
        time_tensorflow_run(sess,grad,'FORWARD-BACKWARD',{keep_prob:0.5})


if __name__ == '__main__':
    run_benchmark()