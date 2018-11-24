# -*- coding: utf-8 -*-
# @Time    : 2018/10/17 9:43
# @Author  : MengnanChen
# @FileName: alexnet.py
# @Software: PyCharm

from datetime import datetime
import time
import math
import tensorflow as tf


batch_size=8


def print_activations(t):
    print(f'{t.op.name}\t{t.get_shape().as_list()}')


def inference(images):
    parameters=[]

    with tf.name_scope('conv1') as scope:
        kernel=tf.Variable(tf.truncated_normal([11,11,3,64],stddev=1e-1,dtype=tf.float32),name='weights')
        conv=tf.nn.conv2d(images,kernel,strides=[1,4,4,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[64]),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv1=tf.nn.relu(bias,name=scope)
        print_activations(conv1)
        parameters+=[kernel,biases]

        lrn1=tf.nn.lrn(conv1,4,bias=1.,alpha=1e-2/9.,beta=0.75,name='lrn1')
        pool1=tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1')
        print_activations(pool1)

    with tf.name_scope('conv2') as scope:
        kernel=tf.Variable(tf.truncated_normal([5,5,64,192],stddev=1e-1,dtype=tf.float32),name='weights')
        conv=tf.nn.conv2d(pool1,kernel,strides=[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.,dtype=tf.float32,shape=[192]),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv2=tf.nn.relu(bias,name=scope)
        print_activations(conv2)
        parameters+=[kernel,biases]

        lrn2=tf.nn.lrn(conv2,4,bias=1e-2/9.,beta=0.75,name='lrn2')
        pool2=tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool2')
        print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,192,256],stddev=1e-1,dtype=tf.float32),name='weights')
        conv=tf.nn.conv2d(pool2,kernel,strides=[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.,dtype=tf.float32,shape=[256]),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv3=tf.nn.relu(bias,name=scope)
        print_activations(conv3)
        parameters+=[kernel,biases]

    with tf.name_scope('conv4') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,256,384],stddev=1e-1,dtype=tf.float32),name='weights')
        conv=tf.nn.conv2d(conv3,kernel,strides=[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.,dtype=tf.float32,shape=[384]),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv4=tf.nn.relu(bias,name=scope)
        print_activations(conv4)
        parameters+=[kernel,biases]

    with tf.name_scope('conv5') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,384,256],stddev=1e-1,dtype=tf.float32),name='weights')
        conv=tf.nn.conv2d(conv4,kernel,strides=[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.,dtype=tf.float32,shape=[256]),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv5=tf.nn.relu(bias,name=scope)
        print_activations(conv5)
        parameters+=[kernel,biases]

        pool3=tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool3')
        print_activations(pool3)

    return pool3,parameters


def time_tensorflow_run(sess,target,info_string):
    num_steps_burn_in=10
    num_batches=30
    total_duration=0.
    total_duration_squared=0.

    for i in range(num_steps_burn_in+num_batches):
        start_time=time.time()
        _=sess.run(target)
        duration=time.time()-start_time
        if i>=num_steps_burn_in:
            if not i%10:
                print(f'{datetime.now()}, steps:{i-num_steps_burn_in}, duration:{duration}')
            total_duration+=duration
            total_duration_squared+=duration*duration

    mn=total_duration/num_batches
    vr=total_duration_squared/num_batches-mn*mn
    sd=math.sqrt(vr)
    print(f'{datetime.now()}: {info_string} across {num_batches} steps, {mn} +/- {sd} sec/batch')


def run_benchmark():
    with tf.Graph().as_default():
        image_size=224
        images=tf.Variable(tf.random_normal([batch_size,
                                             image_size,
                                             image_size,3],
                                            dtype=tf.float32,
                                            stddev=1e-1))
        pool3,parameters=inference(images)

        init=tf.global_variables_initializer()
        sess=tf.Session()
        sess.run(init)

        time_tensorflow_run(sess,pool3,'FORWARD')

        objective=tf.nn.l2_loss(pool3)
        grad=tf.gradients(objective,parameters)
        time_tensorflow_run(sess,grad,'FORWARD-BACKWARD')


if __name__ == '__main__':
    run_benchmark()
