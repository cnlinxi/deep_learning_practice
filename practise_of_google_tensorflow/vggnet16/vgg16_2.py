# -*- coding: utf-8 -*-
# @Time    : 2018/10/17 17:57
# @Author  : MengnanChen
# @FileName: vgg16_2.py
# @Software: PyCharm

import time
from datetime import datetime
import tensorflow as tf

batch_size=8

def conv_op(input_op,kh,kw,dh,dw,n_out,p,name):
    n_in=input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+'w',[kh,kw,n_in,n_out],
                               dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv=tf.nn.conv2d(input_op,kernel,strides=[1,dh,dw,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.,dtype=tf.float32,shape=[n_out]),trainable=True,name='b')
        bias=tf.nn.bias_add(conv,biases)
        activation=tf.nn.relu(bias)
        p+=[kernel,biases]

    return activation,p


def fc_op(input_op,n_out,p,name):
    n_in=input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+'w',[n_in,n_out],dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
        biases=tf.Variable(tf.constant(0.,dtype=tf.float32,shape=[n_out]),name='b')
        fc=tf.nn.relu_layer(input_op,kernel,biases)
        p+=[kernel,biases]

    return fc,p


def maxpool_op(input_op,kh,kw,bh,bw,name):
    return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],
                          strides=[1,bh,bw,1],
                          padding='SAME',
                          name=name)


def inference(images,keep_prob):
    parameters=[]

    conv1_1,parameters=conv_op(images,3,3,1,1,64,parameters,'conv1_1')
    conv1_2,parameters=conv_op(conv1_1,3,3,1,1,64,parameters,'conv1_2')
    pool1=maxpool_op(conv1_2,2,2,2,2,'pool1')

    conv2_1,parameters=conv_op(pool1,3,3,1,1,128,parameters,'conv2_1')
    conv2_2,parameters=conv_op(conv2_1,3,3,1,1,128,parameters,'conv2_2')
    pool2=maxpool_op(conv2_2,2,2,2,2,'pool2')

    conv3_1,parameters=conv_op(pool2,3,3,1,1,256,parameters,'conv3_1')
    conv3_2,parameters=conv_op(conv3_1,3,3,1,1,256,parameters,'conv3_2')
    pool3=maxpool_op(conv3_2,2,2,2,2,'pool3')

    conv4_1,parameters=conv_op(pool3,3,3,1,1,512,parameters,'conv4_1')
    conv4_2,parameters=conv_op(conv4_1,3,3,1,1,512,parameters,'conv4_2')
    pool4=maxpool_op(conv4_2,2,2,2,2,'pool4')

    conv5_1,parameters=conv_op(pool4,3,3,1,1,512,parameters,'conv5_1')
    conv5_2,parameters=conv_op(conv5_1,3,3,1,1,512,parameters,'conv5_2')
    pool5=maxpool_op(conv5_2,2,2,2,2,'pool5')

    resh=pool5.get_shape()
    shape1=resh[0].value*resh[1].value*resh[2].value
    shaped1=tf.reshape(pool5,[-1,shape1],name='reshape1')

    fc6,parameters=fc_op(shaped1,4096,parameters,name='fc6')
    fc6_drop=tf.nn.dropout(fc6,keep_prob,name='fc6_drop')

    fc7,parameters=fc_op(fc6_drop,4096,parameters,'fc7')
    fc7_drop=tf.nn.dropout(fc7,keep_prob,name='f7_drop')

    fc8,parameters=fc_op(fc7_drop,1000,parameters,'fc8')

    prediction=tf.argmax(tf.nn.softmax(fc8))
    return prediction,fc8,parameters


def time_tensorflow_run(sess,target,info_string,feed_dict):
    total_duration=0.
    num_steps=20
    num_steps_burn_in=10
    for i in range(num_steps+num_steps_burn_in):
        start_time=time.time()
        _=sess.run(target,feed_dict=feed_dict)
        duration=time.time()-start_time
        if i>=num_steps_burn_in:
            print(f'{datetime.now()}: step {i-num_steps_burn_in}, duration: {duration}')
            total_duration += duration

    print(f'{datetime.now()}: {info_string} across {num_steps} total duration: {total_duration}')


def run_benchmark():
    with tf.Graph().as_default():
        image_size=128
        images=tf.Variable(tf.random_normal(
            [batch_size,image_size,image_size,3],
            stddev=0.1,
            dtype=tf.float32
        ))

        # 由于后面feed_dict中需要引用keep_prob, 因此需要在这里声明keep_prob(tf.placeholder)
        keep_prob=tf.placeholder(dtype=tf.float32)
        prediction, fc8, parameters=inference(images,keep_prob)

        init=tf.global_variables_initializer()
        sess=tf.Session()
        sess.run(init)

        feed_dict={keep_prob:1.}
        time_tensorflow_run(sess,prediction,'forward',feed_dict)

        feed_dict={keep_prob:0.5}
        loss=tf.nn.l2_loss(fc8)
        grads=tf.gradients(loss,parameters)
        time_tensorflow_run(sess,grads,'forward_backward',feed_dict)


if __name__ == '__main__':
    run_benchmark()



