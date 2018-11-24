# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 14:24
# @Author  : MengnanChen
# @FileName: mnist_train.py
# @Software: PyCharm

import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

REGULARIZATION_RATE=1e-3
MOVING_AVERAGE_DECAY=0.99
BATCH_SIZE=32
LEARNING_RATE_DECAY=0.99
LEARNING_RATE_BASE=0.8
TRAIN_STEPS=int(1e4)
MODEL_SAVE_PATH='model'
MODEL_NAME='model.ckpt'


def train(mnist):
    x=tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name='x_input')
    y_=tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name='y_input')

    regularizer=tf.contrib.layer.l2_regularizer()
    y=mnist_inference.inference(x,regularizer)
    global_step=tf.Variable(0.,trainable=False)

    # 变量滑动平均
    variable_averges=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variables_average_op=variable_averges.apply(tf.trainable_variables())

    # 损失
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,axis=1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))

    # 学习率指数降低
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,
                                             decay_steps=mnist.train.num_examples/BATCH_SIZE,decay_rate=LEARNING_RATE_DECAY)

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variables_average_op]):  # 控制计算流图的顺序和依赖关系
        train_op=tf.no_op(name='train')  # tf.no_op什么都不做

    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer()
        for i in range(TRAIN_STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,loss,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%int(1e3)==0:
                print('after {} steps, loss on training: {}'.format(i,loss))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)


def main():
    mnist=input_data.read_data_sets()
    train(mnist)


if __name__ == '__main__':
    tf.app.run()