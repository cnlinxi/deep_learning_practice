# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 15:29
# @Author  : MengnanChen
# @FileName: mnist_eval.py
# @Software: PyCharm

import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 100


def evaluate(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='input_y')
    y = mnist_inference.inference(x, None)
    # 定义计算正确率的计算图
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    variable_average = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_average.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.mdoel_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
                print('step {}, validation accuracy: {}'.format(global_step, acc))
            else:
                print('no checkpoint file exist.')
            time.sleep(EVAL_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets()
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
