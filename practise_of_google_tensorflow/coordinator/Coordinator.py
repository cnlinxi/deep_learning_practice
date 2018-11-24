# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 9:21
# @Author  : MengnanChen
# @FileName: Coordinator.py
# @Software: PyCharm
# refer to: https://blog.csdn.net/dcrmg/article/details/79780331

'''
tensorflow 支持多线程，可以在同一会话内创建多个线程并行执行。
在Session中所有线程必须能够同步终止，异常必须能正确捕获并报告

queue -> tf.train.batch
tf.train.Coordinator -> sess.run(batch)
'''

import tensorflow as tf
import numpy as np

sample_num=5
epoch_num=2
batch_size=3
batch_total=sample_num//batch_size+1


def generate_data(sample_num=sample_num):
    labels=np.asarray(range(0,sample_num))
    images=np.random.random([sample_num,224,224,3])
    return images,labels


def get_batch_data(batch_size=batch_size):
    images,labels=generate_data(sample_num)
    images=tf.cast(images,tf.int32)
    labels=tf.cast(labels,tf.int32)

    # 从tensor列表中按顺序或随机抽取tensor放入queue
    input_queue=tf.train.slice_input_producer([images,labels],num_epochs=epoch_num)

    # 将输入队列input_queue和batch_size传入tf.train.batch以获取一个batch的数据
    image_batch,label_batch=tf.train.batch(input_queue,batch_size=batch_size)
    return image_batch,label_batch

image_batch,label_batch=get_batch_data(batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 开启一个tensorflow线程管理器
    coord=tf.train.Coordinator()
    # 使用start_queue_runners启动队列填充
    threads=tf.train.start_queue_runners(sess,coord)

    try:
        while not coord.should_stop():
            # 启动会话，获取真正的数据
            image_batch_v,label_batch_v=sess.run([image_batch,label_batch])
            print(image_batch_v.shape,label_batch_v)
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
        print('all threads are asked to stop!')
    # 将开启的线程加入主线程，等待threads结束
    coord.join(threads)
