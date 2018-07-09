#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/8 20:41
# @Author  : MengnanChen
# @Site    : 
# @File    : utils.py
# @Software: PyCharm Community Edition

import os
import numpy as np
import tensorflow as tf
from scipy import misc

def load_mnist(batch_size,is_train=True):
    path=os.path.join('data','mnist')
    if is_train:
        fp = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded=np.fromfile(file=fp,dtype=np.int8)
        trainX=loaded[16:].reshape((60000,28,28,1)).astype(np.float32)

        fp=open(os.path.join(path,'train-labels-idx1-ubyte'))
        loaded=np.fromfile(file=fp,dtype=np.int8)
        trainY=loaded[8:].reshape((60000)).astype(np.int32)

        trX=trainX[:55000]/255.
        trY=trainY[:55000]

        valX=trainX[55000:,]/255.
        valY=trainY[55000:]

        num_train_batch=55000//batch_size
        num_valid_batch=5000//batch_size

        return trX,trY,num_train_batch,valX,valY,num_valid_batch
    else:
        fp=open(os.path.join(path,'t10k-images-idx3-ubyte'))
        loaded=np.fromfile(file=fp,dtype=np.int8)
        teX=loaded[16:].reshape((10000,28,28,1)).astype(np.float32)

        teX=teX/255.

        fp=open(os.path.join(path,''))
        loaded=np.fromfile(file=fp,dtype=np.int8)
        teY=loaded[8:].reshape((10000)).astype(np.int32)

        num_test_batch=10000//batch_size
        return teX,teY,num_test_batch

def load_fashion_mnist(batch_size,is_train=True):
    path=os.path.join('data','fashion-mnist')
    if is_train:
        fp=open(os.path.join(path,'train-images-idx3-ubyte'))
        loaded=np.fromfile(file=fp,dtype=np.int8)
        trainX=loaded[16:].reshape((60000,28,28,1)).astype(np.float32)

        fp=open(os.path.join(path,'train-labels-idx1-ubyte'))
        loaded=np.fromfile(file=fp,dtype=np.int8)
        trainY=loaded[8:].reshape((60000)).astype(np.int32)

        trX=trainX[:55000]/255.
        trY=trainY[:55000]

        valX=trainX[55000:,]/255.
        valY=trainY[55000:]

        num_train_batch=55000//batch_size
        num_valid_batch=5000//batch_size

        return trX,trY,num_train_batch,valX,valY,num_valid_batch
    else:
        fp=open(os.path.join(path,'t10k-images-idx3-ubyte'))
        loaded=np.fromfile(file=fp,dtype=np.int8)
        teX=loaded[16:].reshape((10000,28,28,1)).astype(np.float32)

        teX=teX/255.

        fp=open(os.path.join(path,'t10k-labels-idx1-ubyte'))
        loaded=np.fromfile(file=fp,dtype=np.int8)
        teY=loaded[8:].reshape((10000)).astype(np.int32)

        num_test_batch=10000//batch_size
        return teX,teY,num_test_batch

def get_batch_data(dataset,batch_size,is_train=True):
    if dataset=='mnist':
        trX, trY, num_train_batch, valX, valY, num_valid_batch=load_mnist(batch_size,is_train)
    elif dataset=='fashion-mnist':
        trX, trY, num_train_batch, valX, valY, num_valid_batch=load_fashion_mnist(batch_size,is_train)
    data_queues=tf.train.slice_input_producer([trX,trY])
    X,Y=tf.train.shuffle_batch(data_queues,batch_size=batch_size,
                               capacity=batch_size*64,
                               min_after_dequeue=batch_size*32,
                               allow_smaller_final_batch=True)
    return X,Y

def load_data(dataset,batch_size,is_train=True):
    if dataset=='mnist':
        return load_mnist(batch_size,is_train)
    elif dataset=='fashion-mnist':
        return load_fashion_mnist(batch_size,is_train)
    else:
        raise Exception('Invaild dataset')

def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return misc.imsave(path, mergeImgs(imgs, size))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs