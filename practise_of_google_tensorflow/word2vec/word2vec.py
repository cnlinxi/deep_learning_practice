# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 19:30
# @Author  : MengnanChen
# @FileName: word2vec.py
# @Software: PyCharm

import os
import collections
import math
import zipfile
import random

import urllib
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf

data_index=0
vocabulary_size = 50000


def maybe_download(filename,expected_bytes):
    url='http://mattmahoney.net/dc/'
    file_path=os.path.join('../data',filename)
    if not os.path.exists(file_path):
        print('start download...')
        filename,_=urllib.request.urlretrieve(url+filename,filename)
    statinfo=os.stat(file_path)
    if statinfo.st_size==expected_bytes:
        print('file exists and verify.')
    else:
        print(statinfo.st_size)
        raise Exception('failed to verify',filename)
    return file_path


def read_data(file_path):
    with zipfile.ZipFile(file_path) as fin:
        data=tf.compat.as_str(fin.read(fin.namelist()[0])).split()
    return data


def build_dataset(words):
    count=[['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary=dict()
    for word,_ in count:
        dictionary[word]=len(dictionary)
    data=list()
    unk_count=0
    for word in words:
        if word in dictionary:
            index=dictionary[word]
        else:
            index=0
            unk_count=0
        data.append(index)
    count[0][1]=unk_count
    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))

    return data,count,dictionary,reverse_dictionary


def generate_batch(data,batch_size,num_skips,skip_window):
    '''
    skip_window: 单词最远可联系的距离
    num_skips: 对每个单词生成样本数量，num_skips<=2*skip_window
    batch_size应是num_skips的整数倍
    span是某个单词创建相关样本时使用的单词个数，包括目标单词本身和其前后的单词，因此span=2*skip_window+1

    batch[i],reverse_dictionary[batch[i]],'->',labels[i,0],reverse_dictionary[labels[i,0]]
    59 used -> 46 first
    59 used -> 156 against
    156 against -> 128 early
    156 against -> 59 used
    128 early -> 742 working
    128 early -> 156 against
    '''
    global data_index
    assert batch_size%num_skips==0
    assert num_skips<=2*skip_window
    batch=np.ndarray((batch_size),dtype=np.int32)
    labels=np.ndarray((batch_size,1),dtype=np.int32)
    span=2*skip_window+1
    buffer=collections.deque(maxlen=span)

    for _ in range(span):  # buffer初始化
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)
    for i in range(batch_size//num_skips):  # 要生成batch_size个训练样本，需要使用batch_size//num_skips个特征单词
        target=skip_window
        target_to_avoid=[skip_window]
        for j in range(num_skips):
            while target in target_to_avoid:
                target=random.randint(0,span-1)
            target_to_avoid.append(target)
            batch[i*num_skips+j]=buffer[skip_window]  # 特征单词
            labels[i*num_skips+j,0]=buffer[target]  # label，语境单词/特征单词上下文
        buffer.append(data[data_index])  # 将buffer后移一位，此时作为特征单词的data[skip_window]也后移一位，变为新的单词
        data_index=(data_index+1)%len(data)

    return batch,labels


def train():
    batch_size=128
    embedding_size=128
    skip_window=1
    num_skips=2

    valid_size=16
    valid_window=100
    valid_examples=np.random.choice(valid_window,size=valid_size,replace=True)
    num_sampled=64

    data=read_data(maybe_download('text8.zip',31344016))
    data, count, dictionary, reverse_dictionary=build_dataset(data)

    graph=tf.Graph()
    with graph.as_default():
        train_inputs=tf.placeholder(tf.int32,shape=[batch_size])
        train_labels=tf.placeholder(tf.int32,shape=[batch_size,1])
        valid_dataset=tf.constant(valid_examples,dtype=tf.int32)

        with tf.device('/cpu:0'):
            embeddings=tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.,1.))
            embed=tf.nn.embedding_lookup(embeddings,train_inputs)
            nce_weights=tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],stddev=1./math.sqrt(embedding_size)))
            nce_biases=tf.Variable(tf.zeros([vocabulary_size]))

        loss=tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                           biases=nce_biases,
                                           labels=train_labels,
                                           inputs=embed,
                                           num_sampled=num_sampled,
                                           num_classes=vocabulary_size))
        optimizer=tf.train.GradientDescentOptimizer(1.).minimize(loss)
        norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings),axis=1,keepdims=True))
        normalized_embeddings=embeddings/norm
        valid_embeddings=tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
        similarity=tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)

        init=tf.global_variables_initializer()
        num_steps=10001
        with tf.Session() as sess:
            init.run()
            print('initialized...')

            average_loss=0.
            for step in range(num_steps):
                batch_inputs,batch_labels=generate_batch(data,batch_size,num_skips,skip_window)
                _,loss_val=sess.run([optimizer,loss],
                                    feed_dict={train_inputs:batch_inputs,train_labels:batch_labels})
                average_loss+=loss_val

                if step%2000==0:
                    if step>0:
                        average_loss/=2000
                        print(f'step-{step} average loss: {average_loss}')
                    average_loss=0

                if step%5000==0:
                    sim=similarity.eval()
                    for i in range(valid_size):
                        valid_word=reverse_dictionary[valid_examples[i]]
                        top_k=8
                        nearest=(-sim[i,:]).argsort()[1:top_k+1]  # sim越大越相似，argsort()从小到大排序
                        log_str=f'Nearest to {valid_word}'
                        for k in range(top_k):
                            close_word=reverse_dictionary[nearest[k]]
                            log_str+=f', {close_word}'
                        print(log_str)

            final_embeddings=normalized_embeddings.eval()

            return final_embeddings,reverse_dictionary


def plot_with_labels(low_dim_embs,labels,filename='../data/tsne.png'):
    # low_dim_embs是已经降维到二维的向量
    assert low_dim_embs.shape[0]>=len(labels)
    plt.figure(figsize=(18,18))
    for i,label in enumerate(labels):
        x,y=low_dim_embs[i,:]
        plt.scatter(x,y)
        plt.annotate(label,
                     xy=(x,y),
                     xytext=(5,2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)


def plot_image(final_embeddings,reverse_dictionary):
    plot_only=500
    tsne=TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
    low_dims_embs=tsne.fit_transform(final_embeddings[:plot_only,:])
    labels=[reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dims_embs,labels)


if __name__ == '__main__':
    final_embeddings,reverse_dictionary=train()
    plot_image(final_embeddings,reverse_dictionary)
