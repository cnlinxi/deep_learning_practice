# -*- coding: utf-8 -*-
# @Time    : 2018/10/22 15:11
# @Author  : MengnanChen
# @FileName: data_loader.py
# @Software: PyCharm


import os
import pickle
import random

import nltk


pad_token,go_token,eos_token,unknown_token=0,1,2,3


class Batch:
    def __init__(self):
        self.encoder_inputs=[]
        self.encoder_inputs_length=[]
        self.decoder_targets=[]
        self.decoder_targets_length=[]


def load_dataset(file_path):
    print('load data from {}'.format(file_path))
    with open(file_path,'rb') as fin:
        data=pickle.load(file_path)
        word2id=data['word2id']
        id2word=data['id2word']
        train_samples=data['trainingSamples']

    print('word2id length: {}'.format(len(word2id)))
    print('id2word length: {}'.format(len(id2word)))
    print('training samples length: {}'.format(len(train_samples)))

    return word2id,id2word,train_samples


def create_batch(samples):
    batch=Batch()
    batch.encoder_inputs_length=[len(sample[0]) for sample in samples]
    batch.decoder_inputs_length=[len(sample[1]) for sample in samples]

    max_source_length=max(batch.encoder_inputs_length)
    max_target_length=max(batch.decoder_inputs_length)

    for sample in samples:
        source=sample[0]
        pad=[pad_token]*(max_source_length-len(sample))
        batch.encoder_inputs.append(source+pad)

        target=sample[1]
        pad=[pad_token]*(max_target_length-len(target))
        batch.decoder_targets.append(target+pad)

    return batch


def get_batchs(data,batch_size):
    random.shuffle(data)
    batches=[]
    data_len=len(data)

    def generate_next_samples():
        for i in range(0,data_len,batch_size):
            yield data[i:min(i+batch_size,data_len)]

    for samples in generate_next_samples():
        batch=create_batch(samples)
        batches.append(batch)

    return batches

def sentence2encoder(sentence,word2id):
    if sentence=='':
        return None
    tokens=nltk.word_tokenize(sentence)
    if len(tokens)>20:
        return None
    word_ids=[]
    for token in tokens:
        word_ids.append(word2id.get(token,unknown_token))

    batch=create_batch([[word_ids,[]]])
    return batch
