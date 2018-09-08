# -*- coding: utf-8 -*-
# @Time    : 2018/8/29 11:40
# @Author  : MengnanChen
# @FileName: utils.py
# @Software: PyCharm Community Edition

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def load_data(data_dir):
    lines=[]
    with open(data_dir,'rb') as inputs:
        for line in inputs:
            line=line.decode()
            line=line.split(',')
            lines.append(line)
    return np.array(lines)

def create_tokenizer(lines):
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    return max(len(line.split()) for line in lines)

def encode_sequences(tokenizer,length,lines):
    # encoded sequences(integer list)
    X=tokenizer.texts_to_sequences(lines)
    # pad sequences with 0
    X=pad_sequences(X,maxlen=length,padding='post')
    return X

def encode_output(sequences,vocab_size):
    y_list=[]
    for sequence in sequences:
        # one-hot, [1,2,0,0] -> [[0,1,0],[0,0,1],[1,0,0],[1,0,0]]
        encoded=to_categorical(sequence,num_classes=vocab_size)
        y_list.append(encoded)
    y=np.array(y_list)
    y=y.reshape(sequences.shape[0],sequences.shape[1],vocab_size)
    return y

def id_to_word(integer,tokenizer):
    for word,index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None