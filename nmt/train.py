# -*- coding: utf-8 -*-
# @Time    : 2018/8/29 10:01
# @Author  : MengnanChen
# @FileName: train.py
# @Software: PyCharm Community Edition

import os
import sys
sys.path.append('')

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Input,LSTM,Dense
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from six.moves import cPickle as pickle

from nmt.utils import *
from nmt.model import NMT

# def load_data(data_path):
#     data=pd.read_csv(data_path)
#     input_texts=data['source'].values.tolist()
#     target_texts=data['target'].values.tolist()
#     num_samples=len(input_texts)
#
#     input_characters=sorted(list(set(data['source'].unique().sum())))
#     targets_characters=sorted(list(set(data['target'].unique().sum())))
#     # print(targets_characters)
#
#     input_length=max([len(i) for i in input_texts])
#     output_length=max([len(i) for i in target_texts])
#     input_feature_length=len(input_characters)
#     output_feature_length=len(targets_characters)
#
#     encoder_input=np.zeros((num_samples,input_length,input_feature_length))
#     decoder_input=np.zeros((num_samples,output_length,output_feature_length))
#     decoder_output=np.zeros((num_samples,output_length,output_feature_length))
#
#     input_dict={char:index for index,char in enumerate(input_characters)}
#     input_dict_reverse={index:char for index,char in enumerate(input_characters)}
#
#     target_dict={char:index for index,char in enumerate(targets_characters)}
#     target_dict_reverse={index:char for index,char in enumerate(targets_characters)}
#
#     for seq_index,seq in enumerate(input_texts):
#         for char_index,char in enumerate(seq):
#             encoder_input[seq_index,char_index,input_dict[char]]=1
#
#     for seq_index,seq in enumerate(target_texts):
#         for char_index,char in enumerate(seq):
#             decoder_input[seq_index,char_index,target_dict[char]]=1.0
#             if char_index>0:
#                 decoder_output[seq_index,char_index-1,target_dict[char]]=1.0
#
#     return input_texts,target_texts,input_dict,target_dict,target_dict_reverse,output_length,\
#            input_feature_length,output_feature_length,encoder_input,decoder_input,decoder_output

# TODO: to be delete...
# def define_model(src_vocab_size,embedding_size,encoder_inputs,num_units=256):
#     embedding_encoder=tf.get_variable('embedding_encoder',[src_vocab_size,embedding_size])
#     encoder_embed_input=tf.nn.embedding_lookup(embedding_encoder,encoder_inputs)
#     encoder_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
#     encoder_outputs,encoder_state=tf.nn.dynamic_rnn(encoder_cell,encoder_embed_input,
#                                                     time_major=True,dtype=tf.float32)
#     with tf.variable_scope('decoder_scope'):
#         attention_states=tf.transpose(encoder_outputs,[1,0,2])
#         attention_mechanism=tf.contrib.seq2seq.LuongAttention(
#             num_units=num_units,memory=attention_states,
#             memory_sequence_length=None
#         )
#
#     decoder_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
#     attention_cell=tf.contrib.seq2seq.AttentionWrapper(
#         decoder_cell,attention_mechanism,
#         attention_layer_size=num_units
#     )
#
# class Seq2Seq:
#     def __init__(self,n_input,n_output,n_units):
#         self.n_input=n_input
#         self.n_output=n_output
#         self.n_units=n_units
#
#     def create_model(self):
#         # 训练阶段
#         # encoder
#         encoder_input = Input(shape=(None, self.n_input))
#         # encoder输入维度n_input为每个时间步的输入xt的维度，这里是用来one-hot的英文字符数
#         encoder = LSTM(self.n_units, return_state=True)
#         # n_units为LSTM单元中每个门的神经元的个数，return_state设为True时才会返回最后时刻的状态h,c
#         _, encoder_h, encoder_c = encoder(encoder_input)
#         encoder_state = [encoder_h, encoder_c]
#         # 保留下来encoder的末状态作为decoder的初始状态
#
#         # decoder
#         decoder_input = Input(shape=(None, self.n_output))
#         # decoder的输入维度为中文字符数
#         decoder = LSTM(self.n_units, return_sequences=True, return_state=True)
#         # 训练模型时需要decoder的输出序列来与结果对比优化，故return_sequences也要设为True
#         decoder_output, _, _ = decoder(decoder_input, initial_state=encoder_state)
#         # 在训练阶段只需要用到decoder的输出序列，不需要用最终状态h.c
#         decoder_dense = Dense(self.n_output, activation='softmax')
#         decoder_output = decoder_dense(decoder_output)
#         # 输出序列经过全连接层得到结果
#
#         # 生成的训练模型
#         model = Model([encoder_input, decoder_input], decoder_output)
#         # 第一个参数为训练模型的输入，包含了encoder和decoder的输入，第二个参数为模型的输出，包含了decoder的输出
#
#         # 推理阶段，用于预测过程
#         # 推断模型—encoder
#         encoder_infer = Model(encoder_input, encoder_state)
#
#         # 推断模型-decoder
#         decoder_state_input_h = Input(shape=(self.n_units,))
#         decoder_state_input_c = Input(shape=(self.n_units,))
#         decoder_state_input = [decoder_state_input_h, decoder_state_input_c]  # 上个时刻的状态h,c
#
#         decoder_infer_output, decoder_infer_state_h, decoder_infer_state_c = decoder(decoder_input,
#                                                                                      initial_state=decoder_state_input)
#         decoder_infer_state = [decoder_infer_state_h, decoder_infer_state_c]  # 当前时刻得到的状态
#         decoder_infer_output = decoder_dense(decoder_infer_output)  # 当前时刻的输出
#         decoder_infer = Model([decoder_input] + decoder_state_input, [decoder_infer_output] + decoder_infer_state)
#
#         return model, encoder_infer, decoder_infer

def main():
    data_dir = 'data/output.csv'
    dataset = load_data(data_dir)
    chi_tokenizer = create_tokenizer(dataset[:, 0])
    chi_vocab_size = len(chi_tokenizer.word_index) + 1
    chi_length = max_length(dataset[:, 0])
    print('chinese vocabulary size:', chi_vocab_size)
    print('chinese max length:', chi_length)

    pinyin_tokenizer = create_tokenizer(dataset[:, 1])
    pinyin_vocab_size = len(pinyin_tokenizer.word_index) + 1
    pinyin_length = max_length(dataset[:, 1])
    print('pinyin vocabulary size:', pinyin_vocab_size)
    print('pinyin max length:', pinyin_length)

    # prepare train & test
    train,test=train_test_split(dataset,random_state=2018,test_size=0.2)

    trainX=encode_sequences(chi_tokenizer,chi_length,train[:,0])
    trainY=encode_sequences(pinyin_tokenizer,pinyin_length,train[:,1])
    trainY=encode_output(trainY,pinyin_vocab_size)

    testX=encode_sequences(chi_tokenizer,chi_length,test[:,0])
    testY=encode_sequences(pinyin_tokenizer,pinyin_length,test[:,1])
    testY=encode_output(testY,pinyin_vocab_size)

    nmt=NMT(chi_vocab_size,pinyin_vocab_size,chi_length,pinyin_length,n_units=256)
    model=nmt.create_model()
    model.compile(optimizer='adam',loss='categorical_crossentropy')
    # summary for model
    if not os.path.exists('model_file'):
        os.mkdir('model_file')

    with open('model_file/tokenizer.pik','wb') as outputs:
        pickle.dump([chi_tokenizer,pinyin_tokenizer],outputs)

    print(model.summary())
    # plot_model(model,to_file='model_file/seq2seq1.png',show_shapes=True)
    model_file_name='model_file/seq2seq1.h5'
    checkpoint=ModelCheckpoint(filepath=model_file_name,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
    model.fit(trainX,trainY,epochs=30,batch_size=16,validation_data=(testX,testY),callbacks=[checkpoint],verbose=2)


if __name__ == '__main__':
    main()

    # file_path='data/output.csv'
    # n_units=256
    # batch_size=32
    # epoch=30
    #
    # # 加载数据
    # input_texts, target_texts, input_dict, target_dict, target_dict_reverse, \
    # output_length, input_feature_length, output_feature_length, \
    # encoder_input, decoder_input, decoder_output = load_data(file_path)
    #
    # seq2seq = Seq2Seq(input_feature_length, output_feature_length, n_units)
    # model_train, encoder_infer, decoder_infer = seq2seq.create_model()
    # model_train.compile(optimizer='rmsprop',loss='categorical_crossentropy')
    #
    # # 模型训练
    # model_train.fit([encoder_input, decoder_input], decoder_output, batch_size=batch_size, epochs=epoch,
    #                 validation_split=0.2)
    #
    # if not os.path.exists('model'):
    #     os.mkdir('model')
    #
    # model_train.save('model/model_train.h5')
    # encoder_infer.save('model/encoder_infer.h5')
    # decoder_infer.save('model/decoder_infer.h5')