# -*- coding: utf-8 -*-
# @Time    : 2018/8/29 10:55
# @Author  : MengnanChen
# @FileName: predict.py
# @Software: PyCharm Community Edition

import os

import numpy as np
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from six.moves import cPickle as pickle

from nmt.train import load_data
from .utils import *

# def predict_pinyin(source,encoder_inference, decoder_inference, n_steps, features):
#     #先通过推理encoder获得预测输入序列的隐状态
#     state = encoder_inference.predict(source)
#     #第一个字符'\t',为起始标志
#     predict_seq = np.zeros((1,1,features))
#     predict_seq[0,0,target_dict['\t']] = 1
#
#     output = ''
#     #开始对encoder获得的隐状态进行推理
#     #每次循环用上次预测的字符作为输入来预测下一次的字符，直到预测出了终止符
#     for i in range(n_steps):#n_steps为句子最大长度
#         #给decoder输入上一个时刻的h,c隐状态，以及上一次的预测字符predict_seq
#         yhat,h,c = decoder_inference.predict([predict_seq]+state)
#         #注意，这里的yhat为Dense之后输出的结果，因此与h不同
#         char_index = np.argmax(yhat[0,-1,:])
#         char = target_dict_reverse[char_index]
#         output += char
#         state = [h,c]#本次状态做为下一次的初始状态继续传递
#         predict_seq = np.zeros((1,1,features))
#         predict_seq[0,0,char_index] = 1
#         if char == '\n':#预测到了终止符则停下来
#             break
#     return output

def predict_sequence(model,tokenizer,source):
    prediction=model.predict(source,verbose=0)[0]
    integers=[np.argmax(vector) for vector in prediction]
    target=[]
    for i in integers:
        word=id_to_word(i,tokenizer)
        if word is None:
            word='none'
        target.append(word)
    return ' '.join(target)

def evaluate_model(model,tokenizer,sources,raw_dataset):
    actual,predicted=[],[]
    for i,line in enumerate(raw_dataset):
        assert len(line)==2
        source=sources[i]
        source=source.reshape((1,source.shape[0]))
        translation=predict_sequence(model,tokenizer,source)
        raw_src,raw_target=line
        if i<10:
            print(f'src={raw_src}, target={raw_target}, predicted={translation}')

    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


def main():
    evaluation_dataset_dir='data/evaluation.csv'
    dataset=load_data(evaluation_dataset_dir)

    # chi_tokenizer=create_tokenizer(dataset[:,0])
    # chi_vocab_size=len(chi_tokenizer.word_index)+1
    # chi_length=max_length(dataset[:,0])
    # print(f'chinses vocabulary size(evaluation):{chi_vocab_size}')
    # print(f'chinses max length:{chi_length}')
    #
    # pinyin_tokenizer=create_tokenizer(dataset[:,1])
    # pinyin_vocab_size=len(pinyin_tokenizer.word_index)+1
    # pinyin_length=max_length(dataset[:,1])
    # print(f'pinyin vocabulary size(evaluation):{pinyin_vocab_size}')
    # print(f'pinyin max length:{pinyin_length}')

    with open('model/tokenizer.pik','rb') as inputs:
        chi_tokenizer,pinyin_tokenizer=pickle.load(inputs)
    chi_length=max_length(dataset[:,0])

    sources=encode_sequences(chi_tokenizer,chi_length,dataset[:,0])

    model=load_model('model_file/model.h5')
    evaluate_model(model,pinyin_tokenizer,sources,dataset)

if __name__ == '__main__':
    main()

    # seq='你好'
    # file_path='data/output.csv'
    #
    # input_texts, target_texts, input_dict, target_dict, target_dict_reverse, \
    # output_length, input_feature_length, output_feature_length, \
    # encoder_input, decoder_input, decoder_output = load_data(file_path)
    #
    # if not (os.path.exists('model/encoder_infer.h5') and os.path.exists('model/decoder_infer.h5')):
    #     raise FileExistsError('no model exists')
    # encoder_infer=load_model('model/encoder_infer.h5')
    # decoder_infer=load_model('model/decoder_infer.h5')
    #
    # input_length = max([len(i) for i in input_texts])
    # encoder_input = np.zeros((1, input_length, input_feature_length))
    # for char_index, char in enumerate(seq):
    #     encoder_input[0, char_index, input_dict[char]] = 1
    # out = predict_pinyin(encoder_input, encoder_infer, decoder_infer, output_length, output_feature_length)
    # print(f'src:{seq}, pinyin:{out}')
