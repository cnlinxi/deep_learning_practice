#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/18 22:20
# @Author  : MengnanChen
# @Site    : 
# @File    : tacotron.py
# @Software: PyCharm Community Edition

import os
import sys
sys.path.append(os.path.join(os.getcwd(),'\\model'))

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell,MultiRNNCell,OutputProjectionWrapper,ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder
from text.symbols import symbols
from utils.infolog import log
from .modules import encoder_cbhg,post_cbhg,prenet



class Tacotron():
    def __init__(self, hparam):
        self._hparam = hparam

    def initialize(self, inputs, input_lengths, mel_targets=None, linear_targets=None):
        '''
        初始化模型以进行inference
        :param inputs:[N,T_in], 其中N为batch_size, T_in是输入时间序列的步数，张量中的值为字符的id
        :param input_lengths:[N], 其中N为batch_size, 张量中的值为每个输入序列的长度
        :param mel_targets:[N,T_out,M]， 其中N为batch_size,T_out为输出序列的步数,M为num_mels，张量中的值为mel谱的entries ####
        :param linear_targets:[N,T_out,F]，其中N为batch_size,T_out为输出序列的步数,F为num_freq，张量中的值为线性谱的entries####
        :return:
        '''
        with tf.variable_scope('inference') as scope:
            is_training = linear_targets is not None
            batch_size = tf.shape(inputs)[0]
            hp = self._hparam

            embedding_table = tf.get_variable(
                'embedding', [len(symbols), hp.embed_depth], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.5)
            )
            embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)  # [N,T_in,embed_depth=256]

            # encoder
            prenet_outputs = prenet(embedded_inputs, is_training, hp.prenet_depth)  # [N,T_in,prenet_depths[-1]=128]
            encoder_outputs=encoder_cbhg(prenet_outputs,input_lengths,is_training,hp.encoder_depth)  # [N,T_in,encoder_depth=256]

            # attention
            attention_mechanism=LocationSensitiveAttention(hp.attention_depth,encoder_outputs)

            # decoder
            multi_rnn_cell=MultiRNNCell([
                ResidualWrapper(GRUCell(hp.decoder_depth)),
                ResidualWrapper(GRUCell(hp.decoder_depth))
            ],state_is_tuple=True)  # [N,T_id,decoder_depth=256]

            # 投影到r个mel谱上(在每一个RNN步上，投影r个输出)
            decoder_cell=TacotronDecoderWrapper(is_training,attention_mechanism,multi_rnn_cell)

            if is_training:
                helper=TacoTrainHelper(inputs,mel_targets,hp.num_mels,hp.outputs_per_steps)
            else:
                helper=TacoTestHelper(batch_size,hp.num_mels,hp.outputs_per_steps)

            decoder_init_state=decoder_cell.zero_state(batch_size=batch_size,dtype=tf.float32)

            (decoder_outputs,_),final_decoder_state,_=tf.contrib.seq2seq.dynamic_decode(
                BasicDecoder(OutputProjectionWrapper(decoder_cell,hp.num_mels*hp.outputs_per_steps),
                             helper,decoder_init_state),maxmum_iterations=hp.max_iters
            )  # [N,T_out/r,M*r]

            # 将输出reshape到每个entry对应一个输出
            mel_outputs=tf.reshape(decoder_outputs,[batch_size,-1,hp.num_mels])  # [N,T_out,M]

            post_outputs=post_cbhg(mel_outputs,hp.num_mels,is_training,hp.postnet_depth)  # [N,T_out,postnet_depth=256]
            linear_outputs=tf.layers.dense(post_outputs,hp.num_freq)  # [N,T_out,F]

            # 从最终的解码器状态获取对齐信息
            alighments=tf.transpose(final_decoder_state.alighment_history.stack(),[1,2,0])  ####

            self.inputs=inputs
            self.input_lengths=input_lengths
            self.mel_outputs=mel_outputs
            self.linear_outputs=linear_outputs
            self.alighments=alighments
            self.mel_targets=mel_targets
            self.linear_targets=linear_targets
            log('Initialized Tacotron model. Dimensions: ')
            log('embedding: %d'%embedded_inputs.shape[-1])
            log('prenet out: %d'%prenet_outputs.shape[-1])
            log('encoder out: %d'%encoder_outputs.shape[-1])
            log('decoder out (%d frames): %d'%(hp.outputs_per_steps,decoder_outputs.shape[-1]))
            log('decoder out (1 frame): %d'%mel_outputs.shape[-1])
            log('postnet out: %d'%post_outputs.shape[-1])
            log('linear out: %d'%linear_outputs.shape[-1])

    def add_loss(self):
        '''
        为模型设置loss，并且设置self.loss,这在初始化时必然被调用
        :return:
        '''
        with tf.variable_scope('loss') as scope:
            hp=self._hparam
            self.mel_loss=tf.reduce_mean(tf.abs(self.mel_targets-self.mel_outputs))
            l1=tf.abs(self.linear_targets-self.linear_outputs)
            # 频率低于4000Hz的优先损失(prioritize loss for frequencies under 4000 Hz)? ####
            n_priority_freq=int(4000/(hp.sample_rate*0.5)*hp.num_freq)
            self.linear_loss=0.5*tf.reduce_mean(l1)+0.5*tf.reduce_mean(l1[:,:,0:n_priority_freq])
            self.loss=self.mel_loss+self.linear_loss

    def add_optimizer(self,global_step):
        '''
        添加优化器optimizer，设置self.gradients和self.optimize
        :param global_step: scalar(int32). 值表示训练中当前总步数
        :return:
        '''
        with tf.variable_scope('optimizer') as scope:
            hp=self._hparam
            if hp.decay_learning_rate:
                self.learning_rate=_learning_rate_decay(hp.initial_learning_rate,global_step)
            else:
                self.learning_rate=tf.convert_to_tensor(hp.initial_learning_rate)
            optimizer=tf.train.AdamOptimizer(self.learning_rate,hp.adam_beta1,hp.adam_beta2)
            gradients,variables=zip(*optimizer.compute_gradients(self.loss))  #### 为何要加*
            self.gradients=gradients
            clipped_gradients,_=tf.clip_by_global_norm(gradients,1.0)  ####防止梯度消失，clip下？

            # 在UPDATE_OPS上添加依赖，否则batch normalization不会正常工作
            # 参见: https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize=optimizer.apply_gradients(zip(clipped_gradients,variables),
                                                        global_step=global_step)

def _learning_rate_decay(init_lr,global_step):
    warmup_steps=4000.
    step=tf.cast(global_step+1,dtype=tf.float32)
    return init_lr*warmup_steps**0.5*tf.minimum(step*warmup_steps**-1.5,step**-0.5)  #### 挺有意思的learning rate decay 的方案