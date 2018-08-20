# -*- coding: utf-8 -*-
# @Time    : 2018/8/20 15:10
# @Author  : MengnanChen
# @FileName: synthesizer.py
# @Software: PyCharm Community Edition

import tensorflow as tf
from models import create_model
from hparams import hparams
from util import audio
from text import text_to_sequence
import numpy as np
import io

class synthesizer:
    def load(self,checkpoint_path,model_name='tacotron'):
        print('constructing model: %s'%model_name)
        inputs=tf.placeholder(tf.int32,[1,None],'inputs')
        input_lengths=tf.placeholder(tf.int32,[1],'input_lengths')
        with tf.variable_scope('model') as scope:
            self.model=create_model(model_name,hparams)
            self.model.initialize(inputs,input_lengths)
            self.wav_output=audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])

        print('loading checkpoint: %s'%checkpoint_path)
        self.session=tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        saver.restore(self.session,checkpoint_path)

    def synthesize(self,text):
        cleaner_names=[x.strip() for x in hparams.cleaner.split(',')]
        seq=text_to_sequence(text,cleaner_names)
        feed_dict={
            self.model.inputs:[np.asarray(seq,dtype=np.int32)],
            self.model.input_lengths:np.asarray([len(seq)],dtype=np.int32)
        }
        wav=self.session.run(self.wav_output,feed_dict=feed_dict)
        wav=audio.inv_preemphasis(wav)
        wav=wav[:audio.find_endpoint(wav)]
        out=io.BytesIO()
        audio.save_wav(wav,out)
        return out.getvalue()