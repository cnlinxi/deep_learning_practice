# -*- coding: utf-8 -*-
# @Time    : 2018/9/1 10:35
# @Author  : MengnanChen
# @FileName: helpers.py
# @Software: PyCharm Community Edition

import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper

class TacoTrainingHelper(Helper):
    def __init__(self,batch_size,targets,stop_targets,hparams,gta,
                 is_evaluating,global_step):
        with tf.name_scope('TacoTrainingHelper'):
            self._batch_size=batch_size
            self._hparams=hparams
            self._output_dim=hparams.num_dims
            self._reduction_factor=hparams.outputs_per_step
            self._ratio=tf.convert_to_tensor(hparams.tacotron_teacher_forcing_ratio)
            self._targets=targets
            self._stop_targets=stop_targets
            self.gta=gta
            self.eval=is_evaluating
            self.global_step=global_step

            r=self._reduction_factor
            # for one sample, feed every r-th target frame as input
            self._targets=targets[:,r-1::r,:]

            # maximal sequence length
            # tf.shape(self._targets)[1]: the number of frames in one sample
            self._lengths=tf.tile([tf.shape(self._targets)[1]],[self._batch_size])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def token_output_size(self):
        return self._reduction_factor

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        if self.gta:
            self._ratio=tf.convert_to_tensor(1.)
        elif self.eval and self._hparams.natural_eval:
            self._ratio=tf.convert_to_tensor(0.)
        else:
            if self._hparams.tacotron_teacher_forcing_mode=='scheduled':
                self._ratio=_teacher_forcing_ration_decay(self._hparams.tacotron_teacher_forcing_init_ratio,
                                                          self.global_step,
                                                          self._hparams)
                return (tf.tile([False],[self._batch_size]),_go_frames(self._batch_size,self._output_dim))

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0],[self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, stop_token_prediction, name=None):
        with tf.name_scope(name or 'TacoTrainingHelper'):
            # synthesis stop
            finished=(time+1>=self._lengths)
            # next_inputs will be one of below input:
            # self._target[:,time,:] -> one frame of ground truth
            # output[:,-self._output_dim:] -> one predicted frame
            next_inputs=tf.cond(
                tf.less(tf.random_uniform([],minval=0,maxval=1,dtype=tf.float32),self._ratio),
                lambda : self._targets[:,time,:],  # teacher-forcing: return true frame
                lambda :outputs[:,-self._output_dim:]
            )

            next_state=state
            return (finished,next_inputs,next_state)

def _go_frames(batch_size,output_dim):
    # return all-zero <GO> frames for a given batch_size and output dimension
    return tf.tile([[0.0]],[batch_size,output_dim])


def _teacher_forcing_ration_decay(init_tfr,global_step,hparams):
    # narrow cosine decay:
    # phase 1 -> tfr=1: start learning_rate_decay after 10k step
    # phase 2 -> tfr in (0,1): decay reach minimal value at step~280k
    # phase 3 -> tfr=0: clip by minimal value at step>~280k

    # compute natural cosine decay
    # tfr=1 at step 10k
    # tfr=0% of init_tfr as final value
    tfr=tf.train.cosine_decay(init_tfr,global_step=global_step-hparams.tacotron_teacher_forcing_start_decay,
                              decay_steps=hparams.tacotron_teacher_forcing_decay_steps,
                              alpha=hparams.tacotron_teacher_forcing_decay_alpha,
                              name='tfr_cosine_decay')
    # force teacher_focing_ratio to take initial value when global_step < start_decay_step
    narrow_tfr=tf.cond(tf.less(global_step,tf.convert_to_tensor(hparams.tacotron_teacher_forcing_start_decay)),
                       lambda :tf.convert_to_tensor(init_tfr),
                       lambda :tfr)

    return narrow_tfr