#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/19 22:10
# @Author  : MengnanChen
# @Site    : 
# @File    : hparams.py
# @Software: PyCharm Community Edition

import tensorflow as tf

hparams=tf.contrib.training.HParams(
    cleaners='basic_cleaners',

    # Audio
    num_mels=80,
    num_freq=1025,
    sample_rate=22050,
    frame_length_ms=50,
    frame_shift_ms=12.5,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    max_frame_num=1000,

    # Model
    outputs_per_step=5,
    embed_depth=256,
    premet_depths=[256,128],
    encoder_depth=256,
    postnet_depth=256,
    attention_depth=256,
    decoder_depth=256,

    # Training
    batch_size=32,
    adam_beta1=0.9,
    adam_beta2=0.999,
    initial_learning_rate=0.001,
    decay_learning_rate=True,
    use_cmudict=False,

    # Eval
    max_iters=300,
    griffin_lim_iters=60,
    power=1.5,  # power to raise magnitudes to prior to Griffin-Lim
)

def hparams_debug_string():
    values=hparams.values()
    hp=[' %s: %s'%(name,values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n'+'\n'.join(hp)