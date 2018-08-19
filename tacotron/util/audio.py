#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/19 22:09
# @Author  : MengnanChen
# @Site    : 
# @File    : audio.py
# @Software: PyCharm Community Edition

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile
from hparams import hparams

def load_wav(path):
    return librosa.core.load(path,sr=hparams.sample_rate)[0]

def save_wav(wav,path):
    wav*=32767/max(0.01,np.max(np.abs(wav)))
    wavfile.write(path,hparams.sample_rate,wav.astype(np.int16))

def preemphasis(x):
    # 使用IIR或FIR过滤一维数据
    return signal.lfilter([1,-hparams.preempahsis],[1],x)

def inv_preemphasis(x):
    return signal.lfilter([1],[1,-hparams.preempahsis],x)

def spectrogram(y):
    D=_stft(preemphasis(y))
    S=_amp_to_db(np.abs(D))-hparams.ref_level_db
    return _normalize(S)

def _stft(y):
    n_fft,hop_length,win_length=_stft_parameters()
    return librosa.stft(y=y,n_fft=n_fft,hop_length=hop_length,win_length=win_length)

def _stft_parameters():
    n_fft=(hparams.num_freq-1)*2
    hop_length=

