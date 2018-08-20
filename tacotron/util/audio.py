#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/19 22:09
# @Author  : MengnanChen
# @Site    : 
# @File    : audio.py
# @Software: PyCharm Community Edition

import librosa
import tensorflow as tf
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
    return _normalize(S)  # normalize是干嘛的？####

def inv_spectrogram(spectrogram):
    '''
    使用librosa将频谱转化为波形，Vocoder
    :param spectrogram:
    :return:
    '''
    S=_db_to_amp(_denormalize(spectrogram)+hparams.ref_level_db)  # 转回linear
    return inv_preemphasis(_griffin_lim(S**hparams.power))


def inv_spectrogram_tensorflow(spectrogram):
    S=_db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram)+hparams.ref_level_db)
    return _griffin_lim_tensorflow(tf.pow(S,hparams.power))

def melspectrogram(y):
    D=_stft(preemphasis(y))
    S=_amp_to_db(_linear_to_mel(np.abs(D)))-hparams.ref_level_db
    return _normalize(S)

def find_endpoint(wav,threshold_db=-40,min_slience_sec=0.8):
    '''
    终止点，以静音(低于threshold_db之下的响度级)作为终止点
    :param wav:
    :param threshold_db:
    :param min_slience_sec: 静音时间应该大于min_slience_sec，防止发生误判
    :return:
    '''
    window_length=int(hparams.sample_rate*min_slience_sec)
    hop_length=int(window_length/4)
    threshold=_db_to_amp(threshold_db)
    for x in range(hop_length,len(wav)-window_length,hop_length):
        if np.max(wav[x:x+window_length])<threshold:
            return x+hop_length
    return len(wav)

def _stft(y):
    n_fft,hop_length,win_length=_stft_parameters()
    return librosa.stft(y=y,n_fft=n_fft,hop_length=hop_length,win_length=win_length)

def _stft_tensorflow(signals):
    n_fft,hop_length,win_length=_stft_parameters()
    return tf.contrib.signal.stft(signals,win_length,hop_length,n_fft,pad_end=False)

def _stft_parameters():
    n_fft=(hparams.num_freq-1)*2  # 为什么这么计算？n_fft是什么？傅里叶变换？####
    hop_length=int(hparams.frame_shift_ms/1000*hparams.sample_rate)
    win_length=int(hparams.frame_length_ms/1000*hparams.sample_rate)
    return n_fft,hop_length,win_length

# Conversions
_mel_basis=None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis=build_mel_basis()  # 此处似乎是建立一个mel谱基础
    return np.dot(_mel_basis,spectrogram)  # 与传入的linear谱做矩阵乘即可？####

def build_mel_basis():
    n_fft=(hparams.num_freq-1)*2
    return librosa.filters.mel(hparams.sample_rate,n_fft,n_mels=hparams.num_mels)

def _amp_to_db(x):
    return 20*np.log10(np.maximum(1e-5,x))

def _normalize(S):
    return np.clip((S-hparams.min_level_db)/-hparams.min_level_db,0,1)

def _db_to_amp(x):
    return np.power(10,x*0.05)

def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x))*10.0,x*0.05)

def _denormalize(S):
    return (np.clip(S,0,1)*-hparams.min_level_db)+hparams.min_level_db

def _denormalize_tensorflow(S):
    return (tf.clip_by_value(S,0,1)*-hparams.min_level_db)+hparams.min_level_db


def _istft(y):
    _,hop_length,win_length=_stft_parameters()
    return librosa.istft(y,hop_length,win_length)

def _istft_tensorflow(stfts):
    n_fft,hop_length,win_length=_stft_parameters()
    return tf.contrib.signal.inverse_stft(n_fft,hop_length,win_length)


def _griffin_lim(S):
    '''
    使用librosa实现的Griffin-Lim
    :param S:
    :return:
    '''
    angles=np.exp(2j*np.pi*np.random.rand(*S.shape))  # S.shape之前加*是干嘛的####
    S_complex=np.abs(S).astype(np.complex)
    y=_istft(S_complex*angles)
    for i in range(hparams.griffin_lim_iters):
        angels=np.exp(1j*np.angle(_stft(y)))
        y=_istft(S_complex*angels)
    return y


def _griffin_lim_tensorflow(S):
    with tf.variable_scope('griffinlim'):
        S=tf.expand_dims(S,0)
        S_complex=tf.identity(tf.cast(S,dtype=tf.complex64))
        y=_istft_tensorflow(S_complex)
        for i in range(hparams.griffin_lim_iters):
            est=_stft_tensorflow(y)
            angles=est/tf.cast(tf.maximum(1e-8,tf.abs(est)),tf.complex64)
            y=_istft_tensorflow(S_complex*angles)
        return tf.squeeze(y,0)