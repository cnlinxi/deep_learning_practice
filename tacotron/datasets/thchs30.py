# -*- coding: utf-8 -*-
# @Time    : 2018/8/19 21:39
# @Author  : MengnanChen
# @FileName: thchs30.py
# @Software: PyCharm Community Edition

import os

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import glob
import numpy as np

from hparams import hparams as hp
from util import audio


def build_from_path(in_dir,out_dir,num_workers=1,tqdm=lambda x:x):
    executor=ProcessPoolExecutor(max_workers=num_workers)
    futures=[]
    index=1

    trn_files=glob.glob(os.path.join(in_dir,'data','*.trn'))

    for trn in trn_files:
        with open(trn) as f:
            f.readline()
            pinyin=f.readline().strip('\n')
            wav_file=trn[:-4]
            task=partial(_process_utterance,out_dir,index,wav_file,pinyin)
            futures.append(executor.submit(task))
            index+=1
    return [future.result() for future in tqdm(futures) if future.result is not None]

def _process_utterance(out_dir,index,wav_path,pinyin):
    wav=audio.load_wav(wav_path)

    spectrogram=audio.spectrogram(wav).astype(np.float32)
    n_frame=spectrogram.shape[1]
    if n_frame>hp.max_frame_num:
        return None

    mel_spectrogram=audio.melspectrogram(wav).astype(np.float32)

    spectrogram_filename='thchs30-spec-%05d.npy'%index
    mel_filename='thchs30-mel-%05d.npy'%index
    np.save(os.path.join(out_dir,spectrogram_filename),spectrogram.T,allow_pickle=False)
    np.save(os.path.join(out_dir,mel_filename),mel_spectrogram.T,allow_pickle=False)

    return (spectrogram_filename,mel_filename,n_frame,pinyin)