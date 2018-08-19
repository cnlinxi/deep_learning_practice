# -*- coding: utf-8 -*-
# @Time    : 2018/8/19 21:39
# @Author  : MengnanChen
# @FileName: thchs30.py
# @Software: PyCharm Community Edition

import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import glob


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
            task=partial(_process_utterance)

def _process_utterance(out_dir,index,wav_path,pinyin):
    pass