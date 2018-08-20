# -*- coding: utf-8 -*-
# @Time    : 2018/8/20 14:21
# @Author  : MengnanChen
# @FileName: preprocess.py
# @Software: PyCharm Community Edition

import os

from tqdm import tqdm
from datasets import thchs30
from hparams import hparams


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir,'train.txt'),'w',encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m])+'\n')

    frames=sum([m[2] for m in metadata])
    hours=frames*hparams.frame_shift_ms/(3600*1000)
    print('write %d utterances, %d frames, (%.2f hours)'%(len(metadata),frames,hours))
    print('max input length: %d'%max(len(m[3]) for m in metadata))
    print('max output length: %d'%max(m[2] for m in metadata))


def preprocess_thchs30(args):
    in_dir=os.path.join(args.base_dir,'data_thchs30')
    out_dir=os.path.join(args.base_dir,args.output)
    os.makedirs(out_dir,exist_ok=True)
    metadata=thchs30.build_from_path(in_dir,out_dir,args.num_workers,tqdm=tqdm)
    write_metadata(metadata,out_dir)
