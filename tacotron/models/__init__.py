# -*- coding: utf-8 -*-
# @Time    : 2018/8/20 15:15
# @Author  : MengnanChen
# @FileName: __init__.py
# @Software: PyCharm Community Edition

from .tacotron import Tacotron

def create_model(name,hparams):
    if name=='tacotron':
        return Tacotron(hparams)
    else:
        raise Exception('Unknown model: ',name)