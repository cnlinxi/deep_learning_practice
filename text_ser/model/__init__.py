# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 16:27
# @Author  : MengnanChen
# @FileName: __init__.py
# @Software: PyCharm

from .model import TextTransformer


def create_model(name, mode):
    if name == 'transformer':
        return TextTransformer(mode=mode)
    else:
        raise Exception('unknown model')
