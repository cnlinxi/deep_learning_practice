# -*- coding: utf-8 -*-
# @Time    : 2018/11/21 14:12
# @Author  : MengnanChen
# @FileName: __init__.py
# @Software: PyCharm

from .mlp import MLP


def create_model(mode='train'):
    return MLP(mode=mode)
