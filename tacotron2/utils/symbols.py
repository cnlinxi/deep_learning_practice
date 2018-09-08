# -*- coding: utf-8 -*-
# @Time    : 2018/8/28 10:41
# @Author  : MengnanChen
# @FileName: symbols.py
# @Software: PyCharm Community Edition

from . import cmudict

_pad='_'
_eos='~'
_characters='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '

_arpabet=['@' + s for s in cmudict.valid_symbols]

symbols=[_pad,_eos]+list(_characters)+_arpabet