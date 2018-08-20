#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/18 22:48
# @Author  : MengnanChen
# @Site    : 
# @File    : __init__.py
# @Software: PyCharm Community Edition

import re
from text import cleaners
from text.symbols import symbols

_symbol_to_id={s: i for i,s in enumerate(symbols)}
_id_to_symbol={i:s for i,s in enumerate(symbols)}

_curly_re=re.compile(r'(.*?)\{(.*?)\}(.*)')


def text_to_sequence(text,cleaner_names):
    '''
    将字符序列转化为相对应的id序列
    字符序列中可以使用{}包裹ARPAbet音标序列，比如: Turn left on {HH AW1 S S} Street.
    :param text:
    :param cleaner_names:
    :return: list of int. 字符序列对应的ids
    '''
    sequence=[]
    while len(sequence):
        m=_curly_re.match(text)
        if not m:
            sequence+=_symbol_to_sequence(_clean_text(text,cleaner_names))
            break
        sequence+=_symbol_to_sequence(_clean_text(m.group(1),cleaner_names))
        sequence+=_arpabet_to_sequence(m.group(2))
        text=m.group(3)  # 继续转换text后面部分

    sequence.append(_symbol_to_id['~'])
    return sequence

def _clean_text(text, cleaner_names):
    # 对text进行清理，比如转化为小写、规范化空格等
    # 允许使用多个cleaner共同对text清理，cleaner_names: list
    for name in cleaner_names:
        cleaner=getattr(cleaners,name)
        if not cleaner:
            raise Exception('unknown cleaner: %s'%name)
        text=cleaner(text)
    return text

def _arpabet_to_sequence(text):
    '''
    对ARPAbet音标添加一个@再转化为对应的ids，以区别于字母
    :param text:
    :return:
    '''
    return _symbol_to_sequence(['@'+s for s in text.split()])

def _symbol_to_sequence(symbols):
    '''
    string -> ids
    :param symbols:
    :return:
    '''
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbols(s)]

def _should_keep_symbols(s):
    return s in _symbol_to_id and s is not '_' and s is not '~'