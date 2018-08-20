#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/18 22:44
# @Author  : MengnanChen
# @Site    : 
# @File    : symbols.py
# @Software: PyCharm Community Edition

import os
import sys
sys.path.append(os.path.join(os.getcwd(),'\\tacotron\\text'))

import cmudict
_pad='_'
_eos='~'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '

# 为ARPAbet音标(https://en.wikipedia.org/wiki/ARPABET)添加"@"标识符
_arpabet= ['@' + s for s in cmudict.valid_symbols]

# 导出所有符号，充当字典，非英语可以转化为ASCII再使用该字典
symbols=[_pad,_eos]+list(_characters)+_arpabet