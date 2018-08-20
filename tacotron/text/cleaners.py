# -*- coding: utf-8 -*-
# @Time    : 2018/8/20 15:34
# @Author  : MengnanChen
# @FileName: cleaners.py
# @Software: PyCharm Community Edition

import re
from unidecode import unidecode
from numbers import normalize_numbers

_whitespace_re=re.compile(r'\s+')

_abbreviations=[(re.compile('\\b%s\\.'%x[0],re.IGNORECASE),x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]

def expand_abbreviations(text):
    for regex,replacement in _abbreviations:
        text=re.sub(regex,replacement,text)
    return text

def expand_numbers(text):
    return normalize_numbers(text)

def lowercase(text):
    return text.lower()

def convert_to_acsii(text):
    return unidecode(text)

def collapse_whitespace(text):
    return re.sub(_whitespace_re,' ',text)

def basic_cleaner(text):
    text=lowercase(text)
    text=collapse_whitespace(text)
    return text

def transliteration_cleaner(text):
    '''
    pipline for non-English text that transliterates to ASCII
    :param text:
    :return:
    '''
    text=convert_to_acsii(text)
    text=lowercase(text)
    text=collapse_whitespace(text)
    return text