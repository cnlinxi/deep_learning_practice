# -*- coding: utf-8 -*-
# @Time    : 2018/8/29 9:23
# @Author  : MengnanChen
# @FileName: transform_pinyin.py
# @Software: PyCharm Community Edition

from pypinyin import lazy_pinyin,Style

def is_chinses(uchar):
    if uchar>u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False

def clean_line(line):
    cleaned_line=[]
    for char in line:
        if is_chinses(char):
            cleaned_line.append(char)
    return cleaned_line

def get_pinyin(line):
    return lazy_pinyin(line,style=Style.TONE2,errors='ignore')

def transform_pinyin_to_file(data_path,output_path):
    with open(data_path,'rb') as inputs,open(output_path,'wb') as outputs:
        outputs.write('source,target\n'.encode('utf-8'))
        for line in inputs:
            line=line.decode('utf-8')
            line=clean_line(line)
            if not line:
                continue
            transformed_line=get_pinyin(line)
            line=' '.join(line)
            transformed_line=' '.join(transformed_line)
            outputs.write(f'{line},{transformed_line}\n'.encode('utf-8'))

if __name__ == '__main__':
    data_dir='data/input.txt'
    output_dir='data/output.csv'
    transform_pinyin_to_file(data_dir,output_dir)