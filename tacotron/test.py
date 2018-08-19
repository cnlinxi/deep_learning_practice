# -*- coding: utf-8 -*-
# @Time    : 2018/8/19 9:16
# @Author  : MengnanChen
# @FileName: test.py
# @Software: PyCharm Community Edition

class test_class():
    def __init__(self):
        pass

    def set_vallue(self):
        self.this_value=1

    def print_value(self):
        print(self.this_value)

if __name__ == '__main__':
    test=test_class()
    test.print_value()