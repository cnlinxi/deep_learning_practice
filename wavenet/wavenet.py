# -*- coding: utf-8 -*-
# @Time    : 2018/9/21 11:29
# @Author  : MengnanChen
# @FileName: wavenet.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
# from wavenet import masked

DEFAULT_LR_SCHEDULE={
    0: 2e-4,
    90000: 4e-4 / 3,
    120000: 6e-5,
    150000: 4e-5,
    180000: 2e-5,
    210000: 6e-6,
    240000: 2e-6
}

DETAIL_LOG=True

# class WNHelper():
#     @staticmethod
#     def upsample_conv1d(x,num_filters,filter_length,stride,
#                         use_resize_conv,name_patt,act='tanh',
#                         use_weight_norm=False,init=False):
        # act_func=