# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 9:34
# @Author  : MengnanChen
# @FileName: infolog.py
# @Software: PyCharm Community Edition

from datetime import datetime
from threading import Thread

_format='%Y-%m-%d %H:%M:%S.%f'
_file=None
_slack_url=None

def log(msg,end='\n',slack=False):
    print(msg,end=end)
    if _file is not None:
        _file.write('[%s]   %s\n'%(datetime.now().strftime(_format)[:-3],msg))
    if slack and _slack_url is not None:
        pass