# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name：     clear_data
   Description :
   Author :       simplefly
   date：          2017/12/23
-------------------------------------------------
   Change Activity:
                   2017/12/23:
-------------------------------------------------
"""
__author__ = 'simplefly'

file = 'utils/iris_data.cvs'
import pandas as pd

def obtain_datas():
    df = pd.read_csv(file, header=None)
    return df

#print(df.head(10))