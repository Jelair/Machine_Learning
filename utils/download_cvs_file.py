# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name：     download_cvs_file
   Description :
   Author :       simplefly
   date：          2017/12/23
-------------------------------------------------
   Change Activity:
                   2017/12/23:
-------------------------------------------------
"""
__author__ = 'simplefly'

import requests

file_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

response = requests.get(file_url)
with open('iris_data.cvs', 'wb') as f:
    f.write(response.content)
print('finish')