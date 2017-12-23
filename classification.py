# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name：     classification
   Description :
   Author :       simplefly
   date：          2017/12/23
-------------------------------------------------
   Change Activity:
                   2017/12/23:
-------------------------------------------------
"""
__author__ = 'simplefly'

from matplotlib.colors import ListedColormap
import numpy as np
def plot_decision_regions(X, y, classifier, resolution=0.02):
    marker = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max()
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max()

    print(x1_min, x1_max)
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

