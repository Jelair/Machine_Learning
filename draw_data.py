# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name：     draw_data
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
import matplotlib.pyplot as plt
import numpy as np
import perc
import adal
# pandas中数据截取方式，loc,iloc,ix对比
# 1.loc:通过行索引获取行数据
# 2.iloc:通过行号获取行数据
# 3.ix:混合方式，既可以通过行号，又可以通过行索引获取行数据
df = pd.read_csv(file, header=None)
y = df.loc[0:100, 4].values
y = np.where( y == 'Iris-setosa', -1, 1)
X= df.loc[0:100, [0,2]].values
# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
# plt.xlabel(u'花瓣长度',fontproperties='SimHei')
# plt.ylabel(u'花镜长度',fontproperties='SimHei')
# plt.legend(loc='upper left')
#plt.show()

ppn = perc.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
# plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel(u'错误分类次数',fontproperties='SimHei')
#plt.show()

from matplotlib.colors import ListedColormap
import numpy as np
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max()
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max()


    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

# plot_decision_regions(X, y, ppn)
# plt.xlabel(u'花瓣长度',fontproperties='SimHei')
# plt.ylabel(u'花镜长度',fontproperties='SimHei')
# plt.legend(loc='upper left')
#plt.show()

ada = adal.AdalineGD(eta=0.0001, n_iter=50)
ada.fit(X, y)
plot_decision_regions(X, y, classifier=ada)
plt.title('Adaline-Gradient descent')
plt.xlabel(u'花瓣长度',fontproperties='SimHei')
plt.ylabel(u'花镜长度',fontproperties='SimHei')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('sum-equard-error')
plt.show()