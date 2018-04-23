# -*- coding: utf-8 -*-
# __author__ = 'cjn'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()

# 2.生成模拟数据
# x_val为一个二维顶点，y_vals为该顶点的分类结果
(x_vals, y_vals) = datasets.make_circles(n_samples=500, factor=.5, noise=.1)
class1_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == 1] # 获取分类结果为１的x坐标
class1_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == 1] # 获取分类结果为１的y坐标
class2_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == -1]
class2_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == -1]
# 以下代码可以查看生成的图形为两种颜色的圆点，以圆圈的形式进行分布
# fig=plt.figure(1)
# x1,y1=datasets.make_circles(n_samples=500,factor=0.0,noise=.1)
# plt.subplot(121)
# plt.title('make_circles function example')
# plt.scatter(x1[:,0],x1[:,1],marker='o',c=y1)
# plt.show()

# 3.声明批量的大小,占位符,创建模型变量b
batch_size = 250
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)
b = tf.Variable(tf.random_normal(shape=[1, batch_size]))

# 4.创建高斯核函数,该核函数用矩阵操作来表示
gamma = tf.constant(-50.0)
dist = tf.reduce_sum(tf.square(x_data), 1)