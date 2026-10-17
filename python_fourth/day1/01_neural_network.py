'''
使用python代码实现神经网络的正向传播
'''
import numpy as np

x = np.array([[1.0,2.0]])
# 权重的初始值一般为随机数（期望值为0的正态分布）
w1 = np.random.normal(0,1,(2,3))
# 偏置的初始值一般为0 or 1
b1 = np.ones(shape=(3,))
#第一层的全连接
pred_y1 = x @ w1 + b1 #(1,3)
pred_y1 = 1 / (1 + np.exp(-pred_y1))

#第二层全连接
w2 = np.random.normal(0,1,(3,2))
b2 = np.ones(shape=(2,))
pred_y2 = pred_y1 @ w2 + b2
pred_y2 = 1 / (1 + np.exp(-pred_y2))

#第三层的全连接
w3 = np.random.normal(0,1,(2,2))
b3 = np.ones(shape=(2,))
pred_y3 = pred_y2 @ w3 + b3

print(pred_y3)
