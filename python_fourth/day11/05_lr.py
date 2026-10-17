'''
使用 Moudle类搭建线性回归 Linear_Regression
'''
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 数据准备
x = torch.linspace(-1,1,50).reshape(50,1)
# unsqueeze指插入一个维度, 否则张量为一维
# x = Variable(torch.unsqueeze(torch.linspace(-1, 1, 50), dim=1))
# 与上述方法殊途同归


# y=2x+5
y = x * 2 + 5 + torch.randn(x.size())
print(y)

# 为什么不能使用一维？
# 神经网络输入要求
# torch.nn.Linear(1, 1) 期望的输入形状是 [batch_size, input_features]
# nn.Linear 期望 [batch_size, input_features] 格式
# 一维输入 [50]：被解释为50个特征的一个样本 ❌
# 二维输入 [50, 1]：50个样本，每个样本1个特征 ✅

class LinearRegression(torch.nn.Module):  # 继承torch.nn.Module这个父类
    def __init__(self):
        # 继承父类的构造方法
        super(LinearRegression,self).__init__()
        self.linear = torch.nn.Linear(1,1)  # 知道了w的形状


    def forward(self,x):# 正向传播  此处的x类似于静态图中的占位符  形参 可传训练集/测试集
        pred_y = self.linear(x)
        # 搭两层神经网络
        # pred_y1 = self.linear(pred_y)
        # return pred_y1
        return pred_y


model = LinearRegression()
# 损失函数
loss = torch.nn.MSELoss()

# print(list(model.parameters()))  # 模型参数不需要我们自己设置

# 梯度下降
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.01)

iters,costs = [],[]

for pass_id in range(500):  # 训练轮数
    pred_y = model(x)  # 预测值
    train_loss = loss(pred_y,y)  # 损失函数
    train_loss.backward()  # 反向传播
    optimizer.step()  # 更新模型参数
    optimizer.zero_grad()  # 梯度清零,防止梯度累计  因为在w.grad中都是累加值


    print(f'轮数:{pass_id},loss:{train_loss}')

    # 收集每一轮的损失值进行可视化
    iters.append(pass_id)
    costs.append(train_loss.item())

plt.plot(iters,costs)
plt.show()  # 阻塞函数

print(type(train_loss))  # tensor 张量类型
print(type(train_loss.item()))  # float 浮点型
# Matplotlib的绘图函数需要Python的基本数据类型（int, float），不能处理PyTorch张量。
# .item() 会从计算图中提取数值，断开与梯度计算的连接，减少内存占用。将PyTorch张量转换为Python数值














