'''
简单的线性回归
'''
import paddle.fluid as fluid
import numpy as np
import matplotlib.pyplot as plt


# 数据准备
# 也就五个样本
train_x = np.array([[0.5],[0.6],[0.8],[1.1],[1.4]],dtype='float32')
train_y = np.array([[5.0],[5.5],[6.0],[6.8],[7.1]],dtype='float32')

# 占位符
x = fluid.layers.data(name='xxx',shape=(1,),dtype='float32')
y = fluid.layers.data(name='yyy',shape=(1,),dtype='float32')

# 划分训练集和测试集
# 此处就五个样本  就不划分了


# 搭建模型
# num_flatten_dims=1 表示维度是1
# size 表示 全连接层的 输出维度
pred_y = fluid.layers.fc(input=x,size=1)

# 损失函数
cost = fluid.layers.square_error_cost(pred_y,y)
avg_cost = fluid.layers.mean(cost)

# 梯度下降优化器
# SGD 随机梯度下降
# .minimize(avg_cost) 表示针对均方损失函数进行求导
fluid.optimizer.SGD(learning_rate=0.01).minimize(avg_cost)


# 执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place=place)
exe.run(fluid.default_startup_program())   # 初始化

# 开始训练
# 训练迭代500次
# 模型更新500次参数

iters,costs = [],[]

for i in range(500):
    train_cost = exe.run(program=fluid.default_main_program(),
                         feed={'xxx':train_x,'yyy':train_y},
                         fetch_list=[avg_cost])

    print(f'轮数:{i},cost:{train_cost[0][0]}')
    iters.append(i)
    costs.append(train_cost[0][0])

plt.plot(iters,costs,color='orange')
plt.show()

# shape=(1,) 表示 单个样本的维度是 1（即一维特征，如 [0.5]）。
# 实际训练时，PaddlePaddle 会自动将输入数据按 batch 维度堆叠，
# 因此实际输入的 x 是二维的（如 [[0.5], [0.6], ...]）。

# train_x 的 shape=(5, 1) 是一个二维数组，符合 batch_size=5 的输入要求。
# PaddlePaddle 会自动将 train_x 的每一行（如 [0.5]）作为一个样本，传递给占位符 x。
# fluid.layers.fc 默认会处理二维输入（[batch_size, feature_dim]），因此无需手动展平。


# print(train_cost)  # 输出: [array([0.123], dtype=float32)]

# print(train_cost[0])  # 输出: array([0.123], dtype=float32)
# train_cost[0]即avg_cost, 为 numpy.ndarray，形状为 (1,)（即 [0.123]）。
# fluid.layers.mean(cost) 的返回值是一个 Tensor（张量）
# 标量形式，但 Paddle 中所有 Tensor 至少是 1 维的，因此形状为 [1]（而非纯 Python 标量）。
# 数据：存储均值计算结果，例如 2.5，但以 [2.5] 的形式存储。

# print(train_cost[0][0])  # 输出: 0.123



# PaddlePaddle的exe.run为了支持多任务返回（如同时取loss和accuracy），始终返回List。
# 即使只有一个返回值，也会包装成List，而avg_cost的计算结果默认会包装成shape = (1,)的数组。

# fc 的层数	默认单层，通过多次调用 fc 堆叠多层；size 控制每层的输出维度。




