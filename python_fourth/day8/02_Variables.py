'''
Paddle中不同类型的变量
模型(一个完整的模型) = 模型结构 + 模型参数
例子： y = k*x+b
      k = 2   b = 1
      这才是一个完整的模型
'''
import paddle.fluid as fluid
import numpy as np

# 定义 fluid.default_main_program
# 常量类型的变量
x = fluid.layers.fill_constant(shape=(1,5),dtype='float32',value=5.0)

# 样本数据的占位符
y = fluid.layers.data(name='xxx',shape=(1,5),dtype='float32')
# 在python中(program程序)叫y，在paddle中(执行器中)叫xxx

# 模型参数的变量   有默认初始值(接口帮你设置好了，不用自己设置)
w = fluid.layers.create_parameter(shape=[5,3],dtype='float32')
b = fluid.layers.create_parameter(shape=[3],dtype='float32',is_bias=True)

# 矩阵相乘
# pred_y = fluid.layers.matmul(x,w) + b
pred_y = fluid.layers.matmul(y,w) + b
pred_y = fluid.layers.sigmoid(pred_y)   # 激活函数

# 执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place=place)
exe.run(fluid.default_startup_program())   # 初始化  执行器得执行初始化一次
# 之后写代码  不管有没有变量需要初始化  你都把这一行代码写一遍  总不会错
# 执行
# result = exe.run(program=fluid.default_main_program(),fetch_list=[w])

y_data = np.arange(1,6).reshape(1,5).astype('float32 ')
result = exe.run(program=fluid.default_main_program(),
                 feed={'xxx':y_data},
                 fetch_list=[pred_y])
print(result[0])

