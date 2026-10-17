'''
张良的基本构建
'''
import torch
import numpy as np


# 创建一个张量
tensor = torch.tensor(5)
print(tensor)

tensor = torch.Tensor(1)  # (1,)
print(tensor)

# 将数组转成tensor
ary = np.arange(1,10).reshape(3,3)
tensor = torch.tensor(ary)
print(tensor)

# 生成指定形状的张量   随机生成数值
tensor = torch.Tensor(5)  # (5,)
print(tensor)

tensor = torch.Tensor(1,5)  # (1,5)
print(tensor)

tensor = torch.Tensor([1,5])  # 列表变成张量，数组
print(tensor)

tensor = torch.Tensor((1,5))  # 元组变成张量，数组
print(tensor)

tensor = torch.Tensor(2,3)
print(tensor)

tensor = torch.Tensor(2,3,4)
print(tensor)

# 生成固定值的张量
tensor = torch.ones(2,3)
print(tensor)

tensor = torch.zeros(2,3)
print(tensor)

tensor = torch.tensor([[1,2,3,4,5,6]])
print(tensor)

