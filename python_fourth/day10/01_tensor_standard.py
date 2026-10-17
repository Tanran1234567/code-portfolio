'''
张量的基本创建
'''
import torch
import numpy as np
from pkg_resources import require

#创建一个张量
tensor = torch.tensor(5)
print(tensor)

#将数组转成tensor
ary = np.arange(1,10).reshape(3,3)
tensor = torch.tensor(ary)
print(tensor)

#生成指定形状的张量
tensor = torch.Tensor(5) #(5,)
print(tensor)
tensor = torch.Tensor(2,3) #(2,3)
print(tensor)
tensor = torch.Tensor(2,3,4)
print(tensor)

#生成指定元素类型的张量
tensor = torch.FloatTensor(1,5) #(1,5)
print(tensor)
tensor = torch.FloatTensor((1,5))
print(tensor)
tensor = torch.DoubleTensor((1,5))
print(tensor)
tensor = torch.IntTensor([1,5])
print(tensor)
tensor = torch.LongTensor([1,5])
print(tensor)

#生成固定值的张量
tensor = torch.ones(2,3)
print(tensor)
tensor = torch.zeros(2,3)
print(tensor)
tensor = torch.tensor([[1,2,3,4,5,6]])
print(tensor)
ones = torch.ones_like(tensor)
print(ones)
zeros = torch.zeros_like(tensor)
print(zeros)

#创建随机数的张量
tensor = torch.randn(size=(1,5)) #标准正态分布 期望值0，标准差1
print(tensor)
tensor = torch.normal(mean=100,std=200,size=(1,8))
print(tensor)
tensor = torch.rand(size=(1,5)) #[0,1)
print(tensor)

#判断当前数据是否为张量
tensor = torch.Tensor([1,2])
print(torch.is_tensor(tensor)) #True
print(torch.is_tensor([1,2])) #False

#转换张量的元素类型
tensor = torch.FloatTensor([1,2,3,4])
print(tensor)
tensor_int = tensor.type(dtype=torch.IntTensor)
print(tensor_int)
tensor_float = tensor_int.float() #int()  #double()
print(tensor_float)

#张量的自变化  a += 1   a -= 1
tensor_100 = torch.IntTensor([100])
tensor_200 = torch.IntTensor([200])

tensor_100.add_(tensor_200)
print(tensor_100)
tensor_200.sub_(tensor_100)
print(tensor_200)

#将numpy数组转成张量
ary = np.arange(1,10)
print(torch.tensor(ary))
print(torch.Tensor(ary))  # shape=1,5  [1,5] #float32
print(torch.FloatTensor(ary))
print(torch.from_numpy(ary))

#将张量转成数组
tensor = torch.tensor(ary)
print(tensor)
print(tensor.numpy())

#测试数组转张量，共用同一份数据   tensor()不是同一份数据
ary = np.array([1,2,3,4,5])
tensor = torch.from_numpy(ary)
#修改数组中的元素
ary = ary + 10  # ary += 10
print(ary)
print(tensor)
#修改张量
# tensor.add_(100)
# print(tensor)
# print(ary)


#张量的形状的改变   reshape

ary = np.arange(1,10)
tensor = torch.tensor(ary)
print(tensor)

# tensor = torch.reshape(tensor,(3,3))
# print(tensor)

# tensor = tensor.reshape(shape=(3,3))
# print(tensor)

tensor = tensor.reshape(shape=(-1,3))
print(tensor)

#张量组合与拆分
#组合 np.concatenate  pd.concat  torch.cat
#二维张量组合  dim=0  dim=1

tensor_a = torch.Tensor(2,2)
tensor_b = torch.Tensor(2,2)
print(torch.cat([tensor_a,tensor_a],dim=0)) #(4,2)
print(torch.cat([tensor_a,tensor_b],dim=1)) #(2,4)

#三维张量组合  dim=0  dim=1  dim=2
tensor_a = torch.Tensor(3,2,2)
tensor_b = torch.Tensor(3,2,2)
print(torch.cat([tensor_a,tensor_b],dim=0).shape) #(6,2,2)
print(torch.cat([tensor_a,tensor_b],dim=1).shape) #(3,4,2)
print(torch.cat([tensor_a,tensor_b],dim=2).shape) #(3,2,4)

#拆分
tensor = torch.Tensor(4,2)
x,y = torch.chunk(tensor,2,dim=1) #0 1
print(x)
print(y)

#指定片段大小的拆分
tensor = torch.Tensor(5,4)
res = torch.split(tensor,2,dim=0)
for data in res:
    print(data)



























