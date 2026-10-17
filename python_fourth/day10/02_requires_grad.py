'''
张量的requires_grad属性
'''
import torch
from torch.autograd import Variable

#新版本直接用张量
tensor = torch.tensor([1.0,2.0,3.0,4.0],requires_grad=True)
print(tensor)
print(tensor.requires_grad)

#老版本可以这么写（0.4）
var = Variable(tensor,requires_grad=True)
print(var)
print(var.requires_grad)