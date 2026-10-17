'''
张量对张量自动微分
'''
import torch

x = torch.tensor([1,2],dtype=torch.float,requires_grad=True)
y = torch.tensor([2,3],dtype=torch.float,requires_grad=True)

res = 3 * x**2 - y**2
print(res)

ext = torch.ones_like(res)
print(ext)

ext1 = torch.tensor([1,2])
print(ext1)
# 张量对张量的求导
res.backward(gradient=ext1)

print('x_grad',x.grad)
print('y_grad',y.grad)























