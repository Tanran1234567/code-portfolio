'''
打开梯度和关闭梯度
'''
import torch

x = torch.ones(size=(3,3),requires_grad=True)
y = x * 2

print(x,x.requires_grad)
print(y,y.requires_grad)

# 关闭梯度
with torch.no_grad():
    z = x * 2

print(z,z.requires_grad)


x = torch.ones(size=(2,2),requires_grad=False)
a = torch.ones(size=(2,2),requires_grad=True)
k = torch.ones(size=(2,2),requires_grad=False)
y = x * 2

print(x,x.requires_grad)
print(y,y.requires_grad)

with torch.enable_grad():  # 打开梯度
    y = x * 2
    z = x + a
    l = x + k

    print(y,y.requires_grad)  # 因为x不能求梯度 所以y不管怎么处理都不能求梯度
    print(z,z.requires_grad)
    print('哈哈',l,l.requires_grad)

b = x + a
print(b,b.requires_grad)  # 因为a打开了，有一个打开，b就打开


x = torch.ones(size=(1,),requires_grad=False)
a = torch.ones(size=(1,),requires_grad=True)
y = x * a
print(y,y.requires_grad)



