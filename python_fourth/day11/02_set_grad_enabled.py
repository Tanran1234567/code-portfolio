'''
打开梯度和关闭梯度
'''
import torch

x = torch.ones(size=(2,2),requires_grad=True)

# 关闭梯度
torch.set_grad_enabled(False)
# 后续所有的计算都是关闭梯度的
# 但是自己重新创建的变量是打开梯度的
# 但这个变量如果又进行了新运算，那么计算出的新变量也是关闭梯度的

y = x * 2
print(y.requires_grad)

z = torch.tensor(5.0,requires_grad=True)
print(z)

z2 = z * 2
print(z2.requires_grad)

# 打开梯度
torch.set_grad_enabled(True)
z3 = z * 3
print(z3.requires_grad)

# z2关闭 z4不管怎么样也不打开
z4 = z2 * 4
print(z4.requires_grad)










