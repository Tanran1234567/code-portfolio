'''
标量的自动微分
'''
import torch


# 输入数据
x = torch.tensor(5.0,requires_grad=False)
# 权重和偏置
w = torch.randn(size=(1,),requires_grad=True)
b = torch.ones(size=(1,),requires_grad=True)
print(x,w)
print(x.ndim)  # x为标量 维度为0
print(w.ndim)  # w为向量 维度为1

# 正向传播
y = torch.mul(w,x)  # y = w * x  # True
z = torch.add(y,b)  # z = y + b  # True

# 查看梯度开闭情况
print('x_grad',x.requires_grad)  # False
print('w_grad',w.requires_grad)  # True
print('b_grad',b.requires_grad)  # True
print('y_grad',y.requires_grad)  # True
print('z_grad',z.requires_grad)  # True

print('\n')
# 查看是否为叶子节点
# 右键列选择模式 或者鼠标中间可以选择不同行的列
print('x_leaf',x.is_leaf)  # True
print('w_leaf',w.is_leaf)  # True
print('b_leaf',b.is_leaf)  # True
print('y_leaf',y.is_leaf)  # False
print('z_leaf',z.is_leaf)  # False

print(x.grad)
print(y.grad)  # 还没开始反向传播  所以现在还没有梯度
print(y.grad_fn)

z.backward()  # 反向传播计算梯度

print('w_grad',w.grad)  # 5.0
print('b_grad',b.grad)  # 1.0
print('y_grad',y.grad)

print('y_grad',y.grad_fn)  # 地址
print('z_grad',z.grad_fn)  # 地址








