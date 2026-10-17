'''
加载模型
'''
import torch
from w08_save_model import LinearRegression
# py文件不能以数字开头 否则 from不出来


# 必须要提供原模型结构
# 这是PyTorch的一个缺点
# 所以还不如保存模型参数
# 反正都还要把模型结构再写一遍的

# class LinearRegression(torch.nn.Module):  # 继承torch.nn.Module这个父类
#     def __init__(self):
#         # 继承父类的构造方法
#         super(LinearRegression,self).__init__()
#         self.linear = torch.nn.Sequential(
#             torch.nn.Linear(1, 2),
#             torch.nn.Sigmoid(),
#             torch.nn.Linear(2, 1)
#         )
#
#     def forward(self,x):
#         pred_y = self.linear(x)
#
#         return pred_y


# 加载模型
model = torch.load('../model/poly/model.pt')
# print(model)


test_x = torch.tensor([[1.1]])
print(model(test_x))  # grad_fn 非叶子节点
print(model(test_x).detach())


# 加载模型参数
model_param = torch.load('../model/poly/model_param.pt')
model1 = LinearRegression()
model1.load_state_dict(model_param)

print(model1(test_x))
print(model1(test_x).detach())












