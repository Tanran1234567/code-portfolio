'''
保存模型
'''
import torch
import matplotlib.pyplot as plt
import os

# 数据准备
x = torch.linspace(-1,1,50).reshape(50,1)
# unsqueeze指插入一个维度, 否则张量为一维
# x = Variable(torch.unsqueeze(torch.linspace(-1, 1, 50), dim=1))
# 与上述方法殊途同归


# y=2x+5
y = x * 2 + 5 + torch.randn(x.size())
print(y)

# 为什么不能使用一维？
# 神经网络输入要求
# torch.nn.Linear(1, 1) 期望的输入形状是 [batch_size, input_features]
# nn.Linear 期望 [batch_size, input_features] 格式
# 一维输入 [50]：被解释为50个特征的一个样本 ❌
# 二维输入 [50, 1]：50个样本，每个样本1个特征 ✅

class LinearRegression(torch.nn.Module):  # 继承torch.nn.Module这个父类
    def __init__(self):
        # 继承父类的构造方法
        super(LinearRegression,self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(1, 2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(2, 1)
        )

    def forward(self,x):
        pred_y = self.linear(x)

        return pred_y

if __name__ == '__main__':
    model = LinearRegression()
    # 损失函数
    loss = torch.nn.MSELoss()

    # print(list(model.parameters()))  # 模型参数不需要我们自己设置

    # 梯度下降
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=0.01)

    iters,costs = [],[]

    for pass_id in range(500):  # 训练轮数
        pred_y = model(x)  # 预测值
        train_loss = loss(pred_y,y)  # 损失函数
        train_loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        optimizer.zero_grad()  # 梯度清零,防止梯度累计  因为在w.grad中都是累加值


        print(f'轮数:{pass_id},loss:{train_loss}')

        # 收集每一轮的损失值进行可视化
        iters.append(pass_id)
        costs.append(train_loss.item())

    plt.plot(iters,costs)
    plt.show()


    # 保存模型
    model_save_path = '../model/poly/'
    if not os.path.exists(model_save_path):
        # 创建文件夹  原先存文件时无需先检验存不存在  是因为文件可以主动存在文件夹里
        # 但是文件夹不能自动生成  需要手动代码实现
        os.makedirs(model_save_path)
    torch.save(model,model_save_path+'model.pt')  # .pt 指的是训练后的模型的参数
    print('模型保存成功！！！！！！')


    # 只保存模型参数
    torch.save(model.state_dict(),model_save_path+'model_param.pt')












