'''
利用 LeNet 训练 Cifar10 数据集
'''
import os.path

import torch
from torch.utils import data
import torchvision
from torch import nn


# ToTensor() 主要完成两个操作：
# 数据结构转换：将 PIL Image 或 numpy.ndarray 转换为 PyTorch Tensor
# 数值范围转换：将像素值从 [0, 255] 归一化到 [0, 1]（除以255）

# 数据准备
def read_data():
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),  # 转成张量,归一化,从0-255变为0-1
         torchvision.transforms.Normalize(mean=(0.5,0.5,0.5),
                                          std=(0.5,0.5,0.5))]  # 从0-1变为-1-1
    )

    # 训练集
    train_set = torchvision.datasets.CIFAR10('../',
                                             train=True,
                                             transform=transform,
                                             download=True)

    train_reader = data.DataLoader(train_set,
                                   batch_size=32,
                                   shuffle=True,
                                   num_workers=0)

    # 测试集
    test_set = torchvision.datasets.CIFAR10('../',
                                             train=False,
                                             transform=transform,
                                             download=True)

    test_reader = data.DataLoader(test_set,
                                  batch_size=10000,
                                  shuffle=False,
                                  num_workers=0)

    test_reader = iter(test_reader)   # 创建迭代器
    test_image,test_label = next(test_reader)  # 获取第一个批次
    print(test_image.shape)
    print(test_label)
    print(torch.unique(test_label))

    return train_reader,test_image,test_label


# 搭建模型结构:
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        # 卷积池化组
        self.conv_block = nn.Sequential(
            # 第一个卷积池化组，输入数据:3,32,32
            nn.Conv2d(3,16,5), # 16,28,28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2), # 16,14,14
            # 第二个卷积池化组
            nn.Conv2d(16, 32, 5),  # 32,10,10
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32,5,5
        )

        # 全连接层
        self.classifer = nn.Sequential(
            # 第一层全连接
            nn.Linear(32*5*5,120),
            nn.ReLU(inplace=True),
            # 第二层全连接
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            # 输出层
            nn.Linear(84, 10)

        )

    def forward(self,x):
        # 卷积池化组
        feature_map = self.conv_block(x)
        # 将特征图拉伸成1维
        flat_feature = torch.flatten(feature_map,start_dim=1)
        # 全连接
        out = self.classifer(flat_feature)
        # Softmax
        pred_y = nn.Softmax(dim=1)(out)

        # dim = 1
        # 指定了沿着哪个维度进行Softmax计算，这是非常重要的参数。
        # 详细解释
        # 张量形状理解  假设out的形状是[batch_size, num_classes]：
        # batch_size: 批次中的样本数量
        # num_classes: 类别数量

        return pred_y



def main():
    train_reader,test_image,test_label = read_data()
    # 实例化模型
    model = LeNet()
    # 损失函数
    loss = nn.CrossEntropyLoss()
    # 梯度下降优化器
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.001)

    model_save_path = '../model/Cifar/'
    if os.path.exists(model_save_path+'model.pt'):
        model.load_state_dict(torch.load(model_save_path+'model.pt'))
        print('增量模型加载成功！！！')

    # 开始训练
    for pass_id in range(1):
        train_losses = 0.0
        for batch_id,train_data in enumerate(train_reader):
            train_x,train_y = train_data
            # 正向传播
            pred_y = model(train_x)
            # 损失函数
            train_loss = loss(pred_y,train_y)
            # 反向传播
            train_loss.backward()
            # 更新参数
            optimizer.step()
            # 梯度清零
            optimizer.zero_grad()
            # 累加损失值  梯度清零和损失值没有关系
            train_losses += train_loss.item()
            # 打印
            if batch_id % 500 == 499:
                print(f'轮数:{pass_id},批次:{batch_id},train_loss:{train_losses/500}')
                train_losses = 0.0

    print('训练完成')

    # 精度
    pred_test_y = model(test_image)
    pred_label = torch.argmax(pred_test_y,dim=1)

    acc = torch.eq(pred_label,test_label).sum().item()/ test_label.size(0) # 相等为true，,为1，不等为False，为0
    # print(torch.eq(pred_label,test_label).sum())
    # print(torch.eq(pred_label,test_label).sum().item())
    # print(test_label.size())
    # print(test_label.size(0))
    print(f'测试集精度:{acc}')

    # 保存模型
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    torch.save(model.state_dict(),model_save_path+'model.pt')
    print('最新模型保存成功')


# 1. 理解 flatten(start_dim=1) 的作用
# 假设 feature_map 的形状是 [batch_size, C, H, W]：
# batch_size: 批次大小
# C: 通道数
# H: 高度
# W: 宽度
# flatten(start_dim=1) 会：
# 保持第0维（batch_dim）不变
# 将第1维及之后的所有维度展平

# 如果知道批量大小
# batch_size = feature_map.size(0)
# flat_feature = feature_map.view(batch_size, -1)

if __name__ == '__main__':
     # read_data()
     main()















