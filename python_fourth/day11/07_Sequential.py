'''
通过 Sequential 搭建模型
'''
import torch

class ModelNet(torch.nn.Module):
    def __init__(self):
        super(ModelNet,self).__init__()

        # Sequential一定顺序执行 所以定义顺序执行的Sequntial
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,5,
                            1,0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32,64,5,
                            1,0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,
                               stride=2)
        )

        self.linear = torch.nn.Sequential(
            # 11664的由来:需要自己算
            # 假设输入数据为 224*224
            # 首先，第一个卷积层过后，输出的二维数据，边长为(224-5+0)/1 + 1 = 220(第一个pdf有公式，自己回去看) ,故形状为220 * 220
            # 然后，第二个卷积层过后，输出的二维数据，边长为(220-5+0)/1 + 1 = 216, 故形状为220 * 220
            # 再者，第三层是池化层，输出的二维数据，边长为216/2 = 108，故形状为108 * 108
            torch.nn.Linear(108*108*64,512),
            torch.nn.Linear(512, 10),
            # torch.nn.Softmax(10)
        )

    def forward(self,x):
        # 调用卷积池化组
        feature_map = self.conv_block(x)

        # 将特征图拉伸成一维特征
        flat_feature = torch.flatten(feature_map,start_dim=1)
        # flat_feature = feature_map.view(-1,11664)  # 拉成一维
        # squeeze也能拉成一维度

        # 全连接
        pred_y = self.linear(flat_feature)

        return pred_y

model = ModelNet()
print(model)

for params in model.parameters():
    print(params.shape)












