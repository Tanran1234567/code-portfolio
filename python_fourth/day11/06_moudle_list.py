'''
通过 moudlelist 搭建模型
'''
import torch

class ModelNet(torch.nn.Module):
    def __init__(self):
        super(ModelNet,self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(100,100) for i in range(3)])

    def forward(self,x):
        pred_y1 = self.linears[0](x)
        pred_y2 = self.linears[1](pred_y1)
        pred_y = self.linears[2](pred_y2)
        return pred_y

model = ModelNet()
print(model)

for params in model.parameters():
    print(params.shape)












