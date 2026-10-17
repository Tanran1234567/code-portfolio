'''
TensorDataset
'''
import torch
from torch.utils import data

x = torch.linspace(-10,10,1000)
y = torch.linspace(-10,10,1000)

# 打包数据
test_data = data.TensorDataset(x,y)
loader = data.DataLoader(test_data,
                         batch_size=100,
                         shuffle=True,
                         num_workers=2,
                         drop_last=False)

if __name__ == '__main__':
    for pass_id in range(5):  # 外层循环控制轮数
        for batch_id,batch_data in enumerate(loader):
            print(f'批次:{batch_id},数据:{batch_data}')




















