'''
加载数据  读取数据
'''
import torch
from torch.utils import data
import numpy as np

class TestData(data.Dataset):
    def __init__(self):
        self.x = np.array([[1,2],[3,4]])
        self.y = np.array([1,3])

    def __getitem__(self, item):
        train_x = torch.tensor(self.x[item])
        train_y = torch.tensor(self.y[item])
        return train_x,train_y


    def __len__(self):
        return len(self.x)

# 原始读取器
loader = TestData()
print(loader[0])
# print(loader(0))  # 会报错  是索引值

# 随机批量读取器
shuffle_batch_loader = data.DataLoader(dataset=loader,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=2,  # 要么就改成0  解决多进程问题  只在主进程中进行
                                       drop_last=False)

# 工作进程数：用于数据加载的子进程数量
# num_workers=0：在主进程中加载数据（默认）
# num_workers=2：使用2个子进程并行加载，加速数据准备
# 建议设置为CPU核心数的一半或全部

# for batch_id,train_data in enumerate(shuffle_batch_loader):
#     print(f'批次:{batch_id},data:{train_data}')

if __name__ == '__main__':
    for batch_id,train_data in enumerate(shuffle_batch_loader):
        print(f'批次:{batch_id},data:{train_data}')

# Windows系统多进程问题解析
# 这是一个典型的 Windows系统多进程问题，与Linux系统的行为差异导致的。

# 问题根源
# Windows vs Linux 进程创建机制不同
# 特性	      Linux/MacOS	            Windows
# 进程创建	  fork() - 复制当前进程	    spawn - 创建新进程
# 内存共享	  共享内存空间	            完全独立内存
# 导入机制	  继承所有已导入模块	        重新导入所有模块
# 主要问题	  较少	                    需要防止递归导入

# 为什么Linux没问题？
# Linux使用fork()创建进程：
# 子进程继承父进程的所有状态
# 包括已导入的模块和内存状态
# 不会重新执行导入语句

# Windows使用spawn创建进程：
# 创建全新的Python解释器
# 重新执行所有导入语句
# 如果导入的文件中有直接执行的代码，会导致递归

# Windows上的执行流程：
# 主进程执行到 for batch in loader:
# 启动2个子进程（num_workers=2）
# 每个子进程重新执行整个10_load_data.py脚本
# 子进程又遇到 for batch in loader:，尝试启动新的子进程
# 无限递归，导致错误

# 使用 if __name__ == '__main__': 的情况
# Windows上的正确执行流程：
# 主进程执行到 if __name__ == '__main__':，进入循环
# 启动2个子进程（num_workers=2）
# 每个子进程重新执行脚本，但不会进入main块
# 子进程只执行模块级别的导入和定义
# 子进程等待数据加载任务，不会递归



