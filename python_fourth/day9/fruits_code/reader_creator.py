'''
数据读取器的创建
'''
import paddle
import os
from multiprocessing import cpu_count
from temp import *

# 对 yield img_path,int(label)的两个返回值作二次处理
def train_mapper(sample):
    img_path,label = sample
    if os.path.exists(img_path):
        # 根据路径读取图像
        # 将图像通过数组的形式表示出来
        img = paddle.dataset.image.load_image(img_path,is_color=True)

        # 固定图像的大小
        img = paddle.dataset.image.simple_transform(img,
                                                    params['image_size'],
                                                    params['image_size'],
                                                    True,
                                                    True)

        # 归一化
        img = img.astype('float32') / 255.0
        return img,label

def test_mapper(sample):
    img_path,label = sample
    if os.path.exists(img_path):
        # 根据路径读取图像
        img = paddle.dataset.image.load_image(img_path,is_color=True)

        # 固定图像的大小
        img = paddle.dataset.image.simple_transform(img,
                                                    params['image_size'],
                                                    params['image_size'],
                                                    False,
                                                    True)

        # 归一化
        img = img.astype('float32') / 255.0
        return img,label


# 训练集的处理器
def train_r(train_path):
    def reader():
        with open(train_path,'r') as f:
            for line in f.readlines():  # 路径\t类别\n
                img_path,label = line.strip().split('\t')
                yield img_path,int(label)  # label 本来是int，但文件写入和读取之后变成了str，还要再转回来

    return paddle.reader.xmap_readers(mapper=train_mapper,
                                      reader=reader,
                                      process_num=cpu_count(),
                                      buffer_size=1173)


# 测试集的处理器
def test_r(test_path):
    def reader():
        with open(test_path,'r') as f:
            for line in f.readlines():  # 路径\t类别\n
                img_path,label = line.strip().split('\t')
                yield img_path,int(label)  # label 本来是int，但文件写入和读取之后变成了str，还要再转回来

    return paddle.reader.xmap_readers(mapper=test_mapper,
                                      reader=reader,
                                      process_num=cpu_count(),
                                      buffer_size=133)


# 训练集的读取器
train_reader = train_r(train_file_path)
train_shuffle_reader = paddle.reader.shuffle(train_reader,1173)
train_batch_reader = paddle.batch(train_shuffle_reader,
                                  batch_size=params['batch_size'])
# 1173/32 余20  最后一个批次有20个  不少了  所以drop_last不需要为True

# 测试集的读取器
test_reader = test_r(test_file_path)
test_batch_reader = paddle.batch(train_reader,
                                  batch_size=200)
# 一共133条数据  取200条  就是想一次性  一个批次全部取出来

# 本以为x(img)的数据形状是(224,224,3)
# 结果发现是(3,224,224)  # 可以查验出来

# 先将 train_batch_reader中的batch_size=1,然后用next(train_batch_reader())
# 可以得到第一批次(此时只有一个训练样本)  方便检查数据形状

# 如果这三行不注释掉，运行train.py这个文件时也会输出这三行的代码所对应的结果
# Python 模块导入的工作原理：
# 当你 import 一个模块时，Python 会执行该模块的所有顶层代码
# 函数和类定义会被执行（创建函数/类对象）
# 所有的 print()、函数调用等语句都会立即执行
# if __name__ == "__main__": 中的代码只有在直接运行该文件时才会执行

# print(next(train_batch_reader()))  # 发现是[(x,y)]
# img_datalize = next(train_batch_reader())[0][0]  # 对应的是x
# print(img_datalize.shape)  # 发现打印出的是(3,224,224)  与预想中的有差距

if __name__ == '__main__':
    print(next(train_batch_reader()))  # 发现是[(x,y)]
    img_datalize = next(train_batch_reader())[0][0]  # 对应的是x
    print(img_datalize.shape)  # 发现打印出的是(3,224,224)  与预想中的有差距







