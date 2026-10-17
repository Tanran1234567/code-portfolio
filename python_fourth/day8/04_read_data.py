'''
数据读取
'''
import paddle
import numpy as np
from multiprocessing import cpu_count

# 将读取的数据从str转变为 float 数组
# 二次处理
def reader_mapper(sample):
    x,y = sample
    x = np.array([x],dtype='float32')
    y = np.array([y], dtype='float32')
    return (x,y)

# pandas只能读出来dataframe
# 此处要读出来张量
def reader_creator(file_path):
    def reader():
        # 以r(只读)的方式打开
        with open(file_path,'r') as f:
            # f.readlines() 返回值是一个列表，第一行、第二行……
            for line in f.readlines():
                x,y = line.strip().split(',')  # 干掉换行符
                yield x,y
    # return reader
    return paddle.reader.xmap_readers(mapper=reader_mapper,
                                      reader=reader,
                                      process_num=cpu_count(),
                                      buffer_size=10)
# process_num 是进程数  一般与电脑cpu核数匹配  保证快速读入和处理


# 原始读取器
reader = reader_creator('./demo_data.txt')

# for line in reader():
#     print(line)

# 随机读取器
shuffle_reader = paddle.reader.shuffle(reader,buf_size=10)
# buf_size表示在几组数据中随机排列

# for line in shuffle_reader():
#     print(line)


# 随机批量读取器
batch_reader = paddle.batch(shuffle_reader,3,drop_last=True)
# drop_last=True 丢掉最后一个批次(只有一个数据了)

for line in batch_reader():
    print(line)


# open(file_path, 'r')
# file_path：要打开的文件的路径（字符串类型，如 "data.txt"）。
# 'r'：表示以 只读模式（read） 打开文件（默认是文本模式）。
# 返回值：返回一个 文件对象（file object），后续通过它操作文件内容。

# with ... as f
# with 语句：上下文管理器，确保文件在使用后自动关闭，即使发生异常也不会泄漏资源。
# as f：将文件对象赋值给变量 f，方便后续调用（如 f.readlines()）

# with 的优势：
# 自动管理资源：退出 with 块时，文件自动关闭（调用 f.close()）。
# 避免遗忘关闭：防止因未关闭文件导致内存泄漏或文件锁冲突。

# 为什么用 yield 而不用 return？
# yield 使函数变为 生成器，逐行返回数据，避免一次性加载大文件到内存。









