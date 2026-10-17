'''
存放公共变量
'''

data_root_path = '../../fruits/'  # 数据所在路径
train_file_path = data_root_path + 'train.txt'  # 训练集文件路径
test_file_path = data_root_path + 'test.txt'  # 测试集文件路径
# 在 Python 中，+ 是字符串连接运算符，用于将两个字符串拼接在一起
# 根据你的代码，训练集文件的完整路径是：
# ../../fruits/train.txt


name_dict ={'apple':0,'banana':1,'grape':2,'orange':3,'pear':4}
name_dict_inv ={0:'apple',1:'banana',2:'grape',3:'orange',4:'pear'}
class_num = 5  # 类别数量
model_train_path = '../../model/fruits/train/'
model_infer_path = '../../model/fruits_prediction/infer/'
test_img_path = '../../test_data'

# 模型超参数
params={
    'image_size':224,   # 图像的尺寸
    'batch_size':32,
    'num_filters1':32,  # 第一组的卷积核数量
    'num_filters2':64,  # 第二组的卷积核数量
    'num_filters3':64,  # 第三组的卷积核数量
    'filter_size':3, # 卷积核的尺寸
    'pool_size':2, # 池化区域大小
    'pool_stride':2, # 池化的步长
    'pool_padding':0, # 池化的填充
    'pool_type':'max', # 池化的类型
    'conv_stride':1, # 卷积的步长
    'conv_padding':0, # 卷积的填充
    'act':'relu', # 激活函数
    'dropout_prob':0.5, # 丢弃率
    'fc_size':512, # 全连接层的神经元数量
    'lr':0.001, # 学习率
    'epoch':5, # 轮数  # 先把模型跑通 后再做增量训练
}














