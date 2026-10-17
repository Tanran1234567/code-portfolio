'''
搭建模型
'''
import paddle.fluid as fluid
from temp import *
# 占位符
# 在 reader_creator中的最后查验出了x(img)的数据形状是(3,224,224)
x = fluid.layers.data(name='xxx',shape=(3,224,224),dtype='float32')
y = fluid.layers.data(name='yyy',shape=(1,),dtype='int64')

# 模型结构
def convolution_neural_network(image):
    # 第一个卷积池化组
    conv1 = fluid.nets.simple_img_conv_pool(input=image,
                                    num_filters=params['num_filters1'],
                                    filter_size=params['filter_size'],
                                    pool_size=params['pool_size'],
                                    pool_stride=params['pool_stride'],
                                    pool_padding=params['pool_padding'],
                                    pool_type=params['pool_type'],
                                    conv_padding=params['conv_padding'],
                                    conv_stride=params['conv_stride'],
                                    conv_groups=1,
                                    act =params['act']
                                    )

    # dropout
    # dropout_prob 丢弃率 过拟合严重 多丢一点
    drop1 = fluid.layers.dropout(x=conv1,dropout_prob=params['dropout_prob'])

    # 第二个卷积池化组
    conv2 = fluid.nets.simple_img_conv_pool(input=drop1,
                                            num_filters=params['num_filters2'],
                                            filter_size=params['filter_size'],
                                            pool_size=params['pool_size'],
                                            pool_stride=params['pool_stride'],
                                            pool_padding=params['pool_padding'],
                                            pool_type=params['pool_type'],
                                            conv_padding=params['conv_padding'],
                                            conv_stride=params['conv_stride'],
                                            conv_groups=1,
                                            act=params['act']
                                            )

    # dropout
    drop2 = fluid.layers.dropout(x=conv2, dropout_prob=params['dropout_prob'])

    # 第三个卷积池化组
    conv3 = fluid.nets.simple_img_conv_pool(input=drop2,
                                            num_filters=params['num_filters3'],
                                            filter_size=params['filter_size'],
                                            pool_size=params['pool_size'],
                                            pool_stride=params['pool_stride'],
                                            pool_padding=params['pool_padding'],
                                            pool_type=params['pool_type'],
                                            conv_padding=params['conv_padding'],
                                            conv_stride=params['conv_stride'],
                                            conv_groups=1,
                                            act=params['act']
                                            )

    # dropout
    drop3 = fluid.layers.dropout(x=conv3, dropout_prob=params['dropout_prob'])


    # 全连接
    fc = fluid.layers.fc(input=drop3,size=params['fc_size'],act=params['act'])

    # dropout
    drop4 = fluid.layers.dropout(x=fc,dropout_prob=params['dropout_prob'])

    # 输出层
    pred_y = fluid.layers.fc(input=drop4,size=class_num,act='softmax')

    return pred_y







