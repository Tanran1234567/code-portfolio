import paddle.fluid as fluid
import numpy as np


# 损失函数
def loss(logit, label, num_classes):
    # assign：将输入张量或数组拷贝到新张量中(或拷贝生成一个新张量)
    # less_than:逐元素比较两个张量，返回一个同纬度布尔类型张量
    label_nignore = fluid.layers.less_than(label.astype('float32'),
                                           fluid.layers.assign(np.array([num_classes], 'float32')),
                                           force_cpu=False).astype('float32')

    # transpose:根据perm对输入的多维Tensor进行数据重排。返回多维Tensor的第i维对应输入Tensor的perm[i]维
    logit = fluid.layers.transpose(logit, [0, 2, 3, 1])  # 将预测结果的维度进行重排，将通道维度放在最后
    logit = fluid.layers.reshape(logit, [-1, num_classes])
    label = fluid.layers.reshape(label, [-1, 1])
    label = fluid.layers.cast(label, 'int64')
    label_nignore = fluid.layers.reshape(label_nignore, [-1, 1])
    logit = fluid.layers.softmax(logit, use_cudnn=False)
    loss = fluid.layers.cross_entropy(logit, label, ignore_index=8)  # 不考虑值为8的元素
    label_nignore.stop_gradient = True
    label.stop_gradient = True

    return loss, label_nignore


def optimizer_momentum_setting(*args, **kwargs):
    learning_rate = fluid.layers.polynomial_decay(kwargs["base_lr"],
                                                  kwargs["total_step"],
                                                  end_learning_rate=0,
                                                  power=0.9)
    
    optimizer = fluid.optimizer.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=0.1,
        regularization=fluid.regularizer.L2DecayRegularizer(regularization_coeff=kwargs["weight_decay"]))

    return optimizer
