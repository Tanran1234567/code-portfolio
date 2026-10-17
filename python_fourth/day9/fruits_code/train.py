'''
训练测试，保存模型
'''
import os
import paddle.fluid as fluid
from model import *
from temp import *
from reader_creator import train_batch_reader,test_batch_reader
import numpy as np

# 调用模型得到预测值
pred_y = convolution_neural_network(x)

# 损失函数
cost = fluid.layers.cross_entropy(input=pred_y,label=y)
avg_cost = fluid.layers.mean(cost)

# 精度 accuracy
accuracy = fluid.layers.accuracy(input=pred_y,label=y)

# 梯度下降
fluid.optimizer.Adam(learning_rate=params['lr']).minimize(avg_cost)

# 克隆一个program用于测试模型(不带优化)  不带梯度下降
test_program = fluid.default_main_program().clone(for_test=True)



# 执行器
place = fluid.CPUPlace()  # 用CPU训练非常慢
# place = fluid.CUDAPlace(0)  # 用GPU训练要快得多(电脑显卡中的GPU/安装paddle-GPU,用软件模拟GPU性能/用远程服务器，去AI Studio，用a100)
exe = fluid.Executor(place=place)
exe.run(fluid.default_startup_program()) # 初始化


# 数据喂入器
feeder = fluid.DataFeeder(feed_list=[x,y],place=place)

# 在开始训练之前检查是否有模型保存
if os.path.exists(model_train_path):
    fluid.io.load_persistables(executor=exe,
                               dirname=model_train_path,
                               main_program=fluid.default_main_program())
    print('增量模型加载！')


# 开始训练
for pass_id in range(params['epoch']):
    train_costs,train_accs = [],[]  # 收集一轮中所有批次的损失值和精度
    for batch_id,train_data in enumerate(train_batch_reader()):
        train_cost,train_acc = exe.run(program=fluid.default_main_program(),
                                    feed=feeder.feed(train_data),
                                    fetch_list=[avg_cost,accuracy])

        # 收集损失值和精度
        # avg_cost, accuracy 都是列表，尽管可能就一个值
        train_costs.append(train_cost[0])
        train_accs.append(train_acc[0])

    # 求一轮中所有批次的平均值
    train_avg_cost = np.array(train_costs).mean()
    train_avg_acc = np.array(train_accs).mean()
    print(f'轮数:{pass_id},损失值:{train_avg_cost},精度:{train_avg_acc}')


# 测试模型
test_data = next((test_batch_reader()))
# 此处不能写成program=fluid.default_main_program()
# 因为测试数据时不能再进行梯度下降
test_acc = exe.run(program=test_program,
        feed=feeder.feed(test_data),
        fetch_list=[accuracy])
print(f':测试集精度:{test_acc[0][0]}')

# 保存增量模型
fluid.io.save_persistables(executor=exe,
                           dirname=model_train_path,
                           main_program=fluid.default_main_program())

# 保存推理模型
fluid.io.save_inference_model(dirname=model_infer_path,
                              feeded_var_names=['xxx'],
                              target_vars=[pred_y],
                              executor=exe)



