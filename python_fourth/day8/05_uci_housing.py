'''
波士顿房价预测
保存模型: 增量训练的模型(只能用于训练，不能用于预测)，推理模型(paddlepaddle分的细)
pytorch 和 Tensorflow 将之放在一起
增量模型  意思就是  今天只练一部分(练不完)  明天接着今天的练  故为增量
'''
import paddle
import paddle.fluid as fluid
import os

# 数据准备
train_reader = paddle.dataset.uci_housing.train()
# L = []
# for i in train_reader():
#     L.append(i)
# print(len(L))   # 404 条数据样本

train_shuffle_reader = paddle.reader.shuffle(train_reader,404)
train_batch_reader = paddle.batch(train_shuffle_reader,batch_size=32)
print(next(train_batch_reader()))   # 获取首个批次  一个批次只能拿32个样本
# 一次训练一共有13个批次(最后一个批次拿20个样本，也不少，所以不需要drop_last)

# train_batch_reader为[(x,y),(x,y),(x,y),……]   x=(13,)  y=(1,)

# 占位符
x = fluid.layers.data(name='xxx',shape=(13,),dtype='float32')
y = fluid.layers.data(name='yyy',shape=(1,),dtype='float32')

# 构建模型
pred_1 = fluid.layers.fc(input=x,size=512,act='sigmoid')
pred_2 = fluid.layers.fc(input=pred_1,size=512,act='sigmoid')
pred_y = fluid.layers.fc(input=pred_2,size=1)

# 损失函数
cost = fluid.layers.square_error_cost(pred_y,y)
avg_cost = fluid.layers.mean(cost)

# 梯度下降
# Adam()  高级优化器  自适应学习率 动量衰减(学习率)
fluid.optimizer.Adam(learning_rate=0.001).minimize(avg_cost)
# 求完损失函数的极小值会自动的反向传播


# 执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place=place)
exe.run(fluid.default_startup_program())   # 初始化


# 数据喂入器
feeder = fluid.DataFeeder(feed_list=[x,y],place=place)



model_train_path = '../model/boston/train/'
# 在开始训练之前，检查是否有模型被保存  如果有就加载
# 需要打检查点  防止后一个模型还没有前一个模型训练得好
if os.path.exists(model_train_path):
    # 加载增量模型
    # 执行器要保证是同一个  将原先的模型加载到现在的执行器exe上面
    fluid.io.load_persistables(executor=exe,
                               dirname=model_train_path,
                               main_program=fluid.default_main_program())
    print('增量模型加载成功！！！继续训练！！！')


# 开始训练
for pass_id in range(50):  # 外层循环控制轮数   一轮有13个批次  一个批次取32个样本
    for batch_id,train_data in enumerate(train_batch_reader()):
        # batch_id 为当前的批次数
        # train_data 为当前批次的数据
        train_cost = exe.run(program=fluid.default_main_program(),
                             feed=feeder.feed(train_data),
                             fetch_list=[avg_cost])
        # 取出这个批次所有样本中的X数据(13个特征)
        # [data[0] for data in train_data]
        if (batch_id + 1 ) % 5 == 0:
            print(f'轮数:{pass_id},批次:{batch_id+1},cost:{train_cost[0][0]}')

# 保存增量训练的模型
# model_train_path = '../model/boston/train/'
fluid.io.save_persistables(executor=exe, # 执行器
                           dirname=model_train_path, # 模型保存路径
                           main_program=fluid.default_main_program())
print('增量训练模型保存成功！！！！！！')





