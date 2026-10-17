'''
加载模型，执行预测
'''

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import paddle.fluid as fluid

import numpy as np
from temp import *
import paddle
from PIL import Image

# 推理没几张图片，就用CPU不会太慢  训练模型必须要用GPU
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place=place)



# 加载推理模型
(infer_program,feed_name,target_name) = fluid.io.load_inference_model(model_infer_path,
                                                                      infer_exe)

# 将待预测数据进行处理，处理成和训练模式格式一样
def load_img(path):
    # paddle.dataset.image.load_image()
    # paddle.dataset.image.simple_transform()
    # 两行代码转变成一行来完成  参见reader_creator
    img = paddle.dataset.image.load_and_transform(path,
                                            params['image_size'],
                                            params['image_size'],
                                            False,
                                            True)
    # 归一化
    img = img.astype('float32') / 255.0
    return img

# 把待预测的图像路径拼接成完整路径
test_data_path = []  # 保存每一张测试图像的路径
test_data = []   # 保存每一张测试图像的具体图像数据
for img in os.listdir(test_img_path):
    if not img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        continue
    all_path = os.path.join(test_img_path, img)
    test_data_path.append(all_path)
    test_data.append(load_img(all_path))

test_data = np.array(test_data)
print(test_data.shape)  # 2,3,224,224

# 执行预测
result = infer_exe.run(program=infer_program,
              feed={feed_name[0]:test_data},
              fetch_list=target_name)

print(result[0])  # shape = [2,5]

# 预测类别
pred_label = np.argmax(result[0],axis=1)  # 取索引值
print(pred_label)

# 拿概率
pred_val = np.max(result[0],axis=1)

for i in range(len(pred_val)):
    pred_lab = name_dict_inv[pred_label[i]] # 字符串的预测
    pred_v = pred_val[i]
    print(f'第{i+1}张图像，预测类别:{pred_lab},概率:{pred_v}')

    # 显示图像
    img = Image.open(test_data_path[i])
    plt.subplot(1,2,i+1)
    plt.imshow(img)
    plt.xticks([])  # 刻度清空
    plt.yticks([])
    plt.xlabel('{}:{:.2f}'.format(pred_lab,pred_v),fontsize=24)
    

plt.show()





