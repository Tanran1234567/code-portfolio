'''
将 labelme的标注的 json文件，转成每个像素对应类别的文件
'''
import json

import numpy as np
import cv2


def create_mask_form_labelme(json_flie,output_file):
    # 背景像素为 0类别，其他目标为1,2,3,4……

    # 读取标注文件
    with open(json_flie,'r',encoding='utf-8') as f:
        data = json.load(f)
        # data 就是python的字典,load实现了json到dict的转化
        # print(data)
        # print(type(data))

        # 获取图像的尺寸
        height = data['imageHeight']
        width = data['imageWidth']

        #创建一张值全为0的空白掩码
        mask = np.zeros(shape=(height,width),dtype=np.uint8)

        # 收集所有的标签类别
        labels = []
        for shape in data['shapes']:
            if shape['label'] not in labels:
                labels.append(shape['label'])
        print(f'目标类别有:{labels}')

        # 为每个标签分配一个像素值
        label_to_value = {}  # {'apple':1}
        for i,label in enumerate(labels):
            label_to_value[label] = i + 1

        # 遍历所有标注的形状
        for shape in data['shapes']:
            if shape['shape_type'] == 'polygon':  # 多边形
                # 获取多边形的点
                points = shape['points']  # 二维列表(从json中能够看出来)
                label = shape['label']
                pixel_value = label_to_value[label]  # 1(苹果是1)

                polygon_points = np.array([[[point[0],point[1]]] for point in points],dtype='int32')

                # opencv填充多边形
                # 三维数组(列表)套列表  输出坐标的格式  可以在前面的代码中去查看
                cv2.fillPoly(mask, [polygon_points], pixel_value)

        np.savetxt(output_file, mask, fmt='%d', delimiter='')





if __name__ == '__main__':
    # create_mask_form_labelme('../test_data/apple_test_label.json')
    create_mask_form_labelme('../test_data/apple_test_label.json',
                             '../test_data/apple_test_label.txt')


























