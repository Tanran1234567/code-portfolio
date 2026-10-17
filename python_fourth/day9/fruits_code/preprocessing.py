'''
数据标注，训练集和测试集的划分
'''

from temp import *
import os

name_data_dict = {}   # {'apple':[xx,xx,xx],'banana':[xx,xx,xx],……}

dirs = os.listdir(data_root_path)
print(dirs)

for d in dirs: # apple
    full_path = os.path.join(data_root_path,d)  # ../../fruits/apple
    name_list = []
    # 判断当前文件是否为目录(文件夹)
    if os.path.isdir(full_path):
        imgs = os.listdir(full_path)  # [1.jpg,2.jpg,……]
        # print(imgs)

        # 拼接完整路径

        # 没有列表就创建一个列表  有列表就不创建
        # name_data_dict.setdefault(d,[])  # {'apple':[]}

        for img in imgs:
            all_path = os.path.join(full_path,img)  # ../../fruits/apple/1.jpg
            # print(all_path)
            name_list.append(all_path)

            # # 另解
            # if d not in name_data_dict.keys():
            #     name_data_dict[d] = [all_path]
            # else:
            #     name_data_dict[d].append(all_path)

            # # 又一个另解
            # # 结合 name_data_dict.setdefault(d,[])  # {'apple':[]}这行代码
            # name_data_dict[d].append(all_path)
        name_data_dict[d] = name_list

# print(name_data_dict)

# for name,val in name_data_dict.items():
#     print(f'类别:{name},数量:{len(val)}')


# 划分训练集和测试集，并进行数据标注    路径\t类别  \t 制表符

# 每次程序运行前先清空  免得多次运行程序之后写入train和test测试集的东西越来越多
# with open(train_file_path,'w') as f:
#     pass
#
# with open(test_file_path,'w') as f:
#     pass

# 该方法的with open在for循环里  说明文件要不停地打开写入  而os库中函数(文件读取和写入函数)执行的本来相较于普通函数 就更慢一些  所以运行效率不高  时间较长
# for name,img_list in name_data_dict.items():
#     for i,img in enumerate(img_list):
#         if i % 10 == 0:  # 写入测试集
#             line = '%s\t%d\n' % (img,name_dict[name])
#             with open(test_file_path,'a') as f:
#                 f.write(line)
#         else: # 写入训练集
#             line = '%s\t%d\n' % (img,name_dict[name])
#             with open(train_file_path,'a') as f:
#                 f.write(line)


# 用with打开两个文件
with open(train_file_path,'w') as train_f,open(test_file_path,'w') as test_f:
    train_lines,test_lines = [],[]
    for name,img_list in name_data_dict.items():
        for i,img in enumerate(img_list):
            # 数据标注
            line = '%s\t%d\n' % (img, name_dict[name])
            if i % 10 == 0:  # 写入测试集
                test_lines.append(line)
            else:
                train_lines.append(line)

    # 一次性写入
    train_f.writelines(train_lines)
    test_f.writelines(test_lines)

