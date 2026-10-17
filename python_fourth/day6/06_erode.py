'''
图像的腐蚀
'''
import cv2
import numpy as np
img = cv2.imread('../data/5.png',0)
cv2.imshow('img',img)

# 腐蚀核
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img, # 原始图像
                    kernel,  # 腐蚀核
                    iterations=3) # 迭代次数
# print(erosion) # 发现背景板的黑色 灰度其实不是0 而是21

erosion[erosion==0] = 21
# erosion == 0：
# 生成一个与 erosion 同尺寸的布尔掩膜（mask），标记所有值为0的像素位置为 True，其他为 False。
#
# python
# # 示例：
# erosion = np.array([[10, 0, 5],
#                    [0, 3, 0]])
# mask = (erosion == 0)
# # mask = [[False, True, False],
# #         [True, False, True]]
# erosion[mask] = 21：
# 将 erosion 中所有被 mask 标记为 True 的位置（即值为0的像素）赋值为21。
#
# 2. 执行步骤
# 遍历数组：逐像素检查是否等于0。
# 条件匹配：若像素值等于0，则将其替换为21。
# 返回结果：修改后的数组（原数组被直接修改，无副本）。

# 如果你的需求是：只有当 erosion 二维数组的某个子列表（行或列）所有元素都为0时，才将该子列表的所有元素替换为21，否则保持不变。
# 没有特别简便的方法  老老实实一步步写

cv2.imshow("erosion", erosion)

cv2.waitKey()
cv2.destroyAllWindows()













