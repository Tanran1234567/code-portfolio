'''
开运算
'''
import cv2
import numpy as np

img = cv2.imread('../data/5.png',0)
cv2.imshow('img',img)

kernel = np.ones(shape=(3,3),dtype=np.uint8)
res = cv2.morphologyEx(img,
                       cv2.MORPH_OPEN,
                       kernel,
                       iterations=3)

cv2.imshow('res',res)

cv2.waitKey()
cv2.destroyAllWindows()

# 开运算（MORPH_OPEN）的迭代含义
# 开运算的定义是：先腐蚀（Erosion），后膨胀（Dilation）。

# 当 iterations=3 时，完整的操作流程是：
# 先连续腐蚀 3 次（对原始图像迭代腐蚀 3 轮）。
# 再对腐蚀后的结果连续膨胀 3 次（用同样的核迭代膨胀 3 轮）。
