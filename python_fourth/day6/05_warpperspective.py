'''
图像的透视变换/投影映射
'''
import cv2
import numpy as np

img = cv2.imread('../data/pers.png')
rows, cols = img.shape[:2]
print(rows, cols)

# print(img[0][0])
# print(img)

pts1 = np.float32([[58, 2], [167, 9], [8, 196], [126, 196]])# 输入图像四个顶点坐标
pts2 = np.float32([[16, 2], [167, 8], [8, 196], [169, 196]])# 输出图像四个顶点坐标
pts3 = np.float32([[16, 2], [167, 8], [8, 196], [180, 196]])# 输出图像四个顶点坐标

# 几何特征：
# 上边 [50,10] → [180,20]（斜向右上方）。
# 下边 [30,150] → [160,160]（斜向右上方）。
# 左侧 [50,10] → [30,150]（斜向左下方）。
# 右侧 [180,20] → [160,160]（斜向左下方）。
# 通过透视变换后，这些斜边会被拉直为矩形的水平/垂直边。


# 生成透视变换矩阵
M = cv2.getPerspectiveTransform(pts1, # 输入图像四个顶点坐标
                                pts2) # 输出图像四个顶点坐标
print(M.shape)
# 执行透视变换，返回变换后的图像
dst = cv2.warpPerspective(img, # 原始图像
                          M, # 3*3的变换矩阵
                          (cols, rows)) # 输出图像大小


# 生成透视变换矩阵
M = cv2.getPerspectiveTransform(pts2, # 输入图像四个顶点坐标
                                pts1) # 输出图像四个顶点坐标
# 执行透视变换，返回变换后的图像
dst2 = cv2.warpPerspective(dst, # 原始图像
                          M, # 3*3的变换矩阵
                          (cols, rows)) # 输出图像大小

# 生成透视变换矩阵
M = cv2.getPerspectiveTransform(pts1, # 输入图像四个顶点坐标
                                pts3) # 输出图像四个顶点坐标
# 执行透视变换，返回变换后的图像
dst3 = cv2.warpPerspective(img, # 原始图像
                          M, # 3*3的变换矩阵
                          (cols, rows)) # 输出图像大小


cv2.imshow("img", img)
cv2.imshow("dst", dst)
cv2.imshow("dst2", dst2)
cv2.imshow("dst3", dst3)   # 梯形

cv2.waitKey()
cv2.destroyAllWindows()

