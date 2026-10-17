'''
图形的模糊
'''
import cv2
import numpy as np
img = cv2.imread('../data/salt.jpg')
cv2.imshow('img',img)

# 均值滤波
mean = cv2.blur(img,(5,5))
cv2.imshow('mean',mean)

# 高斯滤波
# 3.0 用来设置标准差  均值不需要设置  一般为0
gaussian = cv2.GaussianBlur(img,(5,5),3.0)
cv2.imshow('gaussian',gaussian)

# 中值滤波
# median = cv2.medianBlur(img,3)
# 图像仍会存在白点，因为那个白点太大了，3*3的范围内全是255，所以求中位数还是255，所以白点无法去除
median = cv2.medianBlur(img,5)
cv2.imshow('median',median)
cv2.imwrite('median.jpg',median)

# 自定义卷积核，实现二维卷积
# 自己写的均值滤波
kernel = np.ones(shape=(5,5),dtype='float32')/25.0
res = cv2.filter2D(img,-1,kernel)
cv2.imshow('res',res)

# -1 的作用：
# 多数情况下，我们希望输出图像的数值范围和类型与输入一致（例如输入是uint8，输出也是uint8）。

# print(res)


cv2.waitKey()
cv2.destroyAllWindows()