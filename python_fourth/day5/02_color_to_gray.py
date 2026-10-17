# 将彩色图像装为灰度图像

import cv2
img = cv2.imread('../data/lena.jpg',1)
# img1 = cv2.imread('../data/lena.jpg',0)

# 一般还是先提取彩色图像  因为后续opencv 可以实现彩色到灰度的转换
# 但是却不能实现灰度到彩色的转换 而你如果最开始提取的图片 img 就是灰度
# 那就会导致以后在img上提取不到彩色信息了

# 转换色彩空间  BGR --> GRAY (opencv这样规定的图片格式 不是RGB，而是BGR)
# 使用cvtColor进行颜色空间变化，COLOR_BGR2GRAY表示BGR to GRAY
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 彩色图像灰度化
cv2.imshow("Gray", img_gray)




# 有 flag 参数 默认为1 即彩色图像  0 就是灰度图像
cv2.imshow('img',img)
# cv2.imshow('img1',img1)


cv2.waitKey()