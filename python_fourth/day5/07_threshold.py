'''
二值化和反二值化
'''
import cv2
img = cv2.imread('../data/CPU3.png')
cv2.imshow('img',img)

# 二值化一般都是把图片转成灰度图像进行处理  而不是彩色图像
# 灰度化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)

# 二值化
# t,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# 白色有点多 把阈值调高一点 让更多的白点变成黑色
t,binary = cv2.threshold(gray,155,255,cv2.THRESH_BINARY)
cv2.imshow('binary',binary)

# 参数	              含义
# gray	              输入的灰度图像（单通道，dtype=np.uint8）。
# 127	              阈值（Threshold），用于像素分类的临界值。
# 255	              最大值，当像素值超过阈值时赋予的新值（仅对某些阈值类型有效）。
# cv2.THRESH_BINARY	  阈值化类型，此处为最常用的二值化模式。

# 返回值
# 返回值	    含义
# t	        实际使用的阈值（可能与输入的 thresh 相同，或由算法自动计算）。
# binary	输出的二值图像（黑白图像，像素值为 0 或 maxval）。



cv2.waitKey()
cv2.destroyAllWindows()








