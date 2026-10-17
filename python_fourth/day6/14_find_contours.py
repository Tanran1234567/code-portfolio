'''
查找轮廓
'''
import cv2
import numpy as np

img = cv2.imread('../data/3.png')
# 其实当前图像是一个彩色图像  可以查看该图片的属性  找到详细信息 看到位深度是24
cv2.imshow('img',img)

# 灰度化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)

# 二值化
t,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imshow('binary',binary)

# 查找轮廓
cnts,hie = cv2.findContours(binary,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)
print(cnts)
print(len(cnts))  # 在该图片中查找出了几个轮廓
print(hie)  # 返回轮廓层次 图像的拓扑信息  轮廓的位置分析

# 绘制轮廓
# 绘制可在原彩色图像上绘制(不然也绘制不出颜色来)
# 查找轮廓需要再二值图像上进行处理
# img_copy = img.copy()  # 复制一份
# 下面这个函数  实际上原始图像img和res上都画出了轮廓
res = cv2.drawContours(img,
                       cnts,
                       -1,
                       (0,0,255),
                       2)
cv2.imshow('res',res)
# cv2.imshow('res11',img)



# 如何单独绘制轮廓呢?
# 就是不绘制在原图上!
# 方法：拿到和原图像一模一样大小的纯黑图像
mask = np.zeros_like(img)
res22 = cv2.drawContours(mask,
                       cnts,
                       -1,
                       (0,0,255),
                       2)

cv2.imshow('res22',res22)




cv2.waitKey()
cv2.destroyAllWindows()

# cv2.findContours() 在处理二值图像时，默认只能检测白色（高亮度）区域的轮廓，
# 而不会直接检测黑色（低亮度）填充区域的轮廓。但通过调整参数或预处理，可以间接实现这一目标