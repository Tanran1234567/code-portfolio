'''
椭圆形轮廓的拟合
'''
import cv2
import numpy as np

img = cv2.imread('../data/cloud.png')
img_copy = img.copy()
cv2.imshow('img',img)

# 灰度化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)

# 二值化
t,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imshow('binary',binary)

# 查找轮廓
cnts,hie = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

# 拟合轮廓
params = cv2.fitEllipse(cnts[0])
print(params)

# 画椭圆的接口更智能  不需要传整数  他自己会帮我们找到最接近的整数
# 画椭圆
cv2.ellipse(img,params,color=(0,0,255),thickness=2)
cv2.imshow('res',img)


cv2.waitKey()
cv2.destroyAllWindows()











cv2.waitKey()
cv2.destroyAllWindows()