'''
圆形轮廓的拟合
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
center,radius = cv2.minEnclosingCircle(cnts[0])
print(center,radius)   # (204.5, 91.5)

# 将圆心和半径转为整数
center = (int(center[0]),int(center[1]))
radius = int(radius)

# 画圆
cv2.circle(img,center,radius,(0,0,255),2)
cv2.imshow('res',img)


cv2.waitKey()
cv2.destroyAllWindows()











cv2.waitKey()
cv2.destroyAllWindows()