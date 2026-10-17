'''
多边形轮廓的拟合
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
adp1 = img.copy()
adp2 = img.copy()
# eps 值越小  拟合得越好
eps = 0.005 * cv2.arcLength(cnts[0],True)  # arclength 是求周长
eps2 = 0.01 * cv2.arcLength(cnts[0],True)
points = cv2.approxPolyDP(cnts[0],eps,True)
points2 = cv2.approxPolyDP(cnts[0],eps2,True)
print(points)

# 画逼近多边形
cv2.drawContours(adp1,[points],0,(0,0,255),2)
cv2.drawContours(adp2,[points2],0,(0,0,255),2)
cv2.imshow('adp1',adp1)
cv2.imshow('adp2',adp2)


cv2.waitKey()
cv2.destroyAllWindows()











cv2.waitKey()
cv2.destroyAllWindows()