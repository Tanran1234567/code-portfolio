'''
矩形轮廓的拟合
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
x,y,w,h = cv2.boundingRect(cnts[0])   # 取出第一个轮廓  该轮廓数据类型为三维数组
                                      #               整个轮廓集合为一个列表(元组)
                                      #               所以整体上，数据类型为列表(元组)套三维数组
points = [np.array([[[x,y]],
           [[x,y+h]],
           [[x+w,y+h]],
           [[x+w,y]]])]
# 到底是列表套三维数组，还是元组套三维数组 和版本有关
# points = (np.array([[[x,y]],
#            [[x,y+h]],
#            [[x+w,y+h]],
#            [[x+w,y]]]))

cv2.drawContours(img,points,0,(0,0,255),2)
cv2.imshow('res',img)

# openCV中有专门绘制矩形轮廓拟合的函数接口
cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow('res11',img_copy)










cv2.waitKey()
cv2.destroyAllWindows()