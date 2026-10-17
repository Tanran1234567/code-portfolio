'''
度盘区域的瑕疵检测
'''
import cv2
import numpy as np

img = cv2.imread('../data/CPU3.png')
cv2.imshow('img',img)
#灰度化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
#二值化
t,binary = cv2.threshold(gray,158,255,cv2.THRESH_BINARY)
cv2.imshow('binary',binary)
#在二值化的图像上查找轮廓（度盘区域）
cnts,hie = cv2.findContours(binary,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)
# print(len(cnts))
#构建一张和二值化shape一模一样的纯黑色的图像mask
mask = np.zeros_like(binary)
# cv2.imshow('mask',mask)
#将查找到的度盘的轮廓画在mask上，线条粗细-1为实心化填充
img_fill = cv2.drawContours(mask,
                            cnts,
                            -1,
                            255,
                            -1)
cv2.imshow('img_fill',img_fill)
#让二值化的图像(带瑕疵)和mask(不带瑕疵)做减法，减出瑕疵区域
img_sub = cv2.subtract(img_fill,binary)
cv2.imshow('img_sub',img_sub)
#对瑕疵进行闭运算
kernel = np.ones(shape=(3,3),dtype=np.uint8)
img_close = cv2.morphologyEx(img_sub,cv2.MORPH_CLOSE,kernel)
cv2.imshow('close',img_close)
#对瑕疵进行查找轮廓
cnts,hie = cv2.findContours(img_close,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)
# print(len(cnts))
#对瑕疵进行按照面积倒序排序
if len(cnts) > 1:
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
#计算面积并与行业标准进行比较：标准为8
max_area = cv2.contourArea(cnts[0])
if max_area > 8:
    #拟合最大的瑕疵的外接圆形
    center,radius = cv2.minEnclosingCircle(cnts[0])
    center = (int(center[0]),int(center[1]))
    radius = int(radius)
    #将外接圆形画在原始图像上
    res = cv2.circle(img,center,radius,(0,0,255),2)
    cv2.imshow('res',res)
    print(f'有瑕疵，瑕疵面积为：{max_area}')
    # max_area的单位是像素的平方
elif len(cnts) > 10:
    print('虽然没有大瑕疵，但是小瑕疵的数量太多了，也不合格')
    res = cv2.drawContours(img,cnts,-1,(0,0,255),2)
    cv2.imshow('res',res)
else:
    print('好图像！！！！！！')


cv2.waitKey()
cv2.destroyAllWindows()