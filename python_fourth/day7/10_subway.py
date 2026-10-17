'''
检测铁轨
'''
import cv2
import numpy as np
img = cv2.imread('../data/subway.jpg')
cv2.imshow('img',img)

# 灰度化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)

# 高斯模糊
gau = cv2.GaussianBlur(gray,(11,11),0)
cv2.imshow('gau',gau)

kernel = np.ones(shape=(5,5),dtype=np.uint8)
close = cv2.morphologyEx(gau,cv2.MORPH_CLOSE,kernel)
# 边缘检测
canny = cv2.Canny(close,120,255)
cv2.imshow('canny',canny)


kernel = np.ones(shape=(7,7),dtype=np.uint8)
close1 = cv2.morphologyEx(canny,cv2.MORPH_CLOSE,kernel,iterations=2)
cv2.imshow('close1',close1)


# 二值化
t,binary = cv2.threshold(gray,190,255,cv2.THRESH_BINARY)
cv2.imshow('binary',binary)

cnts, hie = cv2.findContours(close1,
                             cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)

mask = np.zeros_like(binary)
# cv2.imshow('mask',mask)
#将查找到的度盘的轮廓画在mask上，线条粗细-1为实心化填 充
img_fill = cv2.drawContours(mask,
                            cnts,
                            -1,
                            255,
                            -1)
cv2.imshow('img_fill',img_fill)

# for cnt in cnts:
#     params = cv2.fitEllipse(cnt)
#     cv2.ellipse(img, params, color=(0, 0, 255), thickness=2)
# # print(params)
#
# # 画椭圆的接口更智能  不需要传整数  他自己会帮我们找到最接近的整数
# # 画椭圆
# cv2.ellipse(img,params,color=(0,0,255),thickness=2)
# cv2.imshow('res',img)





cv2.waitKey()
cv2.destroyAllWindows()