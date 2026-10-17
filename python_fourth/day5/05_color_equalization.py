'''
彩色图像的直方图均衡化
均衡亮度通道之后 就会变得更亮
'''
import cv2

img = cv2.imread('../data/sunrise.jpg')
cv2.imshow('img',img)

# BGR ----> HSV
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# 将均衡之后的亮度重新赋值给亮度数组
hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])

# 将hsv
res = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

# 因为cv2.imread()读取图片时，都会把图片当作BGR进行提取
# cv2.imshow()展示图片时，也是如此，所以hsv要转回BGR
# 否则hsv的三通道数据会被解析为BGR的参数，最终显示出的图片会很奇怪
# 实际上就是张冠李戴了

cv2.imshow('real_res',res)
cv2.imshow('res',hsv)

cv2.waitKey()
cv2.destroyAllWindows()