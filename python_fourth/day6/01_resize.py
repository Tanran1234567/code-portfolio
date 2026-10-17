'''
图像缩放
'''
import cv2
img = cv2.imread('../data/lena.jpg')
cv2.imshow('img',img)

h,w = img.shape[:2]

# 缩小
dst_h = h // 2
dst_w = w // 2
dst_size = (dst_w,dst_h)
reduce = cv2.resize(img,dst_size)
cv2.imshow('reduce',reduce)

# 放大
dst_size = (w*2,h*2)
nearest = cv2.resize(img,dst_size,interpolation=cv2.INTER_NEAREST)
cv2.imshow('nearest',nearest)

linear = cv2.resize(img,dst_size,interpolation=cv2.INTER_LINEAR)
cv2.imshow('linear',linear)






cv2.waitKey()
cv2.destroyAllWindows()























