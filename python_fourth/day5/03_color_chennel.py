# 彩色图像的通道操作
import cv2
img = cv2.imread('../data/opencv2.png')
print(img.shape)
cv2.imshow('img',img)

# 分量法，取出蓝色通道并显示
b = img[:,:,0]  # BGR 中取出蓝色通道
cv2.imshow('b',b)

# 在原始图像中，将蓝色通道赋值为0
img[:,:,0] = 0
cv2.imshow('b0',img)

# 在原始图像中，在将蓝色通道赋值为0的基础上，再将绿色通道赋值为0
img[:,:,1] = 0
cv2.imshow('b0g0',img)





cv2.waitKey()
cv2.destroyAllWindows()