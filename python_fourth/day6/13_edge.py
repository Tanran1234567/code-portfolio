'''
边沿(缘)检测
'''
import cv2

img = cv2.imread('../data/lily.png',0)
cv2.imshow('img',img)

# Sobel
# dx = 1 表示水平方向要进行边缘检测  0 表示不检测  dy 同理
# cv2.CV_64F 更精确   用-1的话  则图像数据为uint8  只能取 0-255
# 但Sobel核算子中有负数  这样算出的像素值有可能有负数  此时会报错
sobel = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)
cv2.imshow('sobel',sobel)

# Laplacian滤波：对细节反映更明显
laplacian = cv2.Laplacian(img, cv2.CV_64F)
cv2.imshow('Laplacian', laplacian)

# Canny边沿提取
canny = cv2.Canny(img,
                 50, # 滞后阈值
                 240) # 模糊度
cv2.imshow('Canny', canny)

blurrd11 = cv2.GaussianBlur(canny,(5,5),1.5)
cv2.imshow('blurred11',blurrd11)
# 一般高低阈值不知道怎么设置  可以设置为3倍
# 边缘不连续 怎么解决
# 可以在边沿提取之后，再用高斯模糊


# 高斯模糊处理(图像的平滑)
# 由于平滑处理  会淘汰更多的假边缘(像素值突变不够厉害，边缘不够锐利)
blurrd = cv2.GaussianBlur(img,(5,5),1.5)
# Canny边沿提取
canny11 = cv2.Canny(blurrd,
                 50, # 滞后阈值
                 240) # 模糊度
cv2.imshow('Canny11', canny11)


cv2.waitKey()
cv2.destroyAllWindows()

