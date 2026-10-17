'''
灰度图像的直方图均衡化
'''
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('../data/sunrise.jpg',0)
cv2.imshow('img',img)

# 直方图均衡化
res = cv2.equalizeHist(img)
cv2.imshow('res',res)

plt.figure('Hist',figsize=(12,7))
plt.subplot(2,1,1)
plt.hist(img.ravel(),bins=256,range=(0,255))
# plt.hist(img.ravel(),bins=256)

plt.subplot(2,1,2)
plt.hist(res.ravel(),bins=256,range=(0,255))
plt.show()




cv2.waitKey()
cv2.destroyAllWindows()