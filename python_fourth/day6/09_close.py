'''
并运算
'''
import cv2
import numpy as np

img = cv2.imread('../data/9.png',0)
cv2.imshow('img',img)

# 上边缘会拉花
# kernel = np.ones(shape=(3,3),dtype=np.uint8)
kernel1 = np.ones(shape=(1,3),dtype=np.uint8)

# res = cv2.morphologyEx(img,
#                        cv2.MORPH_CLOSE,
#                        kernel,
#                        iterations=4)

# 核变小了，所以需要迭代的次数更多  否则效果达不到
res = cv2.morphologyEx(img,
                       cv2.MORPH_CLOSE,
                       kernel1,
                       iterations=12)

cv2.imshow('res',res)

cv2.waitKey()
cv2.destroyAllWindows()


