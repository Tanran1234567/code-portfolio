'''
形态学梯度
'''
import cv2
import numpy as np

img = cv2.imread('../data/6.png',0)
cv2.imshow('img',img)

# 做形态学梯度时 核不要弄太大 迭代次数也不能太大
# 否则会导致减出来的轮廓很大
# 因为膨胀得太大 腐蚀得太小
kernel1 = np.ones(shape=(2,2),dtype=np.uint8)

# res = cv2.morphologyEx(img,
#                        cv2.MORPH_CLOSE,
#                        kernel,
#                        iterations=4)

# 核变小了，所以需要迭代的次数更多  否则效果达不到
res = cv2.morphologyEx(img,
                       cv2.MORPH_GRADIENT,
                       kernel1,
                       iterations=2)

cv2.imshow('res',res)

cv2.waitKey()
cv2.destroyAllWindows()


