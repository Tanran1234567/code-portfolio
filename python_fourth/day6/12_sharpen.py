'''
图像的锐化
'''
import cv2
import numpy as np

img = cv2.imread('../data/lena.jpg')
cv2.imshow('img',img)

kernel = np.array([[-1,-1,-1],
                   [-1,9,-1],
                   [-1,-1,-1]],dtype='float32')

res = cv2.filter2D(img,-1,kernel)
cv2.imshow('res',res)








cv2.waitKey()
cv2.destroyAllWindows()