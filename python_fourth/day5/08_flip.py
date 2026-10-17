'''
镜像 (翻转)
'''
import cv2
img = cv2.imread('../data/lena.jpg')
cv2.imshow('img',img)

# 水平镜像
flip1 = cv2.flip(img,1)
cv2.imshow('flip1',flip1)

# 垂直镜像
flip0 = cv2.flip(img,0)
cv2.imshow('flip0',flip0)

# 垂直加水平 的镜像
flip01 = cv2.flip(img,-1)
cv2.imshow('flip01',flip01)






cv2.waitKey()
cv2.destroyAllWindows()










