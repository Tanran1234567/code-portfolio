'''
图像校正(透视变换)
'''
import cv2
import numpy as np

img = cv2.imread('../data/paper.jpg')
cv2.imshow('img',img)

# 先找纸的轮廓
# 灰度化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
# print(gray)

# 发现效果不好
# # 将纸变为白色  背景变为黑色
# t,binary = cv2.threshold(gray,190,255,cv2.THRESH_BINARY)
# cv2.imshow('binary',binary)

# kernel = np.array([[-1,-1,-1],
#                    [-1,9,-1],
#                    [-1,-1,-1]],dtype='float32')
#
# res = cv2.filter2D(gray,-1,kernel)
# cv2.imshow('res',res)

# 高斯滤波 sigmaX=0 不会退化为均值滤波，因为 OpenCV 会自动计算 σ。
# 如果想使用均值滤波，应该直接调用 cv2.blur。

# 高斯滤波和均值滤波的区别：
# 滤波类型	  权重分布	         边缘保留	             用场景
# 高斯滤波	  中心权重高，四周低	 较好（平滑但保留边缘）	 去噪+保边缘
# 均值滤波	  均匀权重	         较差（边缘变模糊）	     快速平滑

# 高斯模糊
gau = cv2.GaussianBlur(gray,(5,5),0)
cv2.imshow('gau',gau)

# 在图像处理中，开运算（Opening）通常用于消除小物体或噪声，
# 同时保持大物体的形状基本不变。
# 如果原本连通的区域在经过开运算后仍然保持连接，可能有以下原因：

# (1) 结构元素（Kernel）太小
# 问题：
# 如果 kernel 的尺寸 小于两个物体之间的连接宽度，腐蚀阶段无法完全切断连接。
# 后续膨胀会恢复部分被腐蚀的区域，导致物体仍然连通。

# (2) 物体之间的连接部分太强
# 问题：
# 如果两个物体的连接区域 灰度值较高（对比度强），腐蚀阶段难以完全消除它们。
# 例如，医学图像中粘连的细胞或工业零件中的强连接边缘。

# (3) 图像预处理不足
# 问题：
# 如果原始图像 噪声较多 或 对比度不均匀，开运算可能无法有效分离目标。
# 可能需要先进行 高斯模糊 或 自适应阈值处理 来优化输入。



# 在图像处理中，闭运算（Closing）通常用于填充小孔洞或连接邻近的边缘，
# 但有时它反而会导致原本连通的区域断开。这种现象看似矛盾，
# 但根本原因与结构元素（Kernel）的选择和图像的具体特征密切相关。
# 原因 2：图像对比度不均匀
# 如果两个连通区域的边缘灰度值差异较大，膨胀可能优先填充高对比度区域，
# 而忽略低对比度的连接处，导致腐蚀后无法恢复原始连接。

# 闭运算
kernel = np.ones(shape=(3,3),dtype=np.uint8)
# kernel = np.ones(shape=(5,5),dtype=np.uint8)
# 可以一步到位 将多层边缘通过先膨胀  粘连在一起 再一起被腐蚀到正常大小
close = cv2.morphologyEx(gau,cv2.MORPH_CLOSE,kernel)
open = cv2.morphologyEx(gau,cv2.MORPH_OPEN,kernel,iterations=3)
# 当然 通过调节参数  是可以直接通过高斯模糊+canny一步到位 ，只得到我们需要的
# 那一个轮廓  但是现在故意把事情搞复杂一点  看后续怎么补救
# 边沿检测
cv2.imshow('open',open)
cv2.imshow('close',close)
canny = cv2.Canny(close, 30, 120)
canny1 = cv2.Canny(close, 80, 250)
canny2 = cv2.Canny(close, 50, 250)
cv2.imshow('canny', canny)

# open1 = cv2.morphologyEx(canny,cv2.MORPH_OPEN,kernel)
#  cv2.imshow('open1', open1)
#  cv2.imshow('canny1', canny1)
# cv2.imshow('canny2', canny2)

# 在 OpenCV 中，cv2.findContours() 不会因为轮廓不封闭而直接报错，但它的行为会受到轮廓闭合性的影响。以下是详细分析：
# 1. 函数的基本行为
# cv2.findContours() 的输入是一个二值图像（黑白图，通常由 cv2.Canny() 或阈值处理得到），它会检测所有白色像素组成的轮廓，无论是否封闭：
# 开放轮廓：如果边缘未闭合（如一条直线或曲线），仍会被检测为一个轮廓。
# 封闭轮廓：如圆形、矩形等闭合形状，会被识别为闭合轮廓。


# 查找轮廓
cnts, hie = cv2.findContours(canny,
                             cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
# print(len(cnts))
# 目标轮廓：面积最大的四边形  cv2.contourArea()
# 按照面积进行排序
if len(cnts) > 1:
    cnts = sorted(cnts, key=cv2.contourArea,reverse=True)

#找到面积最大的四边形(逼近多边形)
ret = []
for cnt in cnts:
    eps = 0.02 * cv2.arcLength(cnt,True)
    points = cv2.approxPolyDP(cnt,eps,True)
    if len(points) == 4:
        # 默认矩形  轮廓要闭合
        ret.append(points)
        break

# print(ret)
print(ret[0])
img_point = img.copy()
for peak in ret[0]:
    center = tuple(peak[0])
    cv2.circle(img_point,center,10,(0,0,255),2)
cv2.imshow('img_point',img_point)

# 变换之前的顶点坐标
src = ret[0].reshape(4,2).astype('float32')
print(src)    # 观察坐标确定左上角 左下角 右上角 右下角
# 透视变换  原图和新图 要一一对应  所以坐标顺序不能给错

# 变换之后的顶点坐标

# 求出纸的宽度和高度
h = int(np.sqrt((src[0][0]-src[1][0])**2+(src[0][1]-src[1][1])**2))
w = int(np.sqrt((src[0][0]-src[3][0])**2+(src[0][1]-src[3][1])**2))
print(h,w)

# numpy 计算欧氏距离
# 第二种方法：用np接口计算纸的宽度和高度
h1 = int(np.linalg.norm(src[0]-src[1]))
w1 = int(np.linalg.norm(src[0]-src[3]))
print(h1,w1)

dst = np.array([[0,0],[0,h],[w,h],[w,0]],dtype='float32')  # 注意先左上角，再左下角…… 顺序要与src对应

# 构建透视变换的矩阵
M = cv2.getPerspectiveTransform(src,dst)
# 实现透视变换
res = cv2.warpPerspective(img,M,(w,h))
cv2.imshow('res',res)


cv2.waitKey()
cv2.destroyAllWindows()