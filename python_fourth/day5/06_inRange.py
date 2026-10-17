'''
提取指定的颜色
'''

# 肯定是用hsv来提取
# 因为h这一个通道就确定了颜色
# 而用BGR 是三个通道综合在一起形成一种颜色  不便提取

import cv2
import numpy as np
img = cv2.imread("../data/opencv2.png")
cv2.imshow('img', img)

# BGR--->HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# H表示颜色的相位角(hue) ，取值范围是0---360；
# S表示颜色的饱和度(saturation) ，范围从0到1，
# 它表示成所选颜色的纯度和该颜色最大的纯度之间的比率；
# V表示色彩的明亮程度(value) ，范围从0到1。
# 但是openCV中把三个通道的取值范围都改了
# 改成了下述三个

# 110-130 提取蓝色
# 50-255 饱和度不要太低 蓝色要浓一点
# 50-255 亮度不要太低
min_val = np.array([110,50,50])
max_val = np.array([130,255,255])
# 设计掩码
mask = cv2.inRange(hsv,min_val,max_val)  # 图像中白色为蓝色 黑色为不想要的东西

# 作逻辑与运算(规则与C++中的&&略有不同，详情请看8月15日 goodnotes)
# 1 and 8 --> 8
# 0 and 1 --> 0
# 9 or 8 --> 9
res = cv2.bitwise_and(img,img,mask=mask)
# 若没有mask 得来的就是原图img 因为自己和自己做逻辑与操作  结果还是自己

cv2.imshow('mask',mask)
cv2.imshow('res',res)

# 2. res = cv2.bitwise_and(img, img, mask=mask)

# 功能：对原始 BGR 图像 img 应用掩码 mask，保留目标区域的彩色像素，背景区域置黑。
# 参数解析：
# 前两个 img 表示对同一图像做按位与操作（实际无变化）。
# mask=mask：用 mask 控制哪些像素被保留（mask=255 的像素保留，mask=0 的像素置 0）。
# 输出：res 是处理后的图像，目标区域保留原色，背景为黑色。

# 为什么能混用 HSV 的 mask 和 BGR 的 img？
# 掩码的本质：
# mask 是一个二值图像（单通道，0/255），与颜色空间无关。它仅标记哪些像素需要保留，不涉及颜色转换。
# 无论 hsv 还是 img，它们的 像素位置是严格对齐的，因此 mask 可以通用。
# bitwise_and 的工作方式：
# 对 img 的 每个通道（B、G、R）分别与 mask 做按位与操作。
# 如果 mask 某像素为 0，对应位置的 B/G/R 值均被置 0（黑色）；若为 255，则保留原值。
# 因此，mask 的生成方式（HSV）不影响它对 BGR 图像的操作。


cv2.waitKey()
cv2.destroyAllWindows()
