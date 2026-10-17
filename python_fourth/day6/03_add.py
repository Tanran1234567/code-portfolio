'''
图像的相加
'''
import cv2

lena = cv2.imread('../data/lena.jpg',0)
lily = cv2.imread('../data/lily_square.png',0)
# 如果用lily.png 会发现两张图片不一样大 需要将其中一张图片进行resize
cv2.imshow('lena',lena)
cv2.imshow('lily',lily)

add = cv2.add(lena,lily)
cv2.imshow('add',add)
# 图像直接相加，会导致图像过亮、过白

# 加权求和：addWeighted
# 图像进行加权和计算时，要求src1和src2必须大小、类型相同
add_w = cv2.addWeighted(lena, 0.6, lily, 0.4, 0)  # 最后一个参数为亮度调节量
# 权重值不要设置太大，最好1以下 如果设置80和20 为两个权重值
# 那每个像素值加起来算出来都大于255，那就会得到一个全白的图像
cv2.imshow('add_w',add_w)


cv2.waitKey()
cv2.destroyAllWindows()