'''
图像的仿射变换
'''
import cv2
import numpy as np

# 平移
def translate(img,x,y):
    # img.shape 返回的是 h,w,c ,是一个元组，需要切片，拿出前两个
    h,w = img.shape[:2]
    # 平移矩阵
    M = np.float32([[1,0,x],
                    [0,1,y]])
    res = cv2.warpAffine(img,M,(w,h),borderValue=(255,255,255))
    # 平移出去之后空出去的那一部分显示什么颜色，默认为黑色，此处显示为黑色

    # res = cv2.warpAffine(img, M, (w+50, h+30))
    # 这里的dsize指的是显示这张图片的窗口的大小，而不是图片放大
    # 会让图片在更大的窗口平移，但图片只是位置改变，大小没有改变
    return res

def rotate(img,angle,center=None):
    # center=None设置一个缺省值

    h,w = img.shape[:2]
    # print(h,w)   # 300 300
    if center is None:
        # center未传参，默认按中心点进行旋转
        center = (w / 2, h / 2)

    # 旋转矩阵
    M = cv2.getRotationMatrix2D(center,angle,1.0)
    # scale 指的是图片的缩放比例

    # 改变旋转中心
    # 实际上
    # M[0][2] 是水平平移分量(基于x计算)
    # M[1][2] 是垂直平移分量(基于y计算)
    # 成功
    # M[0][2] += (np.sqrt(2)-1)*(w/2)
    # M[1][2] += (np.sqrt(2)-1)*(h/2)

    # M[0][2] += 100
    # M[1][2] += 100

    M[0][2] = np.sqrt(2)*150
    M[1][2] = 0

    # 失败
    # M[1][2] -= M[0][2]
    # M[0][2] = 0
    print(M)  # 打印矩阵

    nw = int(np.sqrt(2) * w)
    nh = int(np.sqrt(2) * h)
    res = cv2.warpAffine(img,M,(nw,nh))
    # res = cv2.warpAffine(img, M, (int(M[1][2]*2), int(M[1][2]*2)))
    return res

if __name__ == '__main__':
    img = cv2.imread('../data/lena.jpg')
    cv2.imshow('img',img)

    # 向右平移 50 px，向下平移 30 px
    translated = translate(img,50,30)
    cv2.imshow('translated',translated)

    # 旋转  正数角度为逆时针旋转  负数角度为顺时针旋转
    rotated = rotate(img,-45)
    cv2.imshow('rotated',rotated)

    cv2.waitKey()
    cv2.destroyAllWindows()


# 原始旋转中心：
# cv2.getRotationMatrix2D(center, angle, 1.0) 中的 center 参数本应是你指定的旋转中心点（通常是图像中心）
# 这个中心点的坐标是 (center[0], center[1])
# 改变旋转中心：
# M[0][2] 和 M[1][2] 是仿射变换矩阵中的平移分量
# 当你执行 M[0][2] += 100 和 M[1][2] += 100 时，实际上是在原始变换基础上额外添加了一个平移
# 这会导致旋转后的图像向右下方各移动100像素

# 实际效果：
# 这不是真正改变了旋转的中心点，而是相当于：
# a) 先按原中心点旋转图像
# b) 然后把结果图像向右下方平移
# 视觉上看起来像是旋转中心移动了，但实际上旋转中心没变，只是最终结果被平移了
# 真正的改变旋转中心：
# 如果你真的想改变旋转中心，应该在计算旋转矩阵时就传入新的中心点坐标

# 例如：new_center = (center[0]+100, center[1]+100)
# 然后 M = cv2.getRotationMatrix2D(new_center, angle, 1.0)







