'''
图像的裁剪: 随机裁剪  中心裁剪
'''
import cv2
import numpy as np

def random_crop(img,cw,ch):
    # cw,ch 为裁剪的大小
    h,w = img.shape[:2]
    # 生成随机的起始坐标
    start_x = np.random.randint(0,w-cw)
    start_y = np.random.randint(0,h-ch)
    # 切片
    new_img = img[start_y:start_y+ch,start_x:start_x+cw]
    return new_img

def center_crop(img,cw,ch):
    # cw,ch 为裁剪的大小
    h, w = img.shape[:2]
    start_x = int((w-cw)/2)
    start_y = int((h-ch)/2)
    new_img = img[start_y:start_y + ch, start_x:start_x + cw, :]
    return new_img

if __name__ == '__main__':
    img = cv2.imread('../data/banana_1.png')
    cv2.imshow('img',img)

    # 随机裁剪
    random = random_crop(img,200,200)
    cv2.imshow('random',random)

    # 中心裁剪
    center = center_crop(img,200,200)
    cv2.imshow('center',center)


    cv2.waitKey()
    cv2.destroyAllWindows()