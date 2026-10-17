'''
使用opencv读取摄像头
'''
import cv2
cap = cv2.VideoCapture(0)  # 摄像头的对象

while cap.isOpened():
    ret,frame = cap.read()   # 捕获一帧图像
    # ret为bool值,捕获到返回True，否则为False
    # frame 为捕获到的一帧图像
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)  # ms
    if key == 27:
        break

cap.release()  # 释放摄像头对象
cv2.destroyAllWindows()