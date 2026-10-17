'''
使用opencv读取摄像头
'''
import cv2
cap = cv2.VideoCapture(r"C:\Users\zjl18\Desktop\python_fourth\day7\output.avi")  # 摄像头的对象
# t 提供视频的路径(写绝对路径)，比如C:\user……
# 同时可能需要 cap = cv2.VideoCapture(r'C:\user……')
# Linux系统不存在这个问题  因为Linux写路径都是C:/user……
# 而win系统写路径用的是反斜杠  比如写到\niuli……s时会将\n读成换行符
# 所以要么在每个'\'前再加一个'\'  要么就在最前面写r''  表示不转义  用最原始的字符串

while cap.isOpened():
    ret,frame = cap.read()   # 捕获一帧图像
    # ret为bool值,捕获到返回True，否则为False
    # frame 为捕获到的一帧图像
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))  # 调节视频画面的大小
    # frame = cv2.resize(frame, (2000, 1200))  # 调节视频画面的大小
    cv2.imshow('frame',frame)
    # frame = cv2.resize(frame,(640,480))  # 调节视频画面的大小   与cv2.imshow('frame',frame)顺序不要搞反了
    key = cv2.waitKey(1)  # ms
    # key = cv2.waitKey(5)  # 视频播放会变慢
    if key == 27:
        # 按Esc键退出
        break

cap.release()  # 释放摄像头对象
cv2.destroyAllWindows()