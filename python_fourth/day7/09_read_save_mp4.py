'''
捕获摄像头对象  并保存视频
'''
import cv2

# 捕获帧的对象
cap = cv2.VideoCapture(0)  # 摄像头对象

fourcc = cv2.VideoWriter.fourcc('I','4','2','0')
# 写入帧的对象
out = cv2.VideoWriter('output.avi',
                      fourcc,# 四字标记值
                      20, #fps
                      (640,480))# 分辨率

while cap.isOpened():
    ret,frame = cap.read()
    if ret == True:
        # 写入帧
        out.write(frame)
        cv2.imshow('frame',frame)

        key = cv2.waitKey(1)
        if key == 27 :
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()