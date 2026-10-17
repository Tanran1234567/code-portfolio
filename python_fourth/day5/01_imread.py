import cv2
img = cv2.imread('../data/Linus.png')
print(img.shape) # h,w,c (height,width,channels)

# 显示图像
cv2.imshow('img',img)
# cv2.imshow('img1',img)
# 窗口名称不能一致，不然后一张图片(窗口)会覆盖前面一个
# 可以在两个窗口显示同一张图片


# 主动进入阻塞等待状态，等待用户按下某个按键，退出阻塞
cv2.waitKey() # 其中有个参数 delay (单位:ms) 可以设置为5s后自动关闭
              # cv2.waitKey(5000)
# while True:
#     key = cv2.waitKey()
#     if key == 27:
#         break
# 显示每个按键对应的返回值  这样可以设置 只有按下特定按键  才能结束循环

# 保存图像
cv2.imwrite('./new_linus.png',img)

# 关闭所有窗口
cv2.destroyAllWindows()