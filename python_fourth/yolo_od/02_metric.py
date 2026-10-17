from ultralytics import YOLO

# 加载模型
# model = YOLO("yolo11n.pt") # 加载官方模型
model = YOLO("best.pt")  # 加载自定义模型

# 验证模型
metrics = model.val(data=r"E:\大二暑期达内python学习\python教学听课笔记——深度学习阶段\python_fourth\RSP\data.yaml") # 无需传入参数，数据集和设置已保存
print(metrics.box.map)  # mAP（mean Average Precision）50%-95%
print(metrics.box.map50)  # mAP50（IoU阈值为0.5时的mAP）
print(metrics.box.map75)  # mAP75（IoU阈值为0.75时的mAP）
print(metrics.box.maps)  # 包含每个类别50%-95% mAP的列表
