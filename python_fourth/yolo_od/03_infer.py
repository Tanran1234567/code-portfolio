from ultralytics import YOLO

# 加载模型
model = YOLO("best.pt")  # 加载预训练的YOLO11n模型

# 对图像列表进行批量推理
results = model(["scissor_infer.jpg"])   # 返回一个Results对象列表

# 处理结果列表
for result in results:
    boxes = result.boxes   # 边界框输出的Boxes对象
    # masks = result.masks   # 分割掩码输出的Masks对象
    # keypoints = result.keypoints  # 姿态输出的Keypoints对象
    # probs = result.probs   # 分类输出的Probs对象
    # obb = result.obb   # 定向边界框输出的Oriented boxes对象
    result.show()   # 在屏幕上显示结果
    result.save( filename = "result.jpg")   # 将结果保存到磁盘