from ultralytics import YOLO
import torch.multiprocessing as mp

def main():
    # 加载模型
    model = YOLO("yolov8n.yaml")  #构建一个空模型(从来没有被训练过)
    model = YOLO("yolov8n.pt")  # 在网上下载使用COCO8数据集预训练好的模型
    # model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # 从YAML文件构建模型并迁移预训练权重

    # 训练模型
    results = model.train(data=r"E:\大二暑期达内python学习\python教学听课笔记——深度学习阶段\python_fourth\RSP\data.yaml", epochs=10, imgsz=640)

if __name__ == '__main__':
    mp.freeze_support()
    main()
