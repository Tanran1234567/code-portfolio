import sys
import cv2
from ultralytics import YOLO
import torch
from PyQt5.QtGui import QIcon,QImage
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QWidget, QStyle,QFileDialog, QLabel, QHBoxLayout, QStatusBar, QAction, QToolBar


# 定义主窗口类，继承自QMainWindow
class YOLOv8GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # 调用父类构造函数
        self.setWindowTitle("目标检测系统(达内人工智能教研部)") # 设置窗口标题
        self.setGeometry(100, 100, 1280, 720) # 设置窗口位置和大小

        # 初始化模型
        self.model = YOLO('./yolo11n.pt')  # 初始化YOLOv8模型（加载预训练权重）
        self.current_mode = None  # 当前模式：image/video/camera
        self.cap = None  # 视频/摄像头捕获对象
        self.timer = QTimer() # 定时器用于实时更新帧

        # 初始化用户界面组件
        self.initUI()

    def initUI(self):
        # 创建菜单栏
        menubar = self.menuBar()

        # 创建文件菜单
        file_menu = menubar.addMenu('文件')
        file_menu.addAction("退出", self.close)

        # 创建编辑菜单（目前只有撤销功能占位）
        edit_menu = menubar.addMenu('编辑')
        edit_menu.addAction("撤销", self.undo)


        # 创建功能菜单
        action_menu = menubar.addMenu('功能')
        # 定义功能按钮动作
        self.open_image_action = QAction('上传图片', self)
        self.open_video_action = QAction('上传视频', self)
        self.open_camera_action = QAction('实时摄像头', self)
        self.stop_action = QAction('暂停(摄像头/视频)', self)

        # 将功能菜单项添加到菜单栏
        action_menu.addActions([self.open_image_action, self.open_video_action,
                                self.open_camera_action, self.stop_action])

        # 创建工具栏
        toolbar = QToolBar(self)
        self.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)
        toolbar.setStyleSheet("QToolBar {background-color: lightgray;}")# 设置工具栏背景色

        # 关于
        about_menu = menubar.addMenu('关于')

        # 帮助
        help_menu = menubar.addMenu('帮助')

        # 设置全局样式表（尝试调整菜单栏字体大小）
        self.setStyleSheet("""
                    QMenuBar {
                        font-size: 18px; /* 尝试设置菜单栏字体大小，但可能不起作用 */
                    }
                    QMenu {
                        font-size: 18px; /* 设置菜单项的字体大小 */
                    }}""")

        # 创建工具栏按钮
        self.btn_image = QtWidgets.QPushButton()
        self.btn_video = QtWidgets.QPushButton()
        self.btn_camera = QtWidgets.QPushButton()
        self.btn_close = QtWidgets.QPushButton()
        self.btn_tedu = QtWidgets.QPushButton()

        # 使用系统标准图标
        style = self.style()
        self.icon_image = style.standardIcon(QStyle.SP_DialogOpenButton)
        self.icon_video = style.standardIcon(QStyle.SP_MediaPlay)
        self.icon_camera = style.standardIcon(QStyle.SP_ComputerIcon)
        self.icon_close = style.standardIcon(QStyle.SP_MediaPause)
        self.icon_tedu = QIcon('./tedu_logo.png') # 自定义达内LOGO图标

        # 设置按钮图标
        self.btn_image.setIcon(self.icon_image)
        self.btn_video.setIcon(self.icon_video)
        self.btn_camera.setIcon(self.icon_camera)
        self.btn_close.setIcon(self.icon_close)
        self.btn_tedu.setIcon(self.icon_tedu)

        # 设置按钮属性,统一设置按钮样式
        for btn in [self.btn_image, self.btn_video, self.btn_camera, self.btn_close]:
            btn.setIconSize(QtCore.QSize(78, 48))  # 调整图标尺寸
            btn.setFixedSize(78, 48)  # 调整按钮尺寸
            btn.setStyleSheet("""
                    QPushButton {
                font-size: 24px;
                background-color: transparent;  /* 按钮背景透明 */
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #45a049; /* 悬停时背景色 */
            }
                """)

        # 单独设置达内LOGO按钮样式
        self.btn_tedu.setIconSize(QtCore.QSize(140,60))

        self.btn_tedu.setStyleSheet("""
                            QPushButton {
                        font-size: 24px;
                        background-color: transparent;  /* 按钮背景透明 */
                        color: white;
                        border: none;
                        border-radius: 5px;
                        padding: 8px 16px;
                    }""")


        # 设置按钮的图标大小和文本位置（可选）
        self.btn_image.setToolTip("上传图片")  # 设置工具提示（鼠标悬停时显示）

        self.btn_video.setToolTip("上传视频")  # 设置工具提示（鼠标悬停时显示）

        self.btn_camera.setToolTip("实时摄像头")  # 设置工具提示（鼠标悬停时显示）

        self.btn_close.setToolTip("暂停(视频/摄像头)")  # 设置工具提示（鼠标悬停时显示）

        # 将按钮添加到工具栏
        toolbar.addWidget(self.btn_image)
        toolbar.addWidget(self.create_spacer(15))  # 添加20像素的间隔
        toolbar.addWidget(self.btn_video)
        toolbar.addWidget(self.create_spacer(15))  # 添加20像素的间隔
        toolbar.addWidget(self.btn_camera)
        toolbar.addWidget(self.create_spacer(15))  # 添加20像素的间隔
        toolbar.addWidget(self.btn_close)
        # 添加弹性空间，将 tedu 按钮推到最右侧
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        # 添加 tedu 按钮
        toolbar.addWidget(self.btn_tedu)

        # 主显示区域
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # 创建图像显示标签
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.label, 3)

        # 状态栏
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # 信号连接
        self.timer.timeout.connect(self.update_frame)# 定时器触发帧更新
        self.btn_image.clicked.connect(self.open_image)# 图片按钮点击事件
        self.btn_video.clicked.connect(self.open_video)# 视频按钮点击事件
        self.btn_camera.clicked.connect(self.open_camera)# 摄像头按钮点击事件
        self.btn_close.clicked.connect(self.stop_current_mode)# 暂停按钮点击事件

    def create_spacer(self, width):
        """创建一个固定宽度的间隔"""
        spacer = QtWidgets.QWidget()
        spacer.setFixedWidth(width)
        return spacer

    def open_image(self):
        self.stop_current_mode() # 先停止当前模式
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.png *.jpeg)")
        if file_path:
            self.current_mode = "image"
            img = cv2.imread(file_path)
            self.process_image(img) # 处理并显示图片

    def open_video(self):
        self.stop_current_mode()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov)")
        if file_path:
            self.current_mode = "video"
            self.cap = cv2.VideoCapture(file_path) # 创建视频捕获对象
            self.timer.start(5)  # 5ms更新一帧

    def open_camera(self):
        self.stop_current_mode()
        self.current_mode = "camera"
        self.cap = cv2.VideoCapture(0)  # 0表示默认摄像头
        self.timer.start(5)  # 5ms更新一帧

    def process_image(self, img):
        # 使用YOLO模型进行推理
        results = self.model(img)
        annotated_frame = results[0].plot()  # 获取带标注的帧

        # 转换颜色空间并显示
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # 在标签上显示处理后的图像（保持比例缩放）
        self.label.setPixmap(QtGui.QPixmap.fromImage(q_img).scaled(
            self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()# 读取下一帧
            if ret:
                self.process_image(frame)# 处理并显示帧

    def stop_current_mode(self):
        """停止当前模式（释放资源）"""
        if self.cap:
            self.cap.release()# 释放视频捕获对象
        if self.timer.isActive():
            self.timer.stop()# 停止定时器
        self.current_mode = None

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        self.stop_current_mode()# 确保关闭时释放资源
        super().closeEvent(event)# 调用父类关闭方法

    def undo(self):
        """撤销操作（占位函数）"""
        print("Undo action triggered.")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv) # 创建Qt应用实例  sys.argv 是一个用于获取命令行参数的列表。当您通过命令行运行脚本时，所有传递给脚本的参数都会被存储在 sys.argv 中 python script.py arg1 arg2
    window = YOLOv8GUI()# 创建主窗口实例
    window.show()# 显示窗口
    sys.exit(app.exec_())# 进入Qt主循环
    #app.exec_()
    #是PyQt / PySide中启动事件循环的方法，它会阻塞直到应用程序退出。exec_()
    #返回一个退出代码，通常0表示正常退出，非0表示错误。然后sys.exit()
    #会根据这个退出代码结束程序。