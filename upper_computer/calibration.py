# 模块功能: 标定
################################################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMessageBox,QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import time
# 以下是自定义模块
import Serial
import imgProcess

updateData = False
dataNum = None

#
# 标定结果全局变量
a_pressure = None
b_pressure = None
a_concentration = None
b_concentration = None

def obtainCoordinates(x,type,serial):
    serial.get_data()
    # 获取当前时间
    # current_time = time.time()
    # timeout = 30  # 超时时间
    # print(type)
    # 等待请求发送后的最新的标定结果
    # while Serial.update_time < current_time:
        # 等待一小段时间再检查条件
        # time.sleep(0.1)
        # # 检查超时
        # continue
        # if time.time() - current_time > timeout:
        #     return False
    # 返回计算坐标
    # print(imgProcess.coordinate)
    # return imgProcess.coordinate


class calibration(QWidget):
    def __init__(self,show_at_startup,Serial):
        super().__init__()
        self.show_at_startup=show_at_startup
        self.Serial = Serial
        self.init_ui()

    def init_ui(self):
        # 设置布局
        main_layout = QHBoxLayout()

        # 左侧布局
        left_layout = QVBoxLayout()

        # 选择标定方式
        calibrate_layout = QHBoxLayout()
        calibrate_label = QLabel("选择标定方式:")
        self.calibrate_combo = QComboBox()
        self.calibrate_combo.addItem("压力标定")
        self.calibrate_combo.addItem("气体浓度标定")
        self.calibrate_combo.currentIndexChanged.connect(self.update_input_labels)
        calibrate_layout.addWidget(calibrate_label)
        calibrate_layout.addWidget(self.calibrate_combo)
        left_layout.addLayout(calibrate_layout)

        # 输入坐标和计算结果
        coords_layout = QVBoxLayout()
        coords_label = QLabel("输入坐标:")
        coords_layout.addWidget(coords_label)

        # 创建位置输入框、按钮和结果标签
        self.position_inputs = []
        self.position_buttons = []
        self.result_labels = []  # 新增结果标签
        for i in range(5):
            position_layout = QHBoxLayout()
            position_label = QLabel(f"位置 {i + 1}:")
            position_input = QLineEdit()
            position_button = self.create_button("位置", position_input)
            result_label = QLabel("结果: ")

            self.position_inputs.append(position_input)
            self.position_buttons.append(position_button)
            self.result_labels.append(result_label)

            position_layout.addWidget(position_label)
            position_layout.addWidget(position_input)
            position_layout.addWidget(position_button)
            position_layout.addWidget(result_label)  # 将结果标签加入布局

            coords_layout.addLayout(position_layout)

        left_layout.addLayout(coords_layout)

        # 拟合按钮
        fit_button = QPushButton("拟合")
        fit_button.clicked.connect(self.fit_curve)
        left_layout.addWidget(fit_button)

        # 实时显示信息标签
        self.slope_label = QLabel("斜率: ")
        self.intercept_label = QLabel("截距: ")
        self.r_square_label = QLabel("R Square: ")
        self.mse_label = QLabel("均方误差: ")
        left_layout.addWidget(self.slope_label)
        left_layout.addWidget(self.intercept_label)
        left_layout.addWidget(self.r_square_label)
        left_layout.addWidget(self.mse_label)

        main_layout.addLayout(left_layout)

        # 右侧布局（曲线图）
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.canvas)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)
        self.setWindowTitle('标定')

        # 是否启动时弹出
        if self.show_at_startup:
            self.show()

    def update_input_labels(self):
        # 更新输入标签
        calibrate_method = self.calibrate_combo.currentText()
        for i in range(5):
            position_label = QLabel(f"位置 {i + 1}:")
            position_input = self.position_inputs[i]
            position_button = self.position_buttons[i]
            result_label = self.result_labels[i]

            position_label.setText(f"位置 {i + 1}:")
            position_button.setText("开始")

            if calibrate_method == "压力标定":
                result_label.setText("结果: 0.0000")  # Update result label if needed
            elif calibrate_method == "气体浓度标定":
                result_label.setText("结果: 0.0000")  # Update result label if needed

    def create_button(self, label, input_field):
        button = QPushButton("开始")
        button.clicked.connect(lambda: self.get_coordinate(label, input_field))
        return button

    def get_coordinate(self, label, input_field):
        # 获取输入坐标
        global updateData,dataNum
        updateData = True

        try:
            x = float(input_field.text())
        except ValueError:
            # 处理非数字输入
            self.show_warning("错误", "请输入有效的数字。")
            return

        # 获取标定方式
        calibrate_type = self.calibrate_combo.currentIndex()

        # 调用 cal 函数，根据返回值计算 y 坐标
        obtainCoordinates(x, calibrate_type,self.Serial)

        dataNum = input_field

    def update_coordinate(self):
        global updateData
        updateData = False
        # 更新坐标对应的输入框
        # input_field.setText(str(imgProcess.coordinate))

        # 更新结果标签
        index = self.position_inputs.index(dataNum)
        result_label = self.result_labels[index]
        result_label.setText(f"结果: {imgProcess.coordinate:.4f}")

    def show_warning(self, title, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()
    def fit_curve(self):
        global a_pressure,b_pressure,a_concentration,b_concentration
        # 获取选择的标定方式
        calibrate_type = self.calibrate_combo.currentIndex()

        # 获取输入数据
        positions = np.array([float(input_field.text()) for input_field in self.position_inputs])

        # 获取标定方式对应的坐标值
        if calibrate_type == 0:  # 压力标定
            pressure_concentration = np.array([float(input_field.text()) for input_field in self.position_inputs])
        elif calibrate_type == 1:  # 气体浓度标定
            pressure_concentration = np.array([float(input_field.text()) for input_field in self.position_inputs])

        # 定义拟合函数
        def linear_fit(x, a, b):
            return a * x + b

        # 初始拟合参数
        initial_params = [1.0, 1.0]

        # 执行拟合
        params, _ = curve_fit(linear_fit, positions, pressure_concentration, p0 = initial_params)

        # 计算相关信息
        slope = params[0]
        intercept = params[1]
        y_pred = linear_fit(positions, *params)
        r_square = r2_score(pressure_concentration, y_pred)
        mse = mean_squared_error(pressure_concentration, y_pred)

        # 保存标定结果
        if calibrate_type == 0:  # 压力标定
            a_pressure = params[0]
            b_pressure = params[1]
        elif calibrate_type == 1:  # 气体浓度标定
            a_concentration = params[0]
            b_concentration = params[1]
        # 实时更新信息标签
        self.slope_label.setText(f"斜率: {slope:.4f}")
        self.intercept_label.setText(f"截距: {intercept:.4f}")
        self.r_square_label.setText(f"R Square: {r_square:.4f}")
        self.mse_label.setText(f"均方误差: {mse:.4f}")

        # 绘制拟合曲线
        self.ax.clear()
        self.ax.scatter(positions, pressure_concentration, label = '实际数据')
        self.ax.plot(positions, linear_fit(positions, *params), 'r-', label = '拟合曲线')
        self.ax.legend()

        # 更新绘图
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    page = calibration(1)
    sys.exit(app.exec_())