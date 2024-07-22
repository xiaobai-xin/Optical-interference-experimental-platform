# 光干涉平台
################################################################################################
import sys
import time
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import os
# 以下是自定义模块
import config
import storagePath
import Serial
import imgProcess
import calibration
import RGB565Conversion
version =  \
"""当前版本v1.5.3
历史版本:
v1.0 串口基本功能&主页面 （24-1-22）
v1.1 串口、主页面图片显示、存储位置设定（24-1-23）
v1.2 首页图像自动处理（24-1-24）
    v1.2.1 增加数据自动保存功能（24-1-25）
    v1.2.2 首页功能完善,中心坐标计算方法改进（24-1-24）
v1.3 标定功能实现（24-1-26）
    v1.3.1 添加系统配置设置（24-2-29）
    v1.3.2 标定功能错误修改（24-2-30）
v1.4 增加应答机制（24-2-6）
    v1.4.1 增加串口超时处理（24-2-22）
    v1.4.2 增加定时连续自动保存功能（24-2-25）
    v1.4.3 增加LAB图像（24-2-28）
    v1.4.4 通过多线程写入excel,解决了页面卡顿问题（24-2-28）
    v1.4.5 解决了可能导致连续自动保存失效的bug,解决了用户串口设置错误导致程序崩溃的问题（24-2-28）
    v1.4.6 增加计算最小值(24-2-29)
    v1.4.7 增加保存下位机数据(24-3-5)
    v1.4.8 解决了标定过程中的错误,excel写入线程异常(24-3-25)
    v1.4.9 解决了中文路径异常的问题(24-3-25)
v1.5 改进位置计算方法(24-3-31)
    v1.5.1 增加图像读入功能(24-4-1)
    v1.5.2 修复了曲线拟合过程中的问题，计算结果和下位机一致(24-4-2)
    v1.5.3 改进计算方法，增加右侧最小值(24-4-16)
"""
class homepage(QWidget):

    def __init__(self):
        super().__init__()
        self.mode_multi_collect = False
        self.init_ui()
        self.multi_collect_restart = False
    # 用于传递其它窗口实例
    def set_Subpages(self, SerialSettingsTool,calibrationPage,configPage):
        self.SerialSettingsTool = SerialSettingsTool
        self.calibrationPage = calibrationPage
        self.configPage = configPage
    def init_ui(self):
        self.ui = uic.loadUi("./static/homepage.ui")
        print(self.ui.__dict__)  # 查看ui文件中有哪些控件
        self.ui.setWindowTitle("光干涉压力测试平台 v1.5.3")
        # 加载串口设置界面   为了确保窗口在关闭时不被销毁，一般建议将窗口实例作为类的成员变量，而不是作为局部变量。这样，窗口的生命周期将与其父窗口或应用程序保持一致，而不是受到局部变量的生命周期限制。
        self.ui.setWindowIcon(QIcon("./static/img/ico64x64.ico"))

        # ui控件
        self.status = self.ui.status   # 状态指示
        font = QFont()
        font.setPointSize(15)
        self.status.setStyleSheet("color:red")
        self.status.setFont(font)
        self.status.setText("软件启动成功，请先进行串口配置")

        # 按钮部分
        self.button_calibration = self.ui.pushButton_calibration  # homepage-标定
        self.button_serial_set = self.ui.pushButton_serial_set  # homepage-串口设置
        self.button_serial_start = self.ui.pushButton_serial_start # homepage-打开串口
        self.button_datasave = self.ui.pushButton_savedata  # homepage-数据采集
        self.checkBox_multi_collect = self.ui.checkBox_multi_collect
        self.button_imgread = self.ui.pushButton_imgread  # homepage-打开图片
        self.pushButton_serialdata_read = self.ui.pushButton_serialdata_read  # homepage-串口数据读入
        self.button_serial_start.setStyleSheet("background-color:red")# homepage-打开串口按钮默认红色
        # 菜单部分
        self.menu_file_address = self.ui.menu_file_address # 菜单-文件-保存地址
        self.menu_file_set = self.ui.menu_file_set  # 菜单-文件-保存地址
        self.menu_about = self.ui.menu_about_version  # 菜单-关于-版本信息
        # 中心位置显示
        # 最大位置
        self.centerMax_RGB_R = self.ui.label_centerMax_RGB_R
        self.centerMax_RGB_G = self.ui.label_centerMax_RGB_G
        self.centerMax_RGB_B = self.ui.label_centerMax_RGB_B
        self.centerMax_GRAY = self.ui.label_centerMax_GRAY
        # self.centerLAB = self.ui.label_center_LAB
        self.centerMax_LAB_L = self.ui.label_centerMax_LAB_L
        # 右侧最小位置
        self.rightMin_RGB_R = self.ui.label_rightMin_RGB_R
        self.rightMin_RGB_G = self.ui.label_rightMin_RGB_G
        self.rightMin_RGB_B = self.ui.label_rightMin_RGB_B
        self.rightMin_GRAY = self.ui.label_rightMin_GRAY
        self.rightMin_LAB_L = self.ui.label_rightMin_LAB_L
        # 左侧最小位置
        self.leftMin_RGB_R = self.ui.label_leftMin_RGB_R
        self.leftMin_RGB_G = self.ui.label_leftMin_RGB_G
        self.leftMin_RGB_B = self.ui.label_leftMin_RGB_B
        self.leftMin_GRAY = self.ui.label_leftMin_GRAY
        self.leftMin_LAB_L = self.ui.label_leftMin_LAB_L
        # 下位机计算结果
        self.MCU_results = self.ui.label_MCU_results

        self.centerAvr = self.ui.label_position_avr # 平均位置
        self.label_pressure = self.ui.label_pressure
        self.label_ch4_concentration = self.ui.label_ch4_concentration


        # 图像显示部分
        self.img_RGB = self.ui.label_img_RGB  # RGB图像
        self.img_RGB_R = self.ui.label_img_R  # RGB R图像
        self.img_RGB_G = self.ui.label_img_G  # RGB G图像
        self.img_RGB_B = self.ui.label_img_B  # RGB B图像
        self.img_GRAY = self.ui.label_img_GRAY  # 灰度图像
        self.img_LAB = self.ui.label_img_LAB  # LAB图像
        self.img_LAB_L = self.ui.label_img_LAB_L  # LAB L图像
        # 初始化首页图像显示
        imgProcess.update_image(self)



        # 绑定信号与槽函数
        self.button_calibration.clicked.connect(self.calibration)  # 标定
        self.button_serial_set.clicked.connect(self.serial_set)
        self.button_serial_start.clicked.connect(self.serial_start) # 打开串口


        self.button_datasave.clicked.connect(self.serial_savedata)  # 保存数据
        self.checkBox_multi_collect.stateChanged.connect(self.toggle_multi_collect)

        self.button_imgread.clicked.connect(self.openImage)  # 读取图片
        self.pushButton_serialdata_read.clicked.connect(self.open_serial_data)  # 读取图片
        self.menu_about.triggered.connect(self.show_about)
        self.menu_file_address.triggered.connect(self.change_save_address)
        self.menu_file_set.triggered.connect(self.sys_set)
    # 系统设置
    def sys_set(self):
        self.configPage.show()
    # 连续数据采集复选框状态切换槽函数
    def toggle_multi_collect(self, state):
        if state == Qt.Checked:
            self.mode_multi_collect = True
            # 如果复选框被选中，设置定时器
            self.multi_collect_timer = QTimer()
            self.multi_collect_timer.timeout.connect(self.serial_savedata)
            # 乘1000将单位转换成秒
            self.multi_collect_timer.start(config.multi_collect_delay*1000)
            self.multi_collect_counter = config.multi_collect_freq
        else:
            # 如果复选框未被选中，停止定时器
            self.mode_multi_collect = False
            if hasattr(self, 'multi_collect_timer'):
                self.multi_collect_timer.stop()

    # 标定按钮
    def calibration(self):
        self.calibrationPage.show()
    # 串口设置
    def serial_set(self):
        self.SerialSettingsTool.show()

    # 开启串口
    def serial_start(self):
        self.SerialSettingsTool.toggle_serial_port()

    # 采集数据
    def serial_savedata(self):
        self.status.setText("正在采集数据...")
        if self.multi_collect_restart:
            self.multi_collect_counter = config.multi_collect_freq
            self.multi_collect_timer.start(config.multi_collect_delay * 1000)  # 重新启动定时器
            self.multi_collect_restart = False
        self.SerialSettingsTool.get_data()
        if self.mode_multi_collect:
            self.button_datasave.setStyleSheet("background-color:red")

            # 减少计数器，当计数器为0时停止定时器
            print(self.multi_collect_counter)
            self.multi_collect_counter -= 1
            self.status.setText(f"第{config.multi_collect_freq - self.multi_collect_counter}次数据采集中...")
            if self.multi_collect_counter == 0:

                if hasattr(self, 'multi_collect_timer'):
                    self.multi_collect_timer.stop()
                    time.sleep(7) # 按钮指示颜色延迟,等待串口接收完毕
                    self.button_datasave.setStyleSheet("")
                    self.status.setText("连续数据采集已完成")
                    self.multi_collect_restart = True





    # 打开图片
    def openImage(self):
        # 使用文件对话框获取用户选择的图片路径
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "BMP 图片 (*.bmp);;所有文件 (*)", options=options)

        if fileName:
            current_time = time.strftime("%Y%m%d%H%M%S")
            imgProcess.BMP_process(fileName,current_time,None,self)
    #
    def open_serial_data(self):
        # 使用文件对话框获取用户选择的图片路径
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "读取串口数据", "", "txt文档 (*.txt);;所有文件 (*)", options=options)
        print("fileName:")
        print(fileName)
        if fileName:
            current_time = time.strftime("%Y%m%d%H%M%S")
            path = f"RGB_{current_time}_serial_reading.bmp"
            pathImg_RGB = os.path.join(storagePath.pathImgData, "RGB_img", path)
            # 调用RGB565转换模块
            RGB565Conversion.RGB565toRGB888(fileName, pathImg_RGB)
            # BMP图像处理
            imgProcess.BMP_process(pathImg_RGB,current_time,None,self)






    # 修改文件存储路径
    def change_save_address(self):
        dialog = storagePath.SavePathDialog(storagePath.path)
        result = dialog.exec_()  # Show the dialog as modal
        if result == QDialog.Accepted:
            print("保存路径:", storagePath.path)
    # 版本信息
    def show_about(self):
        QMessageBox.about(self, "版本信息",version)



if __name__ == '__main__':
    # 读取系统配置文件
    config.init()
    # 初始化存储路径配置
    storagePath.initPath()
    # QT应用实例
    app = QApplication(sys.argv)
    # 实例化主页
    w = homepage()
    # 实例化其他页面
    SerialSettingsTool = Serial.SerialSettingsTool(w)   # 串口设置页面
    calibrationPage = calibration.calibration(config.popup,SerialSettingsTool) # 标定页面
    SerialSettingsTool.set_Subpages(calibrationPage)
    configPage = config.ConfigPage()

    # 向主窗口传递其他窗口的实例,避免循环依赖
    w.set_Subpages(SerialSettingsTool,calibrationPage,configPage)
    w.ui.show()

    app.exec()
