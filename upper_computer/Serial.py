# 模块功能: 串口数据处理
################################################################################################
# 修改日志
# 24-1-24
#           增加功能:调用RGB565Conversation转bmp并保存
#           增加功能:更新主页面的图片
# 24-1-25
#           增加功能:达到51200数据自动保存，无需手动按保存按钮
#           增加功能:超时处理功能，如果超过3秒未收到数据且当前数据未达到51200，直接清空缓存，等着接收下一次数据
#           待修复bug:更新文本框的部分，在波特率高时会导致程序崩溃，暂时禁用该模块

# 23-2-20 c串口应答
#           串口指令    发送方    内容
#           0x00       上位机    下位机数据就绪时直接发送数据(调试使用，只需要向下位机发送一次完成设置)



#           0x01       上位机    请求发送数据
#           0x02       下位机    准备就绪
#           0x03       上位机    开始数据传输
# 23-2-22 增加串口超时处理
import sys
import serial
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QComboBox, QPushButton, \
    QTextEdit, QWidget, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QIcon
import time
import serial.tools.list_ports
import os

# 以下是自定义模块
import storagePath
import RGB565Conversion
import imgProcess
import config
import calibration
# 全局变量
update_time = time.time()
mode_receive = False    # 串口接收进程 数据接收模式
mode_establish = False  # 串口接收进程 请求发送模式
time_establish = time.time()
class SerialReader(QThread):
    data_received = pyqtSignal(str)   # 接收到串口数据时
    save_data_signal = pyqtSignal()   # 保存数据
    Timeout_Handling = pyqtSignal()   # 建立连接超时
    def __init__(self, serial_port, timeout=5):
        super().__init__()
        self.serial_port = serial_port
        self.data_buffer = bytearray()
        self.timeout = timeout
        self.last_data_time = time.time()
    def run(self):
        global mode_establish,mode_receive,time_establish,update_time
        count = 0  # 超时重发次数计数器
        try:
            while True:

                # 与下位机建立连接
                if mode_establish:
                    data = self.serial_port.read(1)
                    if data == bytes([0x02]):
                        print("成功与下位机建立连接")
                        self.serial_port.write(bytes([0x03]))
                        mode_establish = False
                        mode_receive = True
                        count = 0
                    else:
                        self.serial_port.flushInput() # 清空等待期间可能收到的其他数据
                        time.sleep(1)
                        if count < 3:
                            # 超时后尝试再次发送请求
                            self.serial_port.write(bytes([0x01]))
                            print("再次尝试请求")
                            time_establish = time.time()
                            count += 1
                        else:
                            mode_establish = False
                            count = 0
                            print("请求超时1")
                            self.Timeout_Handling.emit()




                # 接收数据模式
                # 使用read_all()可能造成数据异常，因此每次读取固定值
                # data = self.serial_port.read_all()
                elif mode_receive:
                    data = self.serial_port.read()  # 每次读取64字节
                    if data:
                        hex_data = " ".join([format(byte, '02X') for byte in data])
                        self.data_received.emit(hex_data)
                        # print(f"Received data: {hex_data}")

                        self.data_buffer.extend(data)
                        self.last_data_time = time.time()

                        # 目标数据量 320*40*4 =51200
                        # +32以接收下位机的坐标计算结果
                        # print(f"count: {len(self.data_buffer)}")
                        if len(self.data_buffer) >= config.dataSize+4:



                            self.save_data_signal.emit()  # Emit signal to trigger save_data function
                            self.data_buffer.clear()  # Clear the buffer after saving
                            mode_receive = False
                            # 记录最近一次更新时间 当前时间戳
                            update_time = time.time()
                # 超时检查
                elif mode_receive and time.time() - self.last_data_time > self.timeout and len(self.data_buffer) < 51200:
                    self.data_buffer.clear()
                    self.serial_port.flushInput()
                    print("请求超时2")
                    self.Timeout_Handling.emit()
                    mode_receive = False
                else:
                    # 降低CPU占用
                    time.sleep(1)

        except Exception as e:
            print(f"Error reading serial data: {e}")
class SerialSettingsTool(QMainWindow):
    def __init__(self,homepage):
        super().__init__()
        self.homepage =homepage
        self.setWindowTitle("串口设置")
        self.setGeometry(100, 100, 600, 400)

        self.init_ui()


        self.data_buffer = bytearray()
        self.serial_reader = SerialReader(None)
        self.serial_reader.Timeout_Handling.connect(self.Timeout_Handling)
        # Connect save_data function to save_data_signal
        self.serial_reader.save_data_signal.connect(self.save_data)
    def Timeout_Handling(self):
        QMessageBox.warning(self, "请求超时", "请检查串口设置是否正确或下位机是否正常工作",
                            QMessageBox.Yes)
        self.homepage.status.setText("请求超时")
    def set_Subpages(self,calibration):
        self.calibration = calibration
    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.com_label = QLabel("COM口:", self)
        self.com_combobox = QComboBox(self)
        self.populate_com_ports()  # 初始化串口选择列表
        self.refresh_button = QPushButton("刷新串口列表", self)
        self.refresh_button.clicked.connect(self.populate_com_ports)

        self.baudrate_label = QLabel("波特率:", self)
        self.baudrate_combobox = QComboBox(self)
        self.baudrate_combobox.addItems(["9600", "115200", "256000"])
        # 设置默认选项为115200
        self.baudrate_combobox.setCurrentText("115200")


        self.databits_label = QLabel("数据位:", self)
        self.databits_combobox = QComboBox(self)
        self.databits_combobox.addItems(["8", "7", "6", "5"])

        self.parity_label = QLabel("校验位:", self)
        self.parity_combobox = QComboBox(self)
        self.parity_combobox.addItems(["None", "Even", "Odd", "Mark", "Space"])

        self.stopbits_label = QLabel("停止位:", self)
        self.stopbits_combobox = QComboBox(self)
        self.stopbits_combobox.addItems(["1", "1.5", "2"])

        # self.open_button = QPushButton("打开串口", self)
        # self.open_button.clicked.connect(self.open_serial_port)

        # Create "打开串口" button
        self.open_button = QPushButton("打开串口", self)
        self.open_button.clicked.connect(self.toggle_serial_port)
        self.serial_open = False  # 串口状态标志
        self.open_button.setStyleSheet("background-color:red")

        self.save_button = QPushButton("保存数据", self)
        self.save_button.clicked.connect(self.save_data)

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)

        layout.addWidget(self.com_label)
        layout.addWidget(self.com_combobox)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.baudrate_label)
        layout.addWidget(self.baudrate_combobox)
        layout.addWidget(self.databits_label)
        layout.addWidget(self.databits_combobox)
        layout.addWidget(self.parity_label)
        layout.addWidget(self.parity_combobox)
        layout.addWidget(self.stopbits_label)
        layout.addWidget(self.stopbits_combobox)
        layout.addWidget(self.open_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.text_edit)

        self.serial = None
        self.serial_reader = None

    # def populate_com_ports(self):
    #     self.com_combobox.clear()
    #     ports = [info.device for info in serial.tools.list_ports.comports()]
    #     self.com_combobox.addItems(ports)

    def populate_com_ports(self):
        self.com_combobox.clear()
        ports_info = [info for info in serial.tools.list_ports.comports()]
        port_names = [info.device + ': ' + info.description if info.description else info.device for info in ports_info]
        self.com_combobox.addItems(port_names)
    def toggle_serial_port(self):
        if self.serial_open:
            # If the serial port is open, close it
            try:
                if self.serial and self.serial.is_open:
                    self.serial.close()
                    self.serial_open = False
                    self.open_button.setText("打开串口")
                    self.homepage.button_serial_start.setText("打开串口")
                    self.open_button.setStyleSheet("background-color:red")
                    self.homepage.button_serial_start.setStyleSheet("background-color:red")
                    self.homepage.status.setText("串口已关闭")
                    return
            except Exception as e:
                print(f"无法关闭串口: {e}")

        # If the serial port is closed, open it
        port_description = self.com_combobox.currentText()
        port = port_description.split(':')[0].strip()  # 提取出端口号部分
        baudrate = int(self.baudrate_combobox.currentText())
        databits = int(self.databits_combobox.currentText())
        parity = self.map_parity(self.parity_combobox.currentText())
        stopbits = float(self.stopbits_combobox.currentText())

        try:
            self.serial = serial.Serial(port, baudrate, bytesize=databits,
                                        parity=parity, stopbits=stopbits, timeout=1)
            # set_buffer_size在unbuntu系统报错
            # self.serial.set_buffer_size(51200, 51200)
            self.serial_open = True
            self.open_button.setText("关闭串口")
            self.homepage.button_serial_start.setText("关闭串口")
            self.open_button.setStyleSheet("background-color:green")
            self.homepage.button_serial_start.setStyleSheet("background-color:green")
            self.homepage.status.setText("串口已开启")
            print("串口已打开")
            self.serial_reader = SerialReader(self.serial)
            self.serial_reader.Timeout_Handling.connect(self.Timeout_Handling)
            self.serial_reader.data_received.connect(self.update_text_edit)  #接收到串口数据时
            self.serial_reader.start()
        except Exception as e:

            QMessageBox.warning(self, "警告", f"无法打开串口: {e}",
                                QMessageBox.Yes)
        # Connect save_data function to save_data_signal
        self.serial_reader.save_data_signal.connect(self.save_data)

    # 串口发送
    def get_data(self):
        global mode_establish, time_establish
        if self.serial_open:
            if mode_receive:
                # QMessageBox.warning(self, "提示", "请等待当前接收结束",
                #                     QMessageBox.Yes)
                pass
            else:
                mode_establish = True  # 串口接收进程进入建立连接模式
                time_establish = time.time()   # 记录请求数据传输的时间
                self.serial.write(bytes([0x01]))
                # print("sent 0x01")
        else:
            QMessageBox.warning(self, "警告", "请先进行串口设置并打开串口",
                                QMessageBox.Yes)

    def save_data(self):

        print("Saving data...")  # Add this line for debugging

        # if not self.serial or not self.serial.is_open:
        #     print("串口未打开")
        #     return

        # try:
        current_time = time.strftime("%Y%m%d%H%M%S")

        # 解析下位机计算结果部分
        # 获取最后4个字节
        last_8_bytes = [int(chr(byte), 16) for byte in self.data_buffer[-8:]]

        # 解析整数位和小数位
        MCU_results_integer_part = last_8_bytes[7] + last_8_bytes[6] * 16 + last_8_bytes[5] * 16 * 16 + last_8_bytes[
            4] * 16 * 16 * 16
        MCU_results_decimal_part = (last_8_bytes[3] + last_8_bytes[2] * 16 + last_8_bytes[1] * 16 * 16 + last_8_bytes[
            0] * 16 * 16 * 16 + 1) / 1000.0

        # 合并整数位和小数位，并转换为浮点数
        MCU_results = MCU_results_integer_part + MCU_results_decimal_part

        # 保存串口数据
        # pathSerial = storagePath.pathSerialData + f"\data_{current_time}_{MCU_results}.txt"
        path = f"data_{current_time}_{MCU_results}.txt"
        pathSerial = os.path.join(storagePath.pathSerialData, path)
        # 保存bmp图像数据
        # pathImg_RGB = storagePath.pathImgData + f"\RGB_img\RGB_{current_time}_{MCU_results}.bmp"
        path = f"RGB_{current_time}_{MCU_results}.bmp"
        pathImg_RGB = os.path.join(storagePath.pathImgData, "RGB_img", path)
        with open(pathSerial, 'wb') as file:
            file.write(bytes(self.data_buffer))
        print(f"成功: 数据已保存到 {pathSerial}")

        self.data_buffer.clear()

        # Move the clearing of the buffer here
        self.serial_reader.data_buffer.clear()

        # 调用RGB565转换模块
        RGB565Conversion.RGB565toRGB888(pathSerial, pathImg_RGB)

        # BMP图像处理
        imgProcess.BMP_process(pathImg_RGB, current_time, MCU_results, self.homepage)

        # 更新标定数据
        if calibration.updateData:
            self.calibration.update_coordinate()

        # 完成采集提示
        self.homepage.status.setText("已完成一次采集")

        # except Exception as e:
        #     print(f"Error saving serial data: {e}")

    def map_parity(self, parity):
        parity_mapping = {
            "None": 'N',
            "Even": 'E',
            "Odd": 'O',
            "Mark": 'M',
            "Space": 'S',
        }
        return parity_mapping.get(parity, 'N')

    def update_text_edit(self, data):
        try:
            # print(f"更新文本框: {data}")


            # 更新文本框的部分，已知在波特率高时会导致程序崩溃
            # cursor = self.text_edit.textCursor()
            # cursor.movePosition(QTextCursor.End)
            # self.text_edit.setTextCursor(cursor)
            # self.text_edit.insertPlainText(data)
            # QApplication.processEvents()
            # self.text_edit.moveCursor(QTextCursor.End)     #将光标移动到 QTextEdit 小部件中的文本末


            self.data_buffer.extend(data.encode("utf-8"))
        except Exception as e:
            print(f"Error updating text edit: {e}")
