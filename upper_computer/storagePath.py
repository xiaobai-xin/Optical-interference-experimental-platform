# 模块功能: 存储路径设置(包括存储位置全局变量及相关方法 修改存储位置页面）
################################################################################################
# 设置存储路径窗口
from PyQt5.QtWidgets import QMessageBox,QDialog,QVBoxLayout,QLabel,QPushButton,QFileDialog
import os
import pandas as pd
import time
from threading import Thread
import shutil
# 全局变量
path = os.path.abspath('./')  # 默认保存地址
pathSerialData = None # 串口数据默认保存地址
pathImgData = None # 图片数据保存地址
# 修改地址变量
def changePath(newPath):
    global path
    path = newPath
    initPath()


# 创建写入excel线程
def start_write_excel_thread(center_R, center_G, center_B, center_GRAY, center_LAB_L,\
                             rightMin_RGB_R, rightMin_RGB_G, rightMin_RGB_B, rightMin_GRAY, rightMin_LAB_L,\
                             leftMin_RGB_R, leftMin_RGB_G, leftMin_RGB_B, leftMin_GRAY, leftMin_LAB_L, \
                             center_avr, MCU_result,pressure=None, CH4_concentration=None):
    write_excel_thread = Thread(target=store_in_excel, \
                                args=(center_R, center_G, center_B, center_GRAY,center_LAB_L,\
                                      rightMin_RGB_R, rightMin_RGB_G, rightMin_RGB_B, rightMin_GRAY, rightMin_LAB_L,\
                                      leftMin_RGB_R, leftMin_RGB_G, leftMin_RGB_B, leftMin_GRAY, leftMin_LAB_L, \
                                      center_avr,MCU_result,pressure,CH4_concentration))
    write_excel_thread.start()

# 写入excel
def store_in_excel(center_R, center_G, center_B, center_GRAY,center_LAB_L,\
                   rightMin_RGB_R, rightMin_RGB_G, rightMin_RGB_B, rightMin_GRAY, rightMin_LAB_L,\
                   leftMin_RGB_R, leftMin_RGB_G, leftMin_RGB_B, leftMin_GRAY, leftMin_LAB_L, \
                   center_avr, MCU_result, pressure=None, CH4_concentration=None):
    global path
    excelPath = os.path.join(path, 'data.xlsx')  # 构建完整的Excel文件路径

    try:
        df = pd.read_excel(excelPath)
        # 移除全是NA的列
        df = df.dropna(axis=1, how='all')
    except FileNotFoundError:
        # 如果文件不存在，创建一个新的DataFrame
        df = pd.DataFrame(columns=['日期', 'R分量最大位置', 'G分量最大位置', 'B分量最大位置', 'GRAY分量最大位置','L分量最大位置', \
                                   'R分量右侧最小位置', 'G分量右侧最小位置', 'B分量右侧最小位置', 'GRAY分量右侧最小位置','L分量右侧最小位置',\
                                   'R分量左侧最小位置', 'G分量左侧最小位置', 'B分量左侧最小位置', 'GRAY分量左侧最小位置','L分量左侧最小位置',\
                                   '下位机计算结果','平均位置','压力', 'CH4浓度'])

    # 获取当前日期
    current_datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    # 处理可选参数，如果为None，则设为默认值
    pressure = pressure if pressure is not None else 'N/A'
    CH4_concentration = CH4_concentration if CH4_concentration is not None else 'N/A'

    # 创建新的一行数据
    new_row = pd.DataFrame({
        '日期': [current_datetime],
        'R分量最大位置': [center_R],
        'G分量最大位置': [center_G],
        'B分量最大位置': [center_B],
        'GRAY分量最大位置': [center_GRAY],
        'L分量最大位置': [center_LAB_L],
        'R分量右侧最小位置': [rightMin_RGB_R],
        'G分量右侧最小位置': [rightMin_RGB_G],
        'B分量右侧最小位置': [rightMin_RGB_B],
        'GRAY分量右侧最小位置': [rightMin_GRAY],
        'L分量右侧最小位置': [rightMin_LAB_L],
        'R分量左侧最小位置': [leftMin_RGB_R],
        'G分量左侧最小位置': [leftMin_RGB_G],
        'B分量左侧最小位置': [leftMin_RGB_B],
        'GRAY分量左侧最小位置': [leftMin_GRAY],
        'L分量左侧最小位置': [leftMin_LAB_L],
        '下位机计算结果': [MCU_result],
        '平均位置': [center_avr],
        '压力': [pressure],
        'CH4浓度': [CH4_concentration]
    })
    # 将新行添加到DataFrame
    df = pd.concat([df, new_row], ignore_index=True)

    # 将DataFrame保存回Excel文件
    df.to_excel(excelPath, index=False)

# 检查子文件夹是否存在并创建
def initPath():
    global path, pathSerialData, pathImgData
    # 检查serialData文件夹是否存在
    serial_data_path = os.path.join(path, 'serialData')
    if not os.path.exists(serial_data_path):
        os.makedirs(serial_data_path)
        print(f'创建 {serial_data_path}')

    # 检查imgData文件夹是否存在
    img_data_path = os.path.join(path, 'imgData')
    if not os.path.exists(img_data_path):
        os.makedirs(img_data_path)
        print(f'创建 {img_data_path}')

    # 添加RGB_img、RGB_R_img、RGB_G_img、RGB_B_img、Gray_img、LAB_img、LAB_L_img文件夹到imgData中
    subfolders = ['RGB_img', 'RGB_R_img', 'RGB_G_img', 'RGB_B_img', 'Gray_img', 'LAB_img', 'LAB_L_img']
    for folder in subfolders:
        folder_path = os.path.join(img_data_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f'创建 {folder_path}')

    pathSerialData = os.path.join(path, 'SerialData')
    pathImgData = os.path.join(path, 'ImgData')


class SavePathDialog(QDialog):
    def __init__(self, current_path):
        super().__init__()
        self.current_path = current_path

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setGeometry(100, 100, 300, 200)
        self.setWindowTitle("存储路径设置")
        self.label_cur_url = QLabel("当前路径: {}".format(self.current_path))
        self.label_prompt = QLabel(
"""
系统会在您设置的目标文件夹创建如下目录和文件:
1)serialData:串口数据
2)imgData:转换后的图像数据
    |__  RGB_img
    |__  RGB_R_img
    |__  RGB_G_img
    |__  RGB_B_img
    |__  Gray_img
    |__  LAB_img
    |__  LAB_L_img
3)data.xlsx:计算结果数据"""
    )
        button_choose_folder = QPushButton("选择文件夹")
        button_save_path = QPushButton("保存路径")
        button_clear_data = QPushButton("清除数据")

        # 绑定信号与槽函数
        button_choose_folder.clicked.connect(self.choose_folder)
        button_save_path.clicked.connect(self.save_current_path)
        button_clear_data.clicked.connect(self.confirm_clear_data)

        # 页面布局
        layout.addWidget(self.label_cur_url)
        layout.addWidget(button_choose_folder)
        layout.addWidget(button_save_path)
        layout.addWidget(button_clear_data)
        layout.addWidget(self.label_prompt)
        self.setLayout(layout)

    def choose_folder(self):
        new_folder = QFileDialog.getExistingDirectory(self, "选择保存路径", self.current_path)
        # 检查操作系统，如果是Windows，则替换斜杠为反斜杠
        if os.name == 'nt':  # 'nt' 是 Windows 操作系统的标识符
            new_folder = new_folder.replace('/', '\\')
        if new_folder:
            self.current_path = new_folder
            self.label_cur_url.setText("当前路径: {}".format(self.current_path))

    def save_current_path(self):
        print(self.current_path)

        changePath(self.current_path)
        self.accept()

    def confirm_clear_data(self):
        confirm_dialog = QMessageBox()
        confirm_dialog.setIcon(QMessageBox.Warning)
        confirm_dialog.setText("确认清除数据？")
        confirm_dialog.setInformativeText("此操作将清除串口数据、图像数据以及data.xlsx文件，确定要继续吗？")
        confirm_dialog.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        confirm_dialog.setDefaultButton(QMessageBox.Cancel)
        confirm_dialog.buttonClicked.connect(self.handle_clear_data_confirmation)
        confirm_dialog.exec_()

    def handle_clear_data_confirmation(self, button):
        if button.text() == "OK":
            try:
                # 清空serialData文件夹
                serial_data_folder = os.path.join(self.current_path, 'serialData')
                if os.path.exists(serial_data_folder):
                    for file_name in os.listdir(serial_data_folder):
                        file_path = os.path.join(serial_data_folder, file_name)
                        if os.path.isfile(file_path):
                            os.remove(file_path)

                # 递归清空imgData文件夹
                img_data_folder = os.path.join(self.current_path, 'imgData')
                if os.path.exists(img_data_folder):
                    self.clear_folder(img_data_folder)

                # 删除data.xlsx文件
                data_file_path = os.path.join(self.current_path, 'data.xlsx')
                if os.path.exists(data_file_path):
                    os.remove(data_file_path)

                QMessageBox.information(self, "提示", "数据已清除")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"清除数据时出错: {str(e)}")

    def clear_folder(self, folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                os.remove(file_path)

