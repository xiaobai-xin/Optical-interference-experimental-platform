from PyQt5.QtWidgets import QGroupBox, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QLineEdit
from PyQt5.QtCore import pyqtSignal
import os
import json
# 全局变量
width = ''
height = ''
popup = ''
dataSize = 0
multi_collect_freq = ''
multi_collect_delay = ''
def init():
    global width,height,popup,dataSize,multi_collect_freq,multi_collect_delay

    read_config()
    popup = read_config().get('popup')  # 标定页面自动弹出 True:主界面启动时弹出 False:手动弹出
    width = read_config().get('width')
    height = read_config().get('height')
    multi_collect_freq = read_config().get('multi_collect_freq')
    multi_collect_delay= read_config().get('multi_collect_delay')
    # 串口接收数据大小
    dataSize = width*height*2


# 外部读取系统配置文件
def read_config():
    config_path = './CONFIG.json'

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # 若不存在配置文件，则以下面默认值创建
        config = {
            'popup': False,
            'width': 320,
            'height': 40,
            'multi_collect_delay':50,
            'multi_collect_freq':5
        }
    with open(config_path, 'w') as f:
        json.dump(config, f)

    return config
# 系统配置相关设置
class ConfigPage(QWidget):
    configSaved = pyqtSignal()  # 关闭窗口信号

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('配置页面')

        self.checkbox = QCheckBox('标定页面在启动时弹出')
        self.checkbox.setChecked(read_config()['popup'])

        input_image_label = QLabel('输入图像设置:')
        self.width_label = QLabel('宽度(px):')
        self.width_input = QLineEdit()
        self.width_input.setText(str(read_config()['width']))
        self.height_label = QLabel('高度(px):')
        self.height_input = QLineEdit()
        self.height_input.setText(str(read_config()['height']))

        auto_save_label = QLabel('自动连续保存设置:')
        self.save_count_label = QLabel('连续保存次数:')
        self.save_count_input = QLineEdit()
        self.save_count_input.setText(str(read_config()['multi_collect_freq']))
        self.save_interval_label = QLabel('连续保存时间间隔(s):')
        self.save_interval_input = QLineEdit()
        self.save_interval_input.setText(str(read_config()['multi_collect_delay']))

        save_button = QPushButton('保存配置')
        save_button.clicked.connect(self.saveConfig)

        layout = QVBoxLayout()
        layout.addWidget(self.checkbox)

        input_image_group = QGroupBox("输入图像设置")
        input_image_layout = QHBoxLayout()
        input_image_layout.addWidget(self.width_label)
        input_image_layout.addWidget(self.width_input)
        input_image_layout.addWidget(self.height_label)
        input_image_layout.addWidget(self.height_input)
        input_image_group.setLayout(input_image_layout)

        auto_save_group = QGroupBox("自动连续保存设置(建议时间间隔大于50s)")
        auto_save_layout = QHBoxLayout()
        auto_save_layout.addWidget(self.save_count_label)
        auto_save_layout.addWidget(self.save_count_input)
        auto_save_layout.addWidget(self.save_interval_label)
        auto_save_layout.addWidget(self.save_interval_input)
        auto_save_group.setLayout(auto_save_layout)

        layout.addWidget(input_image_group)
        layout.addWidget(auto_save_group)
        layout.addWidget(save_button)

        self.setLayout(layout)

        self.configSaved.connect(self.close)  # 连接信号到关闭窗口槽函数

    def saveConfig(self):
        config = {
            'popup': self.checkbox.isChecked(),
            'width': int(self.width_input.text()),
            'height': int(self.height_input.text()),
            'multi_collect_delay': int(self.save_interval_input.text()),
            'multi_collect_freq': int(self.save_count_input.text())
        }

        config_path = './CONFIG.json'
        with open(config_path, 'w') as f:
            json.dump(config, f)
        init() # 重新初始化变量
        self.configSaved.emit()  # 发射关闭窗口信号

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('系统配置')

        config_page = ConfigPage()


        layout = QVBoxLayout()

        layout.addWidget(config_page)

        self.setLayout(layout)


