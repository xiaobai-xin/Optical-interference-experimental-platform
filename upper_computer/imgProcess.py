# 模块功能: 图像数据处理
################################################################################################
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
import storagePath
from scipy.optimize import curve_fit
import calibration
# 全局变量
pathInitImg='./static/img/homepage_init.bmp'  # 默认首页图像
coordinate = 0  # 计算坐标

def RGB_to_R(pathRGB, pathR):
    try:
        # 读取RGB图像
        img = cv2.imdecode(np.fromfile(pathRGB, dtype=np.uint8), 1)
        # 分割通道，获取红色通道
        r_channel = img[:,:,2]
        # 将红色通道保存到指定路径
        _, encoded_img = cv2.imencode('.' + pathR.split('.')[-1], r_channel)
        with open(pathR, 'wb') as f:
            f.write(encoded_img)
    except Exception as e:
        print("Error:", e)

def RGB_to_G(pathRGB, pathG):
    try:
        # 读取RGB图像
        img = cv2.imdecode(np.fromfile(pathRGB, dtype=np.uint8), 1)
        # 分割通道，获取绿色通道
        g_channel = img[:, :, 1]
        # 将绿色通道保存到指定路径
        _, encoded_img = cv2.imencode('.' + pathG.split('.')[-1], g_channel)
        with open(pathG, 'wb') as f:
            f.write(encoded_img)
    except Exception as e:
        print("Error:", e)

def RGB_to_B(pathRGB, pathB):
    try:
        # 读取RGB图像
        img = cv2.imdecode(np.fromfile(pathRGB, dtype=np.uint8), 1)
        # 分割通道，获取蓝色通道
        b_channel = img[:, :, 0]
        # 将蓝色通道保存到指定路径
        _, encoded_img = cv2.imencode('.' + pathB.split('.')[-1], b_channel)
        with open(pathB, 'wb') as f:
            f.write(encoded_img)
    except Exception as e:
        print("Error:", e)

def RGB_to_GRAY(pathRGB, pathGRAY):
    try:
        # 读取RGB图像
        img = cv2.imdecode(np.fromfile(pathRGB, dtype=np.uint8), 1)
        # 将RGB图像转换为灰度图像
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 保存灰度图像到指定路径
        _, encoded_img = cv2.imencode('.' + pathGRAY.split('.')[-1], gray_img)
        with open(pathGRAY, 'wb') as f:
            f.write(encoded_img)
    except Exception as e:
        print("Error:", e)

def RGB_to_LAB(pathRGB, pathLAB):
    try:
        # 读取RGB图像
        image_rgb = cv2.imdecode(np.fromfile(pathRGB, dtype=np.uint8), 1)
        # 将RGB图像转换为LAB空间
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2LAB)
        # 保存LAB图像到指定路径
        _, encoded_img = cv2.imencode('.' + pathLAB.split('.')[-1], image_lab)
        with open(pathLAB, 'wb') as f:
            f.write(encoded_img)
    except Exception as e:
        print("Error:", e)

def LAB_to_L(pathLAB, pathL):
    try:
        # 读取LAB图像
        image_lab = cv2.imdecode(np.fromfile(pathLAB, dtype=np.uint8), 1)
        # 将LAB图像拆分为各个通道
        L, _, _ = cv2.split(image_lab)
        # 保存L分量到指定路径
        _, encoded_img = cv2.imencode('.' + pathL.split('.')[-1], L)
        with open(pathL, 'wb') as f:
            f.write(encoded_img)
    except Exception as e:
        print("Error:", e)


def BMP_process(pathImg_RGB,current_time,MCU_results,homepage):
    # 保存bmp图像数据
    global coordinate
    pathImg_RGB_R = storagePath.pathImgData + f"\RGB_R_img\RGB_R_{current_time}.bmp"
    pathImg_RGB_G = storagePath.pathImgData + f"\RGB_G_img\RGB_G_{current_time}.bmp"
    pathImg_RGB_B = storagePath.pathImgData + f"\RGB_B_img\RGB_B_{current_time}.bmp"
    pathImg_GRAY = storagePath.pathImgData + f"\Gray_img\GRAY_{current_time}.bmp"
    pathImg_LAB = storagePath.pathImgData + f"\LAB_img\LAB_{current_time}.bmp"
    pathImg_LAB_L = storagePath.pathImgData + f"\LAB_L_img\LAB_L_{current_time}.bmp"
    # 色彩空间转换
    try:
        RGB_to_R(pathImg_RGB, pathImg_RGB_R)
        RGB_to_G(pathImg_RGB, pathImg_RGB_G)
        RGB_to_B(pathImg_RGB, pathImg_RGB_B)
        RGB_to_GRAY(pathImg_RGB, pathImg_GRAY)
        RGB_to_LAB(pathImg_RGB,pathImg_LAB)
        LAB_to_L(pathImg_LAB,pathImg_LAB_L)
    except Exception as e:
     print(f"Error saving serial data: {e}")
    # 计算中心位置
    centerMax_RGB_R,rightMin_RGB_R,leftMin_RGB_R = calculate_interference_center(pathImg_RGB_R)
    centerMax_RGB_G,rightMin_RGB_G,leftMin_RGB_G = calculate_interference_center(pathImg_RGB_G)
    centerMax_RGB_B,rightMin_RGB_B,leftMin_RGB_B = calculate_interference_center(pathImg_RGB_B)
    centerMax_GRAY,rightMin_GRAY,leftMin_GRAY = calculate_interference_center(pathImg_GRAY)
    # centerLAB = calculate_interference_center(pathImg_LAB)
    centerMax_LAB_L,rightMin_LAB_L,leftMin_LAB_L = calculate_interference_center(pathImg_LAB_L)
    # 平均值
    average_value = (centerMax_RGB_R + centerMax_RGB_G + centerMax_RGB_B + centerMax_GRAY) / 4

    # 计算浓度和压力
    if MCU_results:
        if calibration.a_pressure and calibration.b_pressure is not None:
            pressure = calibration.a_pressure*MCU_results+calibration.b_pressure
        else:
            pressure = None
        if calibration.a_concentration and calibration.b_concentration is not None:
            concentration = calibration.a_concentration*MCU_results+calibration.b_concentration
        else:
            concentration = None
    else:
        pressure = None
        concentration = None
    # 更新首页坐标信息
    update_interference_center(centerMax_RGB_R, centerMax_RGB_G, centerMax_RGB_B, centerMax_GRAY, centerMax_LAB_L,\
                               rightMin_RGB_R, rightMin_RGB_G, rightMin_RGB_B, rightMin_GRAY, rightMin_LAB_L, \
                               leftMin_RGB_R, leftMin_RGB_G, leftMin_RGB_B, leftMin_GRAY, leftMin_LAB_L, \
                               average_value,MCU_results,pressure,concentration,homepage)
    # 启动写入excel进程
    if MCU_results:
        storagePath.start_write_excel_thread(centerMax_RGB_R, centerMax_RGB_G, centerMax_RGB_B, centerMax_GRAY, centerMax_LAB_L,\
                                             rightMin_RGB_R, rightMin_RGB_G, rightMin_RGB_B, rightMin_GRAY, rightMin_LAB_L, \
                                             leftMin_RGB_R, leftMin_RGB_G, leftMin_RGB_B, leftMin_GRAY, leftMin_LAB_L, \
                                             average_value,MCU_results,pressure,concentration)
    coordinate = average_value
    # 更新主页图片
    update_image(homepage,pathImg_RGB, pathImg_RGB_R, pathImg_RGB_G, pathImg_RGB_B, pathImg_GRAY,pathImg_LAB,pathImg_LAB_L)


# 更新首页图片
def update_image(homepage,pathRGB = pathInitImg,pathRGB_R =pathInitImg ,pathRGB_G = pathInitImg,\
                 pathRGB_B = pathInitImg,pathGRAY = pathInitImg,pathLAB = pathInitImg,pathLAB_L = pathInitImg):
    # 在这里进行图片处理，这里只是简单地在图片上画一个红色的矩形框
    # 实际情况中，你需要根据接收到的图片数据进行处理
    pixmap_RGB = QPixmap(pathRGB)
    pixmap_GRAY = QPixmap(pathGRAY)
    pixmap_RGB_R = QPixmap(pathRGB_R)
    pixmap_RGB_G = QPixmap(pathRGB_G)
    pixmap_RGB_B = QPixmap(pathRGB_B)
    pixmap_LAB = QPixmap(pathLAB)
    pixmap_LAB_L = QPixmap(pathLAB_L)
    # 更新主页的图片
    homepage.img_RGB.setPixmap(pixmap_RGB)
    homepage.img_RGB_R.setPixmap(pixmap_RGB_R)
    homepage.img_RGB_G.setPixmap(pixmap_RGB_G)
    homepage.img_RGB_B.setPixmap(pixmap_RGB_B)
    homepage.img_GRAY.setPixmap(pixmap_GRAY)
    homepage.img_LAB.setPixmap(pixmap_LAB)
    homepage.img_LAB_L.setPixmap(pixmap_LAB_L)
# 更新首页的中心坐标
def update_interference_center(centerMax_RGB_R, centerMax_RGB_G, centerMax_RGB_B, centerMax_GRAY,centerMax_LAB_L,\
                               rightMin_RGB_R,rightMin_RGB_G,rightMin_RGB_B, rightMin_GRAY,rightMin_LAB_L,\
                               leftMin_RGB_R,leftMin_RGB_G,leftMin_RGB_B,leftMin_GRAY,leftMin_LAB_L,\
                                average_value,MCU_results,pressure,concentration,homepage):
    # 最大值更新
    homepage.centerMax_RGB_R.setText("{:.4f}".format(centerMax_RGB_R))
    homepage.centerMax_RGB_G.setText("{:.4f}".format(centerMax_RGB_G))
    homepage.centerMax_RGB_B.setText("{:.4f}".format(centerMax_RGB_B))
    homepage.centerMax_GRAY.setText("{:.4f}".format(centerMax_GRAY))
    homepage.centerMax_LAB_L.setText("{:.4f}".format(centerMax_LAB_L))
    # 右侧最小值更新
    homepage.rightMin_RGB_R.setText("{:.4f}".format(rightMin_RGB_R))
    homepage.rightMin_RGB_G.setText("{:.4f}".format(rightMin_RGB_G))
    homepage.rightMin_RGB_B.setText("{:.4f}".format(rightMin_RGB_B))
    homepage.rightMin_GRAY.setText("{:.4f}".format(rightMin_GRAY))
    homepage.rightMin_LAB_L.setText("{:.4f}".format(rightMin_LAB_L))
    # 左侧最小值更新
    homepage.leftMin_RGB_R.setText("{:.4f}".format(leftMin_RGB_R))
    homepage.leftMin_RGB_G.setText("{:.4f}".format(leftMin_RGB_G))
    homepage.leftMin_RGB_B.setText("{:.4f}".format(leftMin_RGB_B))
    homepage.leftMin_GRAY.setText("{:.4f}".format(leftMin_GRAY))
    homepage.leftMin_LAB_L.setText("{:.4f}".format(leftMin_LAB_L))
    # 平均值
    homepage.centerAvr.setText("{:.4f}".format(average_value))
    # MCU计算坐标
    if MCU_results:
        homepage.MCU_results.setText("{:.3f}".format(MCU_results))
    else:
        homepage.MCU_results.setText("None")
    # 浓度＆压力
    if pressure is not None:
        homepage.label_pressure.setText("{:.4f}".format(pressure))
    if concentration is not None:
        homepage.label_ch4_concentration.setText("{:.4f}".format(concentration))



##########################################################################################################
#      高斯拟合
#      使用下位机方法
#      找最大值 并在其右侧搜索最小值
##########################################################################################################
def conv(image):
    filter = np.array([[0.057118, 0.124758, 0.057118],
                       [0.124758, 0.272496, 0.124758],
                       [0.057118, 0.124758, 0.057118]])

    copy_data = np.copy(image)

    height, width = image.shape[:2]

    for i in range(height):
        for j in range(width):
            n_con = 0
            for m in range(3):
                for n in range(3):
                    if (i - n - 1) > 0 and (i - n - 1) < height and (j - m - 1) > 0 and (j - m - 1) < width:
                        n_con += copy_data[i - n - 1][j - m - 1] * filter[m][n]

            n_con = max(0, min(n_con, 255))
            image[i][j] = n_con


def polyfit(x, y, poly_n):
    n = len(x)
    tempx = np.ones(n)
    sumxx = np.zeros(2 * poly_n + 1)
    tempy = np.copy(y)
    sumxy = np.zeros(poly_n + 1)
    ata = np.zeros((poly_n + 1, poly_n + 1))

    for i in range(2 * poly_n + 1):
        for j in range(n):
            sumxx[i] += tempx[j]
            tempx[j] *= x[j]

    for i in range(poly_n + 1):
        for j in range(n):
            sumxy[i] += tempy[j]
            tempy[j] *= x[j]

    for i in range(poly_n + 1):
        for j in range(poly_n + 1):
            ata[i][j] = sumxx[i + j]

    a = gauss_solve(poly_n + 1, ata, sumxy)

    return a


def gauss_solve(n, A, b):
    x = np.zeros(n)

    for k in range(n - 1):
        max_val = abs(A[k, k])
        r = k

        for i in range(k + 1, n):
            if max_val < abs(A[i, i]):
                max_val = abs(A[i, i])
                r = i

        if r != k:
            A[[k, r]] = A[[r, k]]
            b[[k, r]] = b[[r, k]]

        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i, j] -= A[i, k] * A[k, j] / A[k, k]
            b[i] -= A[i, k] * b[k] / A[k, k]

    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i, j] * x[j]
        x[i] /= A[i, i]

    return x


def calculate_interference_center(bmp_path):
    image = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
    # 计算结果初始化
    max_index = 0 # 最大值列索引
    right_min_index = 0 # 最大值右侧最小值的列索引
    left_min_index = 0  # 最大值左侧最小值的列索引
    # 过程变量初始化
    max_averj_value = 0  # 列均值
    max_averj = 0 # 列索引
    averj_value = np.zeros(image.shape[1]) # 列均值

    # 卷积运算
    # 改运算会导致前两行为0
    conv(image)

    # 寻找最大列
    for j in range(image.shape[1]):
        for i in range(image.shape[0]):
            averj_value[j] += image[i][j]
        averj_value[j] /= image.shape[0]  # 列均值
        if averj_value[j] > max_averj_value:
            max_averj_value = averj_value[j] # 当前最大列均值
            max_averj = j    # 当前最大列索引

    maxvalue = np.zeros(image.shape[0])  # 每行最大值
    left_minvalue = np.zeros(image.shape[0])  # 每行最大值左侧最小值
    right_minvalue = np.zeros(image.shape[0])  # 每行最大值右侧最小值


    aver_maxIndex = np.zeros(image.shape[0])   # 每行最大值列索引均值
    aver_left_minIndex = np.zeros(image.shape[0])  # 每行最大值左侧最小值列索引均值
    aver_right_minIndex = np.zeros(image.shape[0])  # 每行最大值右侧最小值列索引均值
    poly_n = 4
    a = np.zeros(poly_n + 1)
    min_value_x_right = image.shape[1] - 1  # 将最大列索引做初值

    # print("最大值所在列",end=":")
    # print(max_averj,end="  ")

    # 找到最大列之后，再分别在每一行最大列索引附近的位置找到最大列
    # 在灰度均值最大的列附近找最大值
    for i in range(image.shape[0]):
        # 每一行的最大列
        for j in range(max_averj - 8, max_averj + 9):
            if image[i][j] >= maxvalue[i]:
                maxvalue[i] = image[i][j]
        # 考虑重复最大值情况
        maxnum = 0 # 统计最大值的个数
        for j in range(max_averj - 8, max_averj + 9):
            if image[i][j] == maxvalue[i]:
                aver_maxIndex[i] += j
                maxnum += 1
        # 最大值所在列索引
        aver_maxIndex[i] /= maxnum

        # 左侧最大值
        for j in range(int(aver_maxIndex[i])):
            if image[i][j] > left_minvalue[i]:
                left_minvalue[i] = image[i][j]
        left_minnum = 0
        for j in range(int(aver_maxIndex[i])):
            if image[i][j] == left_minvalue[i]:
                aver_left_minIndex[i] += j
                left_minnum += 1
        aver_left_minIndex[i] /= left_minnum
        # 右侧侧最大值
        for j in range(int(aver_maxIndex[i]),image.shape[1]):
            if image[i][j] > right_minvalue[i]:
                right_minvalue[i] = image[i][j]
        right_minnum = 0
        for j in range(int(aver_maxIndex[i]),image.shape[1]):
            if image[i][j] == right_minvalue[i]:
                aver_right_minIndex[i] += j
                right_minnum += 1
        aver_right_minIndex[i] /= right_minnum



        # 最大值所在列附件找7个点进行多项式拟合
        px_max = np.arange(int(aver_maxIndex[i] - 3), int(aver_maxIndex[i] + 4))
        px_max = np.clip(px_max, 0, image.shape[1] - 1)  # 防止数据溢出
        py_max = image[i][px_max]
        # 多项式拟合
        a_max = polyfit(px_max, py_max, poly_n)


        # 最大值左侧最小值所在列附件找7个点进行多项式拟合
        px_leftmin = np.arange(int(aver_left_minIndex[i] - 3), int(aver_left_minIndex[i] + 4))
        px_leftmin = np.clip(px_leftmin, 0, image.shape[1] - 1)  # 防止数据溢出
        py_leftmin = image[i][px_leftmin]
        # 多项式拟合
        a_leftmin = polyfit(px_leftmin, py_leftmin, poly_n)

        # 最大值右侧最小值所在列附件找7个点进行多项式拟合
        px_rightmin = np.arange(int(aver_right_minIndex[i] - 3), int(aver_right_minIndex[i] + 4))
        px_rightmin = np.clip(px_rightmin, 0, image.shape[1] - 1)  # 防止数据溢出
        py_rightmin = image[i][px_rightmin]
        # 多项式拟合
        a_rightmin = polyfit(px_rightmin, py_rightmin, poly_n)


        # 找每一行的最大值
        # 以最大值所在列为中心,0.01为步进,带入多项式表达式,找最大值
        max_cv = 0.0
        max_mf = 0
        for mf in np.arange(aver_maxIndex[i] - 1, aver_maxIndex[i] + 1, 0.01):
            cv = a_max[0] + a_max[1] * mf + a_max[2] * mf ** 2 + a_max[3] * mf ** 3 + a_max[4] * mf ** 4
            # 过程中可能发生负数导致结果异常
            cv = abs(cv)

            # print(cv,end=",")
            if cv > max_cv:
                max_cv = cv
                max_mf = mf
        max_index += max_mf

        # 找每一行的最大值左侧最小值
        # 以最大值所在列为中心,0.01为步进,带入多项式表达式,找最大值
        max_cv = 0.0
        max_mf = 0
        for mf in np.arange(aver_left_minIndex[i] - 1, aver_left_minIndex[i] + 1, 0.01):
            cv = a_leftmin[0] + a_leftmin[1] * mf + a_leftmin[2] * mf ** 2 + a_leftmin[3] * mf ** 3 + a_leftmin[4] * mf ** 4
            # 过程中可能发生负数导致结果异常
            cv = abs(cv)

            # print(cv,end=",")
            if cv > max_cv:
                max_cv = cv
                max_mf = mf
        left_min_index += max_mf

        # 找每一行的最大值右侧最小值
        # 以最大值所在列为中心,0.01为步进,带入多项式表达式,找最大值
        max_cv = 0.0
        max_mf = 0
        for mf in np.arange(aver_right_minIndex[i] - 1, aver_right_minIndex[i] + 1, 0.01):
            cv = a_rightmin[0] + a_rightmin[1] * mf + a_rightmin[2] * mf ** 2 + a_rightmin[3] * mf ** 3 + a_rightmin[4] * mf ** 4
            # 过程中可能发生负数导致结果异常
            cv = abs(cv)

            # print(cv,end=",")
            if cv > max_cv:
                max_cv = cv
                max_mf = mf
        right_min_index += max_mf
    # 卷积运算会导致前两行为0
    max_index /= (image.shape[0]-2)
    right_min_index /= (image.shape[0]-2)
    left_min_index /= (image.shape[0] - 2)


    return max_index,right_min_index,left_min_index

##########################################################################################################
# 下面的计算方法结果偏差较大
##########################################################################################################


# 计算图像中心
# def gaussian_blur(image, kernel):
#     return cv2.filter2D(image, -1, kernel)
##########################################################################################################
#      高斯拟合
#      找最小值 并在其右侧搜索最小值
##########################################################################################################

# def gaussian_function(x, A, mu, sigma, b):
#     return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + b
#
# def fit_gaussian(x, y):
#     guess = [np.min(y), np.mean(x), np.std(x), np.max(y)]
#     popt, _ = curve_fit(gaussian_function, x, y, p0=guess, maxfev=2000)
#
#     return popt
#
# def calculate_interference_center(bmp_path):
#     # image = cv2.imread(bmp_path, cv2.IMREAD_UNCHANGED)
#     # 解决中文支持问题
#     image = cv2.imdecode(np.fromfile(bmp_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
#     resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
#
#     custom_kernel = np.array([[0.0751, 0.1238, 0.0751],
#                               [0.1238, 0.2042, 0.1238],
#                               [0.0751, 0.1238, 0.0751]])
#
#     blurred_image = cv2.filter2D(resized_image, -1, custom_kernel)
#
#     column_sums = np.sum(blurred_image, axis=0)
#     min_column_index = np.argmin(column_sums)
#
#     # Find the rightmost minimum value after the maximum value
#     max_column_index = min_column_index + np.argmax(column_sums[min_column_index:])
#
#     # Getting left points for minimum value
#     left_points = column_sums[max_column_index - 3: max_column_index + 4]
#     left_indices = np.arange(max_column_index - 3, max_column_index + 4).astype(float)
#
#     left_params = fit_gaussian(left_indices, left_points)
#     min_value_x_left = left_params[1]
#
#     # Getting right points for minimum value
#     right_points = column_sums[max_column_index: max_column_index + 7]
#     right_indices = np.arange(max_column_index, max_column_index + 7).astype(float)
#
#     right_params = fit_gaussian(right_indices, right_points)
#     min_value_x_right = right_params[1]
#
#     return min_value_x_left, min_value_x_right
##########################################################################################################
#      高斯拟合
#      找最大值 并在其右侧搜索最小值
##########################################################################################################
# def gaussian_function(x, A, mu, sigma, b):
#     return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + b
#
# def fit_gaussian(x, y):
#     guess = [np.max(y), np.mean(x), np.std(x), np.min(y)]
#     popt, _ = curve_fit(gaussian_function, x, y, p0=guess, maxfev=2000)
#
#     return popt
#
# def calculate_interference_center(bmp_path):
#     # image = cv2.imread(bmp_path, cv2.IMREAD_UNCHANGED)
#     # 解决中文支持问题
#     image = cv2.imdecode(np.fromfile(bmp_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
#     resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
#
#     custom_kernel = np.array([[0.0751, 0.1238, 0.0751],
#                               [0.1238, 0.2042, 0.1238],
#                               [0.0751, 0.1238, 0.0751]])
#
#     blurred_image = cv2.filter2D(resized_image, -1, custom_kernel)
#
#     column_sums = np.sum(blurred_image, axis=0)
#     max_column_index = np.argmax(column_sums)
#
#     # Find the rightmost minimum value after the maximum value
#     min_column_index = max_column_index + np.argmin(column_sums[max_column_index:])
#
#     left_points = column_sums[min_column_index - 3: min_column_index + 4]
#     left_indices = np.arange(min_column_index - 3, min_column_index + 4).astype(float)
#
#     left_params = fit_gaussian(left_indices, left_points)
#     min_value_x_left = left_params[1]
#
#     # Getting right points for minimum value
#     right_points = column_sums[min_column_index: min_column_index + 7]
#     right_indices = np.arange(min_column_index, min_column_index + 7).astype(float)
#
#     right_params = fit_gaussian(right_indices, right_points)
#     min_value_x_right = right_params[1]
#
#     return min_value_x_left, min_value_x_right

##########################################################################################################
#      高斯拟合
#      找最大值
##########################################################################################################

# def gaussian_function(x, A, mu, sigma, b):
#     return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + b
#
# def fit_gaussian(x, y):
#     # 初始猜测参数
#     guess = [np.max(y), np.mean(x), np.std(x), np.min(y)]
#     # 使用 curve_fit 进行高斯拟合
#     popt, _ = curve_fit(gaussian_function, x, y, p0=guess)
#     return popt
#
# def calculate_interference_center(bmp_path):
#     # 读取BMP图片
#     image = cv2.imread(bmp_path, cv2.IMREAD_UNCHANGED)
#
#     # 进行双线性插值
#     resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
#
#     # 自定义高斯滤波卷积核
#     custom_kernel = np.array([[0.0751, 0.1238, 0.0751],
#                               [0.1238, 0.2042, 0.1238],
#                               [0.0751, 0.1238, 0.0751]])
#
#     # 进行高斯滤波
#     blurred_image = gaussian_blur(resized_image, custom_kernel)
#
#     # 将图片每一列像素相加
#     column_sums = np.sum(blurred_image, axis=0)
#
#     # 找到最小像素所在列的索引
#     min_column_index = np.argmin(column_sums)
#
#     # 获取最小值所在列的左右各3个点
#     left_points = column_sums[min_column_index - 3: min_column_index + 4]
#     left_indices = np.arange(min_column_index - 3, min_column_index + 4).astype(float)  # 将数据类型转为浮点数
#
#     # 拟合左侧高斯函数
#     left_params = fit_gaussian(left_indices, left_points)
#
#     # 寻找高斯函数最小值
#     min_value_x_left = left_params[1]
#
#     return min_value_x_left


##########################################################################################################
#      多项式拟合
#      找最大值
##########################################################################################################


# def fit_polynomial(x, y, degree=3):
#     # 拟合多项式，加入 full=False 参数确保返回浮点数系数
#     coefficients = np.polyfit(x, y, degree, full=False)
#     polynomial = np.poly1d(coefficients)
#     return polynomial
#
#
# def calculate_interference_center(bmp_path):
#
#     # 读取BMP图片
#     image = cv2.imread(bmp_path, cv2.IMREAD_UNCHANGED)
#
#     # 进行双线性插值
#     resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
#
#     # 自定义高斯滤波卷积核
#     custom_kernel = np.array([[0.0751, 0.1238, 0.0751],
#                               [0.1238, 0.2042, 0.1238],
#                               [0.0751, 0.1238, 0.0751]])
#
#     # 进行高斯滤波
#     blurred_image = gaussian_blur(resized_image, custom_kernel)
#
#     # 将图片每一列像素相加
#     column_sums = np.sum(blurred_image, axis=0)
#
#     # 找到最小像素所在列的索引
#     min_column_index = np.argmin(column_sums)
#
#     # 获取最小值所在列的左右各3个点
#     left_points = column_sums[min_column_index - 3: min_column_index + 4]
#     left_indices = np.arange(min_column_index - 3, min_column_index + 4).astype(float)  # 将数据类型转为浮点数
#
#     # 拟合左侧三次多项式
#     left_polynomial = fit_polynomial(left_indices, left_points)
#
#     # 使用 minimize 函数寻找最小值
#     result = minimize(left_polynomial, x0=left_indices.mean(), bounds=[(left_indices.min(), left_indices.max())])
#
#     min_value_x_left = result.x[0]
#
#
#
#     return min_value_x_left

