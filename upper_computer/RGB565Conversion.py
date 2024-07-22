# 模块功能: RGB565转RGB888
################################################################################################
import numpy as np
import sys
import struct
# 以下是自定义模块
import storagePath
def hex_to_bytes(hex_string):
    return bytes.fromhex(hex_string)


def RGB2BMP(rgb_buffer, nWidth, nHeight, fp1):
    # BMP 文件头
    bfType = b'BM'
    bfSize = 54 + 3 * nWidth * nHeight
    bfReserved1 = 0
    bfReserved2 = 0
    bfOffBits = 54

    # BMP 信息头
    biSize = 40
    biWidth = nWidth
    biHeight = nHeight
    biPlanes = 1
    biBitCount = 24
    biCompression = 0
    biSizeImage = 3 * nWidth * nHeight
    biXPelsPerMeter = 2835
    biYPelsPerMeter = 2835
    biClrUsed = 0
    biClrImportant = 0

    # 写入 BMP 文件头
    fp1.write(bfType)
    fp1.write(np.uint32(bfSize))
    fp1.write(np.uint16(bfReserved1))
    fp1.write(np.uint16(bfReserved2))
    fp1.write(np.uint32(bfOffBits))

    # 写入 BMP 信息头
    fp1.write(np.uint32(biSize))
    fp1.write(np.int32(biWidth))
    fp1.write(np.int32(biHeight))
    fp1.write(np.uint16(biPlanes))
    fp1.write(np.uint16(biBitCount))
    fp1.write(np.uint32(biCompression))
    fp1.write(np.uint32(biSizeImage))
    fp1.write(np.int32(biXPelsPerMeter))
    fp1.write(np.int32(biYPelsPerMeter))
    fp1.write(np.uint32(biClrUsed))
    fp1.write(np.uint32(biClrImportant))

    # 颜色顺序调整为BGR
    pVisit = bytearray()
    for i in range(0, len(rgb_buffer), 4):
        hex_str = rgb_buffer[i:i + 4].decode('utf-8')
        byte_data = hex_to_bytes(hex_str)

        R = (np.uint16(struct.unpack('<H', byte_data[:2])[0]) & 0x1F) * 255 // 31
        G = ((np.uint16(struct.unpack('<H', byte_data[:2])[0]) >> 5) & 0x3F) * 255 // 63
        B = ((np.uint16(struct.unpack('<H', byte_data[:2])[0]) >> 11) & 0x1F) * 255 // 31

        pVisit.extend([B, G, R])

    # 补齐行末的字节
    row_padding = (4 - ((3 * nWidth) % 4)) % 4
    padding_bytes = bytearray([0] * row_padding)

    # 写入图像数据
    for i in range(nHeight):
        start_idx = i * nWidth * 3
        end_idx = (i + 1) * nWidth * 3
        fp1.write(pVisit[start_idx:end_idx])
        fp1.write(padding_bytes)


def RGB565toRGB888(inputFilename,outputFilename):
    # if len(sys.argv) != 3:
    #     print("Usage: python script.py <input_filename.txt> <output_filename.bmp>")
    #     sys.exit(-1)

    # inputFilename = sys.argv[1]
    # outputFilename = sys.argv[2]

    try:
        nWidth = 320
        nHeight = 40
        total = nWidth * nHeight * 3
        with open(inputFilename, "r") as inputFile:
            hex_data = inputFile.read().replace('\n', '').replace(' ', '').encode('utf-8')
            # 保证 hex_data 长度是 4 的倍数
            if len(hex_data) % 4 != 0:
                hex_data += b'00'
            print(f"Opened file: {inputFilename}")






        with open(outputFilename, "wb") as outputFile:
            RGB2BMP(hex_data, nWidth, nHeight, outputFile)
            print(f"Total = {total}")

    except FileNotFoundError:
        print(f"Error: File {inputFilename} not found.")

