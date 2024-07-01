import numpy as np
import math
import random
from PIL import Image
import cv2

# 調整差值的區間 並且將目標值減去周邊像素的平均在做判斷  ~~
# 以此版本做3x3的測試，剩餘的像素以MMED來藏
# 最後加入溢位處理+-m
picname = '43'  # 測試圖片的寫法
ori_pic_name = picname + '.tif'  # 原始圖片的檔案名稱
ste_pic_name = picname + '_stego.tif'  # 隱藏後的圖片檔案名稱
ori_pic_path = 'D:/Researh/MyProject/Basecover_categorized/' + picname + '.png'  # 原始圖片的讀取路徑
ste_pic_path = 'D:/Researh/MyProject/Basestego_categorized/' + picname + '_stego.png'  # 隱藏後的圖片讀取路徑
ori = Image.open(ori_pic_path)
ori_array = np.array(ori, dtype=np.int32)
ori_row = np.size(ori_array, 0)  # ori_array的行數
ori_col = np.size(ori_array, 1)  # ori_array的列數
stego_array = ori_array.copy()
lsb_array = ori_array.copy()  # 宣告LSB藏密後的陣列
k_val = 4 # 宣告K值，可從這裡調整
MED_k_val = 2
TValue = 240  # 宣告門檻值T值，可從這裡調整
count = 0

secret_msg = ''.join(str(random.randint(0, 1)) for i in range(1500000))  # 隨機產生密文


# 10進位轉2進位
def dec_to_bin(dec):
    binn = []
    while dec / 2 > 0:
        binn.append(str(dec % 2))
        dec = dec // 2
    binn.reverse()
    return ''.join(binn)


def judge_area(a, b):
    pvd = abs(a - b)
    if 0 <= pvd <= 95:
        print(1)
        return 1
    elif 96 <= pvd <= 255:
        print(2)
        return 2
    # elif 32 <= pvd <= 255:
    # print(3)
    # return 3


# 將密文由2進位變成10進位
def bin_to_dec(ref):
    binary_str = ref
    decimal_int = int(binary_str, 2)
    return decimal_int


# 藏密函數:
def A2MLSB_MMED(t):
    global count, k_val, new_interval

    # 剩餘的使用MMED
    # 右上的2x3的MMED
    for i in range(0, 510, 2):
        for j in range(509, 512, 3):

            r_block_MMED = stego_array[i:i + 2, j:j + 3]
            # modified_MED(block_MMED[0, 1], block_MMED[1, 0], block_MMED[0, 0], block_MMED[1, 1])  # 第一個值
            # modified_MED(block_MMED[0, 2], block_MMED[1, 1], block_MMED[0, 1], block_MMED[1, 2])  # 第二個值
            # 第一個值做OPAP
            remain_MMED_rightside1 = abs(
                modified_MED(r_block_MMED[0, 1], r_block_MMED[1, 0], r_block_MMED[0, 0], r_block_MMED[1, 1]) - (
                        r_block_MMED[0, 1] + r_block_MMED[1, 0] + r_block_MMED[0, 0]) / 3)
            if 0 <= remain_MMED_rightside1 < 2:
                r_block_MMED[1, 1] = LSB_OPAP(
                    modified_MED(r_block_MMED[0, 1], r_block_MMED[1, 0], r_block_MMED[0, 0], r_block_MMED[1, 1]),
                    MED_k_val + 2)
            elif 2 <= remain_MMED_rightside1 < 5:
                r_block_MMED[1, 1] = LSB_OPAP(
                    modified_MED(r_block_MMED[0, 1], r_block_MMED[1, 0], r_block_MMED[0, 0], r_block_MMED[1, 1]),
                    MED_k_val + 2)
            elif 5 <= remain_MMED_rightside1 < 8:
                r_block_MMED[1, 1] = LSB_OPAP(
                    modified_MED(r_block_MMED[0, 1], r_block_MMED[1, 0], r_block_MMED[0, 0], r_block_MMED[1, 1]),
                    MED_k_val + 2)
            elif 8 <= remain_MMED_rightside1 < 64:
                r_block_MMED[1, 1] = LSB_OPAP(
                    modified_MED(r_block_MMED[0, 1], r_block_MMED[1, 0], r_block_MMED[0, 0], r_block_MMED[1, 1]),
                    MED_k_val + 2)
            else:
                r_block_MMED[1, 1] = LSB_OPAP(
                    modified_MED(r_block_MMED[0, 1], r_block_MMED[1, 0], r_block_MMED[0, 0], r_block_MMED[1, 1]),
                    MED_k_val+ 2)

            # 第二個值做OPAP
            remain_MMED_rightside2 = abs(
                modified_MED(r_block_MMED[0, 2], r_block_MMED[1, 1], r_block_MMED[0, 1], r_block_MMED[1, 2]) - (
                        r_block_MMED[1, 1] + r_block_MMED[1, 0] + r_block_MMED[1, 2]) / 3)
            if 0 <= remain_MMED_rightside2 < 2:
                r_block_MMED[1, 2] = LSB_OPAP(
                    modified_MED(r_block_MMED[0, 2], r_block_MMED[1, 1], r_block_MMED[0, 1], r_block_MMED[1, 2]),
                    MED_k_val + 2)
            elif 2 <= remain_MMED_rightside2 < 5:
                r_block_MMED[1, 2] = LSB_OPAP(
                    modified_MED(r_block_MMED[0, 2], r_block_MMED[1, 1], r_block_MMED[0, 1], r_block_MMED[1, 2]),
                    MED_k_val + 2)
            elif 5 <= remain_MMED_rightside2 < 8:
                r_block_MMED[1, 2] = LSB_OPAP(
                    modified_MED(r_block_MMED[0, 2], r_block_MMED[1, 1], r_block_MMED[0, 1], r_block_MMED[1, 2]),
                    MED_k_val + 2)
            elif 8 <= remain_MMED_rightside2 < 64:
                r_block_MMED[1, 2] = LSB_OPAP(
                    modified_MED(r_block_MMED[0, 2], r_block_MMED[1, 1], r_block_MMED[0, 1], r_block_MMED[1, 2]),
                    MED_k_val + 2)
            else:
                r_block_MMED[1, 2] = LSB_OPAP(
                    modified_MED(r_block_MMED[0, 2], r_block_MMED[1, 1], r_block_MMED[0, 1], r_block_MMED[1, 2]),
                    MED_k_val+ 2)

            stego_array[i:i + 2, j:j + 3] = r_block_MMED

    # 左下的2x3的MMED
    for i in range(509, 512, 3):
        for j in range(0, 512, 2):
            l_block_MMED = stego_array[i:i + 3, j:j + 2]
            # modified_MED(block_MMED[0, 1], block_MMED[1, 0], block_MMED[0, 0], block_MMED[1, 1])  # 第一個值
            # modified_MED(block_MMED[1, 1], block_MMED[2, 0], block_MMED[1, 0], block_MMED[2, 1])  # 第二個值
            # 第一個值做OPAP
            remain_MMED_left1 = abs(
                modified_MED(l_block_MMED[0, 1], l_block_MMED[1, 0], l_block_MMED[0, 0], l_block_MMED[1, 1]) - (
                        l_block_MMED[0, 1] + l_block_MMED[1, 0] + l_block_MMED[0, 0]) / 3)
            if 0 <= remain_MMED_left1 < 2:
                l_block_MMED[1, 1] = LSB_OPAP(
                    modified_MED(l_block_MMED[0, 1], l_block_MMED[1, 0], l_block_MMED[0, 0], l_block_MMED[1, 1]),
                    MED_k_val + 2)
            elif 2 <= remain_MMED_left1 < 5:
                l_block_MMED[1, 1] = LSB_OPAP(
                    modified_MED(l_block_MMED[0, 1], l_block_MMED[1, 0], l_block_MMED[0, 0], l_block_MMED[1, 1]),
                    MED_k_val + 2)
            elif 5 <= remain_MMED_left1 < 8:
                l_block_MMED[1, 1] = LSB_OPAP(
                    modified_MED(l_block_MMED[0, 1], l_block_MMED[1, 0], l_block_MMED[0, 0], l_block_MMED[1, 1]),
                    MED_k_val + 2)
            elif 8 <= remain_MMED_left1 < 64:
                l_block_MMED[1, 1] = LSB_OPAP(
                    modified_MED(l_block_MMED[0, 1], l_block_MMED[1, 0], l_block_MMED[0, 0], l_block_MMED[1, 1]),
                    MED_k_val + 2)
            else:
                l_block_MMED[1, 1] = LSB_OPAP(
                    modified_MED(l_block_MMED[0, 1], l_block_MMED[1, 0], l_block_MMED[0, 0], l_block_MMED[1, 1]),
                    MED_k_val+ 2)

            # 第二個值做OPAP
            remain_MMED_left2 = abs(
                modified_MED(l_block_MMED[1, 1], l_block_MMED[2, 0], l_block_MMED[1, 0], l_block_MMED[2, 1]) - (
                        l_block_MMED[1, 1] + l_block_MMED[2, 0] + l_block_MMED[1, 0]) / 3)
            if 0 <= remain_MMED_left2 < 2:
                l_block_MMED[2, 1] = LSB_OPAP(
                    modified_MED(l_block_MMED[1, 1], l_block_MMED[2, 0], l_block_MMED[1, 0], l_block_MMED[2, 1]),
                    MED_k_val + 2)
            elif 2 <= remain_MMED_left2 < 5:
                l_block_MMED[2, 1] = LSB_OPAP(
                    modified_MED(l_block_MMED[1, 1], l_block_MMED[2, 0], l_block_MMED[1, 0], l_block_MMED[2, 1]),
                    MED_k_val + 2)
            elif 5 <= remain_MMED_left2 < 8:
                l_block_MMED[2, 1] = LSB_OPAP(
                    modified_MED(l_block_MMED[1, 1], l_block_MMED[2, 0], l_block_MMED[1, 0], l_block_MMED[2, 1]),
                    MED_k_val + 2)
            elif 8 <= remain_MMED_left2 < 64:
                l_block_MMED[2, 1] = LSB_OPAP(
                    modified_MED(l_block_MMED[1, 1], l_block_MMED[2, 0], l_block_MMED[1, 0], l_block_MMED[2, 1]),
                    MED_k_val + 2)
            else:
                l_block_MMED[2, 1] = LSB_OPAP(
                    modified_MED(l_block_MMED[1, 1], l_block_MMED[2, 0], l_block_MMED[1, 0], l_block_MMED[2, 1]),
                    MED_k_val+ 2)

            stego_array[i:i + 3, j:j + 2] = l_block_MMED

    for i in range(0, ori_row - 2, 3):
        for j in range(0, ori_col - 2, 3):
            block = stego_array[i:i + 3, j:j + 3]

            # 做[0,2]的MMED
            MED_PVD1 = abs(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]) - (
                    block[1, 2] + block[1, 1] + block[0, 1]) / 3)

            if 0 <= MED_PVD1 < 2:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val + 3)
            elif 2 <= MED_PVD1 < 5:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val + 3)
            elif 5 <= MED_PVD1 < 8:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val + 2)
            elif 8 <= MED_PVD1 < 64:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val + 2)
            else:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val + 2)

            # [0,0]與[0,1]的A2MLSB
            # Case 1: Both pixels are less than T
            if block[0][0] < t and block[0][1] < t:
                m = 2 ** k_val
                B = k_val
                # Embed B bits into the first pixel
                block[0][0] = LSB_OPAP(block[0][0], B)
                # Embed the next B bits into the second pixel
                block[0][1] = LSB_OPAP(block[0][1], B)
                # Pixel adjustment for Case 1 => Case 2 switch
                if block[0][0] >= t:
                    block[0][0] -= m
                if block[0][1] >= t:
                    block[0][1] -= m
                # Case 2: At least one of the pixels is greater than or equal to T
            elif block[0][0] >= t or block[0][1] >= t:
                PVD = abs(block[0][0] - block[0][1])
                ori_interval = 0
                if 0 <= PVD <= 95:
                    k_val = 5
                    m = 2 ** k_val
                    B = k_val
                    ori_interval = 1
                    block[0][0] = LSB_OPAP(block[0][0], B)
                    block[0][1] = LSB_OPAP(block[0][1], B)
                elif 96 <= PVD <= 255:
                    k_val = 4
                    m = 2 ** k_val
                    B = k_val
                    ori_interval = 2
                    block[0][0] = LSB_OPAP(block[0][0], B)
                    block[0][1] = LSB_OPAP(block[0][1], B)
                    # 以上區間可以增減
                    # Pixel adjustment for Case 2 => Case 1 switch
                    if block[0][0] < t and block[0][1] < t:
                        block[0][0] += m
                        block[0][1] += m
                    # Pixel adjustment for range switch (this will need more logic as per your pseudocode)
                    PVD_prime = abs(block[0][0] - block[0][1])
                    if 0 <= PVD_prime <= 95:
                        new_interval = 1
                    elif 96 <= PVD_prime <= 255:
                        new_interval = 2
                    # For now, assuming only simple adjustment
                    max_iterations = 1000  # 例如，最多迭代10次
                    iterations = 0
                    while new_interval != ori_interval and iterations < max_iterations:
                        if new_interval - ori_interval > 1:
                            block[0][0] += m
                            block[0][1] -= m
                            new_interval = judge_area(block[0][1], block[0][0])
                        elif new_interval - ori_interval < -1:
                            block[0][0] -= m
                            block[0][1] += m
                            new_interval = judge_area(block[0][1], block[0][0])
                        iterations += 1
            # [1,1]與[1,2]的A2MLSB
            # Case 1: Both pixels are less than T
            if block[1][1] < t and block[1][2] < t:
                m = 2 ** k_val
                B = k_val
                # Embed B bits into the first pixel
                block[1][1] = LSB_OPAP(block[1][1], B)
                # Embed the next B bits into the second pixel
                block[1][2] = LSB_OPAP(block[1][2], B)
                # Pixel adjustment for Case 1 => Case 2 switch
                if block[1][1] >= t:
                    block[1][1] -= m
                if block[1][2] >= t:
                    block[1][2] -= m
            # Case 2: At least one of the pixels is greater than or equal to T
            elif block[1][1] >= t or block[1][2] >= t:
                PVD = abs(block[1][1] - block[1][2])
                ori_interval = 0
                if 0 <= PVD <= 95:
                    k_val = 5
                    m = 2 ** k_val
                    B = k_val
                    ori_interval = 1
                    block[1][1] = LSB_OPAP(block[1][1], B)
                    block[1][2] = LSB_OPAP(block[1][2], B)
                elif 96 <= PVD <= 255:
                    k_val = 4
                    m = 2 ** k_val
                    B = k_val
                    ori_interval = 2
                    block[1][1] = LSB_OPAP(block[1][1], B)
                    block[1][2] = LSB_OPAP(block[1][2], B)
                # 以上區間可以增減
                # Pixel adjustment for Case 2 => Case 1 switch
                if block[1][1] < t and block[1][2] < t:
                    block[1][1] += m
                    block[1][2] += m
                # Pixel adjustment for range switch (this will need more logic as per your pseudocode)
                PVD_prime = abs(block[1][1] - block[1][2])
                if 0 <= PVD_prime <= 95:
                    new_interval = 1
                elif 96 <= PVD_prime <= 255:
                    new_interval = 2
                # For now, assuming only simple adjustment
                max_iterations = 1000  # 例如，最多迭代10次
                iterations = 0
                while new_interval != ori_interval and iterations < max_iterations:
                    if new_interval - ori_interval > 1:
                        block[1][1] += m
                        block[1][2] -= m
                        new_interval = judge_area(block[1][2], block[1][1])
                    elif new_interval - ori_interval < -1:
                        block[1][1] -= m
                        block[1][2] += m
                        new_interval = judge_area(block[1][2], block[1][1])
                    iterations += 1

            # [1,0]與[2,0]的A2MLSB
            # Case 1: Both pixels are less than T
            if block[1][0] < t and block[2][0] < t:
                m = 2 ** k_val
                B = k_val
                # Embed B bits into the first pixel
                block[1][0] = LSB_OPAP(block[1][0], B)
                # Embed the next B bits into the second pixel
                block[2][0] = LSB_OPAP(block[2][0], B)
                # Pixel adjustment for Case 1 => Case 2 switch
                if block[1][0] >= t:
                    block[1][0] -= m
                if block[2][0] >= t:
                    block[2][0] -= m
                # Case 2: At least one of the pixels is greater than or equal to T
            elif block[1][0] >= t or block[2][0] >= t:
                PVD = abs(block[1][0] - block[2][0])
                ori_interval = 0
                if 0 <= PVD <= 95:
                    k_val = 5
                    m = 2 ** k_val
                    B = k_val
                    ori_interval = 1
                    block[1][0] = LSB_OPAP(block[1][0], B)
                    block[2][0] = LSB_OPAP(block[2][0], B)
                elif 96 <= PVD <= 255:
                    k_val = 4
                    m = 2 ** k_val
                    B = k_val
                    ori_interval = 2
                    block[1][0] = LSB_OPAP(block[1][0], B)
                    block[2][0] = LSB_OPAP(block[2][0], B)
                    # 以上區間可以增減
                    # Pixel adjustment for Case 2 => Case 1 switch
                    if block[1][0] < t and block[2][0] < t:
                        block[1][0] += m
                        block[2][0] += m
                    # Pixel adjustment for range switch (this will need more logic as per your pseudocode)
                    PVD_prime = abs(block[1][0] - block[2][0])
                    if 0 <= PVD_prime <= 95:
                        new_interval = 1
                    elif 96 <= PVD_prime <= 255:
                        new_interval = 2
                    # For now, assuming only simple adjustment
                    max_iterations = 1000  # 例如，最多迭代10次
                    iterations = 0
                    while new_interval != ori_interval and iterations < max_iterations:
                        if new_interval - ori_interval > 1:
                            block[1][0] += m
                            block[2][0] -= m
                            new_interval = judge_area(block[1][2], block[1][1])
                        elif new_interval - ori_interval < -1:
                            block[1][0] -= m
                            block[2][0] += m
                            new_interval = judge_area(block[2][0], block[1][0])
                        iterations += 1

            # [2,1]與[2,2]的A2MLSB
            # Case 1: Both pixels are less than T
            if block[2][1] < t and block[2][2] < t:
                m = 2 ** k_val
                B = k_val
                # Embed B bits into the first pixel
                block[2][1] = LSB_OPAP(block[2][1], B)
                # Embed the next B bits into the second pixel
                block[2][2] = LSB_OPAP(block[2][2], B)
                # Pixel adjustment for Case 1 => Case 2 switch
                if block[2][1] >= t:
                    block[2][1] -= m
                if block[2][2] >= t:
                    block[2][2] -= m
            # Case 2: At least one of the pixels is greater than or equal to T
            elif block[2][1] >= t or block[2][2] >= t:
                PVD = abs(block[2][1] - block[2][2])
                ori_interval = 0
                if 0 <= PVD <= 95:
                    k_val = 5
                    m = 2 ** k_val
                    B = k_val
                    ori_interval = 1
                    block[2][1] = LSB_OPAP(block[2][1], B)
                    block[2][2] = LSB_OPAP(block[2][2], B)
                elif 96 <= PVD <= 255:
                    k_val = 4
                    m = 2 ** k_val
                    B = k_val
                    ori_interval = 2
                    block[2][1] = LSB_OPAP(block[2][1], B)
                    block[2][2] = LSB_OPAP(block[2][2], B)
                # 以上區間可以增減
                # Pixel adjustment for Case 2 => Case 1 switch
                if block[2][1] < t and block[2][2] < t:
                    block[2][1] += m
                    block[2][2] += m
                # Pixel adjustment for range switch (this will need more logic as per your pseudocode)
                PVD_prime = abs(block[2][1] - block[2][2])
                if 0 <= PVD_prime <= 95:
                    new_interval = 1
                elif 96 <= PVD_prime <= 255:
                    new_interval = 2
                # For now, assuming only simple adjustment
                max_iterations = 1000  # 例如，最多迭代10次
                iterations = 0
                while new_interval != ori_interval and iterations < max_iterations:
                    if new_interval - ori_interval > 1:
                        block[2][1] += m
                        block[2][2] -= m
                        new_interval = judge_area(block[2][2], block[2][1])
                    elif new_interval - ori_interval < -1:
                        block[2][1] -= m
                        block[2][2] += m
                        new_interval = judge_area(block[2][2], block[2][1])
                    iterations += 1

            stego_array[i:i + 3, j:j + 3] = block


# 邊緣偵測法
def modified_MED(a, b, c, x):
    T = 4
    if c >= a & c >= b:
        if a > b:
            x = b
            return x
        elif a < b:
            x = a
            return x
    elif c <= a & c <= b:
        if a > b:
            x = a
            return x
        elif a < b:
            x = b
            return x
    elif b <= c <= a or a <= c <= b:
        if abs(c - ((a + b) / 2)) < T:
            x = (a + b + c) / 3
            return x
    else:
        x = a + b - c
        return x

    return x


# 處理溢位問題
def clamp_pixel_value(pixel, m):
    value = max(m, min(255 - m, pixel))
    return value


# OPAP函數
def LSB_OPAP(ori_pixel, k):
    global count, final_pixel
    if count + k > len(secret_msg):
        return ori_pixel  # 或者可以選擇其他的錯誤處理方法
    b = int(secret_msg[count:count + k], 2)
    count += k
    lsb_pixel = ori_pixel - np.remainder(ori_pixel, 2 ** k) + b
    d = ori_pixel - lsb_pixel
    if 2 ** (k - 1) < d and lsb_pixel + (2 ** k) <= 255:
        final_pixel = lsb_pixel + 2 ** k
    elif -(2 ** (k - 1)) > d and 0 <= lsb_pixel - (2 ** k) <= 255:
        final_pixel = lsb_pixel - 2 ** k
    else:
        final_pixel = lsb_pixel

    adjusted_value = clamp_pixel_value(final_pixel, (2 ** k))

    return adjusted_value


A2MLSB_MMED(TValue)

ec = count / (512 * 512)
print("藏入的bit數量 : " + str(count))
print("EC = " + str(ec) + "bpp")

# 顯示PSNR 以及處理過的圖像
psnrcount = 0
psnrtmp = 0
for i in range(0, ori_row, 1):
    for j in range(0, ori_col, 1):
        psnrtmp = (stego_array[i][j] - ori_array[i][j]) ** 2.0
        psnrcount = psnrcount + psnrtmp

mse = ((1.0 / (ori_row * ori_col)) * psnrcount)  # 公式ß
if mse == 0:
    print('MSE is zero, cannot calculate PSNR')
else:
    psnr = 10.0 * (math.log10(255.0 ** 2.0 / mse))
    print('MSE: ', mse)
    print('PSNR: ', psnr)

# 儲存成tiff檔
stego_image = Image.fromarray(np.uint8(stego_array))
stego_image.save(ste_pic_path)

# 讀取藏密後的tiff檔
img_saved = cv2.imread(ste_pic_path)
cv2.imshow(ste_pic_path, img_saved)

# 讀取藏密前的tiff檔
img_saved2 = cv2.imread(ori_pic_path)
cv2.imshow(ori_pic_path, img_saved2)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
# 載入藏密後的圖像
ste = Image.open(ste_pic_path)
stego_array = np.array(ste, dtype=np.int16)
ste_row = np.size(stego_array, 0)  # stego_array的行數
ste_col = np.size(stego_array, 1)  # stego_array的列數
"""
