import numpy as np
import math
import random
from PIL import Image
import cv2

# 調整差值的區間 並且將目標值減去周邊像素的平均在做判斷  ~~
# 改為4X4區間，將空白處用PVD來藏
ori_pic_name = 'lena.tif'  # 原始圖片的檔案名稱
ste_pic_name = 'lena_stego.tif'  # 隱藏後的圖片檔案名稱
ori_pic_path = 'D:/Researh/MyProject/cover/lena.tif'  # 原始圖片的讀取路徑
ste_pic_path = 'D:/Researh/MyProject/stego/lena_stego.tif'  # 隱藏後的圖片讀取路徑
ori = Image.open(ori_pic_path)
ori_array = np.array(ori, dtype=np.int32)
ori_row = np.size(ori_array, 0)  # ori_array的行數
ori_col = np.size(ori_array, 1)  # ori_array的列數
stego_array = ori_array.copy()
lsb_array = ori_array.copy()  # 宣告LSB藏密後的陣列
k_val = 3  # 宣告K值，可從這裡調整
MED_k_val = 3
TValue = 240  # 宣告門檻值T值，可從這裡調整
count = 0  # 計算藏密量的變數

secret_msg = ''.join(str(random.randint(0, 1)) for i in range(10000000))  # 隨機產生密文


# 10進位轉2進位
def dec_to_bin(dec):
    binn = []
    while dec / 2 > 0:
        binn.append(str(dec % 2))
        dec = dec // 2
    binn.reverse()
    return ''.join(binn)


# 將密文由2進位變成10進位
def bin_to_dec(ref):
    binary_str = ref
    decimal_int = int(binary_str, 2)
    return decimal_int


def judge_area(a, b):
    pvd = abs(a - b)
    if 0 <= pvd <= 15:
        print(1)
        return 1
    elif 16 <= pvd <= 31:
        print(2)
        return 2
    elif 32 <= pvd <= 255:
        print(3)
        return 3


# 藏密函數:
def A2MLSB_MMED(t):
    global count, k_val, new_interval
    for i in range(0, ori_row, 4):
        for j in range(0, ori_col, 4):
            block = stego_array[i:i + 4, j:j + 4]
            # 做[1,0]與[0,2]的MMED
            MED_PVD1 = abs(modified_MED(block[0, 0], block[1, 1], block[0, 1], block[1, 0]) - (
                    block[1, 1] + block[0, 1] + block[0, 0]) / 3)

            MED_PVD2 = abs(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]) - (
                    block[1, 2] + block[1, 1] + block[0, 1]) / 3)

            if 0 <= MED_PVD1 < 2:
                block[1, 0] = LSB_OPAP(modified_MED(block[0, 0], block[1, 1], block[0, 1], block[1, 0]), MED_k_val + 2)
            elif 2 <= MED_PVD1 < 5:
                block[1, 0] = LSB_OPAP(modified_MED(block[0, 0], block[1, 1], block[0, 1], block[1, 0]), MED_k_val + 1)
            elif 5 <= MED_PVD1 < 8:
                block[1, 0] = LSB_OPAP(modified_MED(block[0, 0], block[1, 1], block[0, 1], block[1, 0]), MED_k_val + 2)
            elif 8 <= MED_PVD1 < 64:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val + 2)
            else:
                block[1, 0] = LSB_OPAP(modified_MED(block[0, 0], block[1, 1], block[0, 1], block[1, 0]), MED_k_val+2)

            if 0 <= MED_PVD2 < 2:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val + 2)
            elif 2 <= MED_PVD2 < 5:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val + 1)
            elif 5 <= MED_PVD2 < 8:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val + 2)
            elif 8 <= MED_PVD2 < 64:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val + 2)
            else:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val+2)

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
            # stego_array[i:i + 2, j:j + 3] = block

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
            # ------------------------------------------------------------------------------------
            # 做[3,1]與[2,3]的MMED
            MED_PVD3 = abs(modified_MED(block[2, 1], block[3, 2], block[2, 2], block[3, 1]) - (
                    block[3, 2] + block[2, 2] + block[2, 1]) / 3)

            MED_PVD4 = abs(modified_MED(block[2, 2], block[3, 3], block[3, 2], block[2, 3]) - (
                    block[3, 3] + block[3, 2] + block[2, 2]) / 3)

            if 0 <= MED_PVD3 < 2:
                block[3, 1] = LSB_OPAP(modified_MED(block[2, 1], block[3, 2], block[2, 2], block[3, 1]), MED_k_val + 2)
            elif 2 <= MED_PVD3 < 5:
                block[3, 1] = LSB_OPAP(modified_MED(block[2, 1], block[3, 2], block[2, 2], block[3, 1]), MED_k_val + 1)
            elif 5 <= MED_PVD3 < 8:
                block[3, 1] = LSB_OPAP(modified_MED(block[2, 1], block[3, 2], block[2, 2], block[3, 1]), MED_k_val + 2)
            elif 8 <= MED_PVD3 < 64:
                block[3, 1] = LSB_OPAP(modified_MED(block[2, 1], block[3, 2], block[2, 2], block[3, 1]), MED_k_val + 2)
            else:
                block[3, 1] = LSB_OPAP(modified_MED(block[2, 1], block[3, 2], block[2, 2], block[3, 1]), MED_k_val+2)

            if 0 <= MED_PVD4 < 2:
                block[2, 3] = LSB_OPAP(modified_MED(block[2, 2], block[3, 3], block[3, 2], block[2, 3]), MED_k_val + 2)
            elif 2 <= MED_PVD4 < 5:
                block[2, 3] = LSB_OPAP(modified_MED(block[2, 2], block[3, 3], block[3, 2], block[2, 3]), MED_k_val + 1)
            elif 5 <= MED_PVD4 < 8:
                block[2, 3] = LSB_OPAP(modified_MED(block[2, 2], block[3, 3], block[3, 2], block[2, 3]), MED_k_val + 2)
            elif 8 <= MED_PVD4 < 64:
                block[2, 3] = LSB_OPAP(modified_MED(block[2, 2], block[3, 3], block[3, 2], block[2, 3]), MED_k_val + 2)
            else:
                block[2, 3] = LSB_OPAP(modified_MED(block[2, 2], block[3, 3], block[3, 2], block[2, 3]), MED_k_val+2)

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
            # stego_array[i:i + 2, j:j + 3] = block

            # [3,2]與[3,3]的A2MLSB
            # Case 1: Both pixels are less than T
            if block[3][2] < t and block[3][3] < t:
                m = 2 ** k_val
                B = k_val
                # Embed B bits into the first pixel
                block[3][2] = LSB_OPAP(block[3][2], B)
                # Embed the next B bits into the second pixel
                block[3][3] = LSB_OPAP(block[3][3], B)
                # Pixel adjustment for Case 1 => Case 2 switch
                if block[3][2] >= t:
                    block[3][2] -= m
                if block[3][3] >= t:
                    block[3][3] -= m
                # Case 2: At least one of the pixels is greater than or equal to T
                elif block[3][2] >= t or block[3][3] >= t:
                    PVD = abs(block[3][2] - block[3][3])
                    ori_interval = 0
                    if 0 <= PVD <= 95:
                        k_val = 5
                        m = 2 ** k_val
                        B = k_val
                        ori_interval = 1
                        block[3][2] = LSB_OPAP(block[3][2], B)
                        block[3][3] = LSB_OPAP(block[3][3], B)
                    elif 96 <= PVD <= 255:
                        k_val = 4
                        m = 2 ** k_val
                        B = k_val
                        ori_interval = 2
                        block[3][2] = LSB_OPAP(block[3][2], B)
                        block[3][3] = LSB_OPAP(block[3][3], B)
                    # 以上區間可以增減
                    # Pixel adjustment for Case 2 => Case 1 switch
                    if block[3][2] < t and block[3][3] < t:
                        block[3][2] += m
                        block[3][3] += m
                    # Pixel adjustment for range switch (this will need more logic as per your pseudocode)
                    PVD_prime = abs(block[3][2] - block[3][3])
                    if 0 <= PVD_prime <= 95:
                        new_interval = 1
                    elif 96 <= PVD_prime <= 255:
                        new_interval = 2
                    # For now, assuming only simple adjustment
                    max_iterations = 1000  # 例如，最多迭代10次
                    iterations = 0
                    while new_interval != ori_interval and iterations < max_iterations:
                        if new_interval - ori_interval > 1:
                            block[3][2] += m
                            block[3][3] -= m
                            new_interval = judge_area(block[3][3], block[3][2])
                        elif new_interval - ori_interval < -1:
                            block[3][2] -= m
                            block[3][3] += m
                            new_interval = judge_area(block[3][3], block[3][2])
                        iterations += 1

            block[2][0],block[3][0] = pvd(block[2][0],block[3][0],count)
            block[0][3],block[1][3] = pvd(block[0][3], block[1][3],count)

            stego_array[i:i + 4, j:j + 4] = block



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


# OPAP_LSB函數
def LSB_OPAP(ori_pixel, k):
    global count
    if count + k > len(secret_msg):
        return ori_pixel  # 或者，您可以選擇其他的錯誤處理方法
    b = int(secret_msg[count:count + k], 2)
    count += k
    lsb_pixel = ori_pixel - np.remainder(ori_pixel, 2 ** k) + b
    d = ori_pixel - lsb_pixel
    if d > 2 ** (k - 1) and 0 <= lsb_pixel + 2 ** k <= 255:
        final_pixel = lsb_pixel + 2 ** k
    elif d < -2 ** (k - 1) and 0 <= lsb_pixel - 2 ** k <= 255:
        final_pixel = lsb_pixel - 2 ** k
    else:
        final_pixel = lsb_pixel
    return final_pixel


def pvd(x, y, n):  # x是原始像素 y是原始像素的下一個像素 d為差值 d_new為新差值 n是count(計算藏密量的變數)
    final_pixel1 = 0  # 產生的新像素值
    final_pixel2 = 0  # 產生的新像素值
    global count
    d = abs(x - y)
    if d >= 0 and d <= 2:
        d_new = 0 + bin_to_dec(secret_msg[n:n + 4])
        n += 4
    elif d >= 2 and d <= 5:
        d_new = 8 + bin_to_dec(secret_msg[n:n + 5])
        n += 5
    elif d >= 6 and d <= 8:
        d_new = 16 + bin_to_dec(secret_msg[n:n + 5])
        n += 5
    elif d >= 9 and d <= 16:
        d_new = 32 + bin_to_dec(secret_msg[n:n + 5])
        n += 5
    elif d >= 17 and d <= 255:
        d_new = 32 + bin_to_dec(secret_msg[n:n + 4])
        n += 4


    if x >= y and d_new > d:
        final_pixel1 = x + math.ceil(abs(d_new - d) / 2)
        final_pixel2 = y - math.floor(abs(d_new - d) / 2)
    elif x < y and d_new > d:
        final_pixel1 = x - math.ceil(abs(d - d) / 2)
        final_pixel2 = y + math.floor(abs(d_new - d) / 2)
    elif x >= y and d_new <= d:
        final_pixel1 = x - math.ceil(abs(d_new - d) / 2)
        final_pixel2 = y + math.floor(abs(d_new - d) / 2)
    elif x < y and d_new <= d:
        final_pixel1 = x + math.ceil(abs(d_new - d) / 2)
        final_pixel2 = y - math.floor(abs(d_new - d) / 2)

    count=n

    return final_pixel1, final_pixel2


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
