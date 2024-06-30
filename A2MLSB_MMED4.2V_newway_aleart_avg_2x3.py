import numpy as np
import math
import random
from PIL import Image
import cv2

# 調整差值的區間 並且將目標值減去周邊像素的平均在做判斷  ~~
picname = 'f16'
ori_pic_name = picname + '.tif'
ste_pic_name = picname + '_stego.tif'
ori_pic_path = 'D:/Researh/MyProject/cover/' + picname + '.tif'
ste_pic_path = 'D:/Researh/MyProject/2x3/stego/' + picname + '_stego.tif'
ori = Image.open(ori_pic_path)
ori_array = np.array(ori, dtype=np.int32)
ori_row = np.size(ori_array, 0)  # ori_array的行數
ori_col = np.size(ori_array, 1)  # ori_array的列數
stego_array = ori_array.copy()
lsb_array = ori_array.copy()  # 宣告LSB藏密後的陣列
k_val = 4  # 宣告K值，可從這裡調整
MED_k_val = 4
TValue = 240  # 宣告門檻值T值，可從這裡調整
count = 0

secret_msg = ''.join(str(random.randint(0, 1)) for i in range(10000000))  # 隨機產生密文

for p in range(509, 512):
    print("", end="\n")
    for q in range(0, 2):
        print(p+1, end="\t")
        print(q+1, end="\t")
        print(ori_array[p][q])


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
    for i in range(0, ori_row - 1, 2):
        for j in range(0, ori_col - 2, 3):
            block = stego_array[i:i + 2, j:j + 3]

            # 做[1,0]與[0,2]的MMED
            MED_PVD1 = abs(modified_MED(block[0, 0], block[1, 1], block[0, 1], block[1, 0]) - (
                    block[1, 1] + block[0, 1] + block[0, 0]) / 3)

            MED_PVD2 = abs(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]) - (
                    block[1, 2] + block[1, 1] + block[0, 1]) / 3)

            if 0 <= MED_PVD1 < 2:
                block[1, 0] = LSB_OPAP(modified_MED(block[0, 0], block[1, 1], block[0, 1], block[1, 0]), MED_k_val + 1)
            elif 2 <= MED_PVD1 < 5:
                block[1, 0] = LSB_OPAP(modified_MED(block[0, 0], block[1, 1], block[0, 1], block[1, 0]), MED_k_val)
            elif 5 <= MED_PVD1 < 8:
                block[1, 0] = LSB_OPAP(modified_MED(block[0, 0], block[1, 1], block[0, 1], block[1, 0]), MED_k_val)
            elif 8 <= MED_PVD1 < 64:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val)
            else:
                block[1, 0] = LSB_OPAP(modified_MED(block[0, 0], block[1, 1], block[0, 1], block[1, 0]), MED_k_val)

            if 0 <= MED_PVD2 < 2:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val + 1)
            elif 2 <= MED_PVD2 < 5:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val)
            elif 5 <= MED_PVD2 < 8:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val)
            elif 8 <= MED_PVD2 < 64:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val)
            else:
                block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val)
            # block[0, 2] = LSB_OPAP(modified_MED(block[0, 1], block[1, 2], block[1, 1], block[0, 2]), MED_k_val + 1)
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

            stego_array[i:i + 2, j:j + 3] = block


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


# OPAP函數
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
