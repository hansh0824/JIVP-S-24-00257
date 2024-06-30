import numpy as np
import math
import random
from PIL import Image
import cv2

picname = '912'  # 測試圖片的寫法
ori_pic_name = picname + '.png'  # 原始圖片的檔案名稱
ste_pic_name = picname + '_stego.png'  # 隱藏後的圖片檔案名稱
ori_pic_path = 'D:/Researh/MyProject/cover/' + picname + '.png'  # 原始圖片的讀取路徑
ste_pic_path = 'D:/Researh/MyProject/stego/' + picname + '_stego.png'  # 隱藏後的圖片讀取路徑
ori = Image.open(ori_pic_path)
ori_array = np.array(ori, dtype=np.int32)
ori_row = np.size(ori_array, 0)  # ori_array的行數
ori_col = np.size(ori_array, 1)  # ori_array的列數
stego_array = ori_array.copy()
lsb_array = ori_array.copy()  # 宣告LSB藏密後的陣列
k_val = 4  # 宣告K值，可從這裡調整
TValue = 240  # 宣告門檻值T值，可從這裡調整
count = 0

secret_msg = ''.join(str(random.randint(0, 1)) for i in range(1033097))  # 隨機產生密文


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
def A_MLSB(t):
    global count, k_val, new_interval

    for i in range(ori_row):
        for j in range(0, ori_col, 2):

            # Case 1: Both pixels are less than T
            if ori_array[i][j] < t and ori_array[i][j + 1] < t:
                m = 2 ** k_val
                B = k_val

                # Embed B bits into the first pixel
                stego_array[i][j] = LSB_OPAP(ori_array[i][j], B)

                # Embed the next B bits into the second pixel
                stego_array[i][j + 1] = LSB_OPAP(ori_array[i][j + 1], B)

                # Pixel adjustment for Case 1 => Case 2 switch
                if stego_array[i][j] >= t:
                    stego_array[i][j] -= m
                if stego_array[i][j + 1] >= t:
                    stego_array[i][j + 1] -= m

            # Case 2: At least one of the pixels is greater than or equal to T
            elif ori_array[i][j] >= t or ori_array[i][j + 1] >= t:
                PVD = abs(ori_array[i][j] - ori_array[i][j + 1])
                ori_interval = 0
                if 0 <= PVD <= 255:
                    k_val = 5
                    m = 2 ** k_val
                    B = k_val
                    ori_interval = 1
                    stego_array[i][j] = LSB_OPAP(ori_array[i][j], B)
                    stego_array[i][j + 1] = LSB_OPAP(ori_array[i][j + 1], B)

                elif 96 <= PVD <= 255:
                    k_val = 4
                    m = 2 ** k_val
                    B = k_val
                    ori_interval = 2
                    stego_array[i][j] = LSB_OPAP(ori_array[i][j], B)
                    stego_array[i][j + 1] = LSB_OPAP(ori_array[i][j + 1], B)

                # 以上區間可以增減
                # Pixel adjustment for Case 2 => Case 1 switch
                if stego_array[i][j] < t and stego_array[i][j + 1] < t:
                    stego_array[i][j] += m
                    stego_array[i][j + 1] += m

                # Pixel adjustment for range switch (this will need more logic as per your pseudocode)
                PVD_prime = abs(stego_array[i][j] - stego_array[i][j + 1])
                if 0 <= PVD_prime <= 255:
                    new_interval = 1
                    """
                elif 96 <= PVD_prime <= 255:
                    new_interval = 2
                    """
                # elif 32 <= PVD_prime <= 255:
                # new_interval = 3
                # For now, assuming only simple adjustment

                max_iterations = 1000  # 例如，最多迭代10次
                iterations = 0

                while new_interval != ori_interval and iterations < max_iterations:
                    if new_interval - ori_interval > 1:
                        stego_array[i][j] += m
                        stego_array[i][j + 1] -= m
                        new_interval = judge_area(stego_array[i][j + 1], stego_array[i][j])
                    elif new_interval - ori_interval < -1:
                        stego_array[i][j] -= m
                        stego_array[i][j + 1] += m
                        new_interval = judge_area(stego_array[i][j + 1], stego_array[i][j])
                    iterations += 1
                    if iterations >= max_iterations:
                        break


# 處理溢位問題
def clamp_pixel_value(pixel, m):
    value = max(0 + m, min(255 - m, pixel))
    return value


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
    adjusted_value = clamp_pixel_value(final_pixel, (2 ** k))
    return adjusted_value


A_MLSB(TValue)
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
