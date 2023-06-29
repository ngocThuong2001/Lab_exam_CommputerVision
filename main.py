import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('cat.jpg')
img = cv2.resize(img, (200, 200))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img_gray.shape)

def conv2d(input, kernelSize):
    # kích thước hình
    height, width = input.shape
    # tạo kênh số ngẫu nhiên
    kernel = np.random.randn(kernelSize, kernelSize)
    print(kernel)
    # khởi tạo ma trận hứng kết quả
    results = np.zeros((height-kernelSize+1, width - kernelSize + 1))

    for row in range(0, height-kernelSize+1):
        for col in range(0, width - kernelSize + 1):
            # Diện tích khung hình (3,3) quét
            results[row, col] = np.sum(input[row: row + kernelSize, col: col + kernelSize] * kernel)
    return results


img_conv2d = conv2d(img_gray, 3)
plt.imshow(img_conv2d, cmap='gray')
plt.show()