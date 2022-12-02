# 在这里编辑拉普拉斯锐化和拉普拉斯锐化增强的代码
import cv2
import numpy as np
import matplotlib.pyplot as plt


def Laplace(img, kernel, i, name):  # 传入i的作用是控制画布显示，传入name是为了命名方便
    des_16S = cv2.filter2D(img, ddepth=cv2.CV_16SC1, kernel=kernel, borderType=cv2.BORDER_DEFAULT)
    g = img - des_16S
    g[g < 0] = 0  # 变换后小于0的像素值置0
    g[g > 255] = 255  # 变换后大于0的像素值置255

    # plt.figure(figsize=(10, 14))

    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title('origin')

    plt.subplot(122)
    plt.imshow(g, cmap='gray')
    plt.title(name + ' Laplace')


if __name__ == '__main__':
    img0 = "..\\figures\\img\\Lenna.png"
    # 拉普拉斯算子锐化
    kernel1 = np.asarray([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])  # 定义拉普拉斯算子
    # 拉普拉斯算子锐化增强
    kernel2 = np.asarray([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])  # 定义拉普拉斯算子
    f = cv2.imread(img0, cv2.IMREAD_GRAYSCALE)
    Laplace(f, kernel1, 0, 'kernel1')
    # Laplace(f, kernel2, 2, 'kernel2')
    plt.show()
