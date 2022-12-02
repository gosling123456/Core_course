import numpy as np
import cv2
from matplotlib import pyplot as plt
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
img = cv2.imread(r'../figures/img/paopao.jpg', 0)
def delect(path):
    eps = 0.01  # 精度
    iry = np.array(img)
    r, c = img.shape
    avg = 0

    T = np.mean(img)  # 用图像均值做初始阈值
    dis = 255
    while dis >= eps:
        G1, G2, cnt1, cnt2 = 0, 0, 0, 0  # G1用来记录像素大于阈值的点的灰度和，G2用来记录像素小于阈值的点的灰度和
        for i in range(r):               # cnt1用来记录大于阈值的点数，cnt2用来记录小于阈值点数
            for j in range(c):
                if iry[i][j] >= T:
                    G1 += iry[i][j]; cnt1 += 1
                else:
                    G2 += iry[i][j]; cnt2 += 1
        T2 = (int(G1 / cnt1) + int(G2 / cnt2)) / 2
        dis = abs(T2 - T)
        T = T2
    new_img = np.zeros((r, c), np.uint8)  # 初始化new_img
    _, img_result = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
    # 添加绘制结果展示代码
    plt.subplot(221)
    plt.imshow(img, 'gray')
    plt.xlabel('原图')
    plt.subplot(223)
    plt.imshow(img_result, 'gray')
    plt.xlabel('全局阈值化处理的图像')
    # 绘制原图直方图并显示最佳阈值
    plt.subplot(222)
    plt.hist(img.ravel(), 256)
    plt.title('hist')
    plt.axvline(T)  # 绘制最佳阈值分割线
    plt.xlabel('原图直方图')
    plt.text(160, 10000, "Best Threshold:{}".format(T), size=10, alpha=0.8)

    plt.subplot(224)
    plt.hist(img_result.ravel(),256)
    plt.title('hist')
    plt.xlabel('全局阈值化处理的直方图')
plt.show()