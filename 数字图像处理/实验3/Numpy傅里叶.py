import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)  # 设置子图左右和上下间隔，一般为0.5为佳


def dft_np(path):
    original = cv2.imread(path, 0)  # 转为灰度图

    dft = np.fft.fft2(original)
    dftShift = np.fft.fftshift(dft)     # 将图像中的低频部分移动到图像的中心
    result = np.log(np.abs(dftShift))   # 幅值谱

    phase = np.log(np.angle(dftShift))  # 相位谱

    ishift = np.fft.ifftshift(dftShift)     # 低频部分从图像中心移开
    iImg = np.fft.ifft2(ishift)                 # 傅里叶反变换
    iImg = np.abs(iImg)

    plt.subplot(141), plt.imshow(original, cmap='gray'),plt.title('原图')
    plt.subplot(142), plt.imshow(result, cmap='gray'),plt.title('幅值谱')
    plt.subplot(143), plt.imshow(phase, cmap='gray'), plt.title('相位谱')
    plt.subplot(144), plt.imshow(iImg, cmap='gray'),plt.title('重构图')


if __name__ == '__main__':
    path = '..\\figures\\img\\Lenna.png'
    dft_np(path)
    plt.show()
