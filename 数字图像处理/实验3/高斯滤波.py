import cv2
import numpy as np
from pylab import mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
mpl.rcParams["axes.unicode_minus"] = False
# font = FontProperties(fname="work/font/simhei.ttf")
font = FontProperties(fname="../figures/font/simhei.ttf")

def addGaussianNoise(src, means, sigma):
    image = np.array(src / 255, dtype=float)
    noise = np.random.normal(means, sigma, image.shape)
    gauss_noise = image + noise
    if gauss_noise.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    gauss_noise = np.clip(gauss_noise, low_clip, 1.0)
    gauss_noise = np.uint8(gauss_noise * 255)
    return gauss_noise


def fz(img):  # 求幅值
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))))


if __name__ == '__main__':
    path = '..\\figures\\img\\iris.jpg'
    img0 = cv2.imread(path)  # 原图
    img1 = addGaussianNoise(img0, 0, 0.1)[:, :, 0]  # 噪声图
    img2 = fz(img1)  # 噪声幅值图
    img3 = cv2.GaussianBlur(img0, (3, 3), 1.3)[:, :, 0]  # 高斯滤波
    img4 = fz(img3)  # 高斯滤波幅值谱

    plt.subplot(151),plt.imshow(img0[:, :, 0], 'gray'), plt.title('原图', fontsize=10, fontproperties=font), plt.axis('off')  # 原图
    plt.subplot(152),plt.imshow(img1, 'gray'), plt.title('噪声图', fontsize=10, fontproperties=font), plt.axis('off')  # 噪声图
    plt.subplot(153),plt.imshow(img2, 'gray'), plt.title('噪声幅值图', fontsize=10, fontproperties=font), plt.axis('off')  # 噪声幅值图
    plt.subplot(154),plt.imshow(img4, 'gray'), plt.title('滤波后幅值图', fontsize=10, fontproperties=font), plt.axis('off')  # 滤波后幅值图
    plt.subplot(155),plt.imshow(img3, 'gray'), plt.title('高斯滤波图', fontsize=10, fontproperties=font), plt.axis('off')  # 高斯滤波图
    plt.show()
