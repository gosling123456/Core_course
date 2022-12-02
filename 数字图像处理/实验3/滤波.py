import cv2
import numpy as np
from pylab import mpl
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
mpl.rcParams["axes.unicode_minus"] = False
# font = FontProperties(fname="work/font/simhei.ttf")
font = FontProperties(fname="../figures/font/simhei.ttf")
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.05)  # 设置子图左右和上下间隔，一般为0.5为佳

# 读取图像
# img = cv2.imread('..\\figures\\img\\iris.jpg', 0)
img = cv2.imread('lenna.jpg', 0)


def Low_Pass_Filter(img):
    # 傅里叶变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)
    # 设置低通滤波器
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
    print(mask.shape)
    # 掩膜图像和频谱图像乘积
    f = fshift * mask
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(f)
    iimg = cv2.idft(ishift)
    res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
    return res


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
    img = cv2.imread(path)
    img0 = img[:, :, 0]
    img1 = addGaussianNoise(img, 0, 0.1)[:, :, 0]  # 添加高斯噪声后的灰度图
    img2 = fz(img1)  # 噪声幅值图
    img3 = Low_Pass_Filter(img1)  # 低通滤波图
    img4 = fz(img3)  # 低通滤波幅值图

    plt.subplot(151), plt.imshow(img0, 'gray'), plt.title('原图', fontsize=10, fontproperties=font),plt.axis('off')
    plt.subplot(152), plt.imshow(img1, 'gray'), plt.title('高斯噪声', fontsize=10, fontproperties=font),plt.axis('off')
    plt.subplot(153), plt.imshow(img2, 'gray'), plt.title('高斯噪声幅值谱', fontsize=10, fontproperties=font),plt.axis('off')
    plt.subplot(154), plt.imshow(img4, 'gray'), plt.title('滤波后幅值谱', fontsize=10, fontproperties=font),plt.axis('off')
    plt.subplot(155), plt.imshow(img3, 'gray'), plt.title('低通滤波', fontsize=10, fontproperties=font),plt.axis('off')
    plt.show()
