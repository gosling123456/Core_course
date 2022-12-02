import cv2
import math
import numpy as np
import matplotlib.pyplot as plt



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


if __name__ == '__main__':
    path = '..\\figures\\img\\iris.jpg'
    img = cv2.imread(path)
    img0 = addGaussianNoise(img, 0, 0.1)[:, :, 0]  # 给图像加高斯噪声
    f = np.fft.fft2(img0)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum0 = 20 * np.log(1 + np.abs(fshift))
    r = 5  # 截止频率
    # 进行理想低通滤波
    [m, n] = fshift.shape
    H = np.zeros((m, n), dtype=complex)  # 滤波核
    for i in range(m):
        for j in range(n):
            d = math.sqrt((i - m / 2) ** 2 + (j - n / 2) ** 2)
            if d <= r:
                H[i, j] = 1
            else:
                H[i, j] = 0
    G = fshift * H  # 理想低通滤波
    magnitude_spectrum1 = 20 * np.log(1 + np.abs(G))  # 理想低通滤波后的幅值谱
    f1 = np.fft.ifftshift(G)
    iimg = cv2.idft(f1)
    img1 = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])  # 重构图像
    plt.imshow(magnitude_spectrum1)
    plt.show()
