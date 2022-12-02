# #  在下面编写傅里叶变换的实验代码
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from pylab import mpl
#
#
# # 基于numpy
# def dft_np(path):
#     original = cv2.imread(path, 0)  # 转为灰度图
#
#     dft = np.fft.fft2(original)
#     dftShift = np.fft.fftshift(dft)  # 将图像中的低频部分移动到图像的中心
#     result = np.log(np.abs(dftShift))  # 幅值谱
#     phase = np.log(np.angle(dftShift))  # 相位谱
#     ishift = np.fft.ifftshift(dftShift)  # 低频部分从图像中心移开
#     iImg = np.fft.ifft2(ishift)  # 傅里叶反变换
#     iImg = np.abs(iImg)
#
#     plt.subplot(141), plt.imshow(original, cmap='gray'), plt.title('原图')
#     plt.subplot(142), plt.imshow(result, cmap='gray'), plt.title('幅值谱')
#     plt.subplot(143), plt.imshow(phase, cmap='gray'), plt.title('相位谱')
#     plt.subplot(144), plt.imshow(iImg, cmap='gray'), plt.title('重构图')
#     plt.show()
#
#
# # 基于OpenCV
# def dft_CV(path):
#     original = cv2.imread(path, 0)  # 转为灰度图
#     dft = cv2.dft(np.float32(original), flags=cv2.DFT_COMPLEX_OUTPUT)
#     dftShift = np.fft.fftshift(dft)  # 将图像中的低频部分移动到图像的中心
#     amplitude = np.log(cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))
#     phase = np.log(cv2.phase(dftShift[:, :, 0], dftShift[:, :, 1]))
#
#     ishift = np.fft.ifftshift(dftShift)  # 低频部分从图像中心移开
#     iImg = cv2.idft(ishift)  # 傅里叶反变换
#     iImg = cv2.magnitude(iImg[:, :, 0], iImg[:, :, 1])  # 转化为空间域
#
#     plt.subplot(141), plt.imshow(original, cmap='gray'), plt.title('原图')
#     plt.subplot(142), plt.imshow(amplitude, cmap='gray'), plt.title('幅值谱')  # 幅值谱
#     plt.subplot(143), plt.imshow(phase, cmap='gray'), plt.title('相位谱')
#     plt.subplot(144), plt.imshow(iImg, cmap='gray'), plt.title('重构图')
#     plt.show()
#
#
# if __name__ == '__main__':
#     print(111)
#     path = '../figures/img/Lenna.png'
#     print('基于numpy')
#     dft_np(path)
#     print('基于OpenCV')
#     dft_CV(path)
#
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('../figures/img/Lenna.png', 0)

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()