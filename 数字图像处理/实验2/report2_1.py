# # -*- coding: utf-8 -*-
# from pylab import mpl
# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
#
# # 设置显示中文字体
# mpl.rcParams["font.sans-serif"] = ["SimHei"]
# # 设置正常显示符号
# mpl.rcParams["axes.unicode_minus"] = False
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)  # 设置子图左右和上下间隔，一般为0.5为佳
#
# fig = plt.figure(figsize=(11, 4))
#
#
# def gammaTransform(imgDir):
#     imgGray = cv.imread(imgDir, cv.IMREAD_GRAYSCALE)  # OpenCV读取灰度图
#     imgGrayNorm = imgGray / 255  # 像素值归一化
#     gamma = 2  # gamma系数
#     dst = np.power(imgGrayNorm, gamma)  # gamma变换
#     plt.subplot(1, 6, 6)
#     plt.imshow(dst, 'gray')
#     plt.xlabel('gamma变换', fontsize='10')
#
#
# def origion(img):
#     plt.subplot(1, 6, 1)
#     plt.imshow(img, 'gray')
#     plt.xlabel('原图', fontsize=10)
#
#
# # 增加图像亮度
# # 注意需要使用_cv.add(),不能直接x+y
# def bright(img):
#     plt.subplot(1, 6, 2)
#     res1 = np.uint8(np.clip((cv.add(1 * img, 100)), 0, 255))  # 每个像素点加100点像素值
#     plt.imshow(res1, 'gray')
#     plt.xlabel('灰度加100', fontsize=10)
#
#
# # 注意需要使用cv.add(),不能直接x+y
# def dark(img):
#     plt.subplot(1, 6, 3)
#     res2 = np.uint8(np.clip((cv.add(1 * img, -100)), 0, 255))  # 每个像素点加-100点像素值
#     plt.imshow(res2, 'gray')
#     plt.xlabel('灰度减100', fontsize=10)
#
#
# # 增加图像对比度
# def low_contrast(img):
#     plt.subplot(1, 6, 4)
#     res3 = np.uint8(np.clip((cv.add(0.01 * img, 0)), 0, 255))  # 整个像素缩小到原来的1/100
#     plt.imshow(res3, 'gray')
#     plt.xlabel('降低对比度', fontsize=10)
#
#
# # 直方图均衡化
# def balance(img):
#     plt.subplot(1, 6, 5)
#     plt.imshow(cv.equalizeHist(cv.imread('..\\figures\\img\\lenna.png', cv.IMREAD_GRAYSCALE)), 'gray')  # 转为均衡图并显示
#     plt.xlabel('均衡化图像', fontsize=10)
#     gammaTransform('..\\figures\\img\\lenna.png')
#
#
# if __name__ == '__main__':
#     path = "..\\figures\\img\\lenna.png"
#     img = cv.imread(path, 0)
#     origion(img)
#     bright(img)
#     dark(img)
#     low_contrast(img)
#     balance(img)
#     gammaTransform(path)
#     plt.show()
# -*- coding: utf-8 -*-
import cv2 as cv
import matplotlib
import numpy as np
from pylab import mpl
import matplotlib.font_manager
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置显示中文字体
font = FontProperties(fname="..\\figures\\font\\simhei.ttf")
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)  # 设置子图左右和上下间隔，一般为0.5为佳
fig = plt.figure(figsize=(11, 4))


def origion(img):
    plt.subplot(1, 5, 1)
    plt.imshow(img, 'gray')
    plt.xlabel('原图', fontsize=10, fontproperties=font)


# 增加图像亮度
# 注意需要使用_cv.add(),不能直接x+y
def bright(img):
    plt.subplot(1, 5, 2)
    res1 = np.uint8(np.clip((cv.add(1 * img, 100)), 0, 255))  # 每个像素点加100点像素值
    plt.imshow(res1, 'gray')
    plt.xlabel('灰度加100', fontsize=10, fontproperties=font)


# 注意需要使用cv.add(),不能直接x+y
def dark(img):
    plt.subplot(1, 5, 3)
    res2 = np.uint8(np.clip((cv.add(1 * img, -100)), 0, 255))  # 每个像素点加-100点像素值
    plt.imshow(res2, 'gray')
    plt.xlabel('灰度减100', fontsize=10, fontproperties=font)


# 增加图像对比度
def low_contrast(img):
    plt.subplot(1, 5, 4)
    res3 = np.uint8(np.clip((cv.add(0.01 * img, 0)), 0, 255))  # 整个像素缩小到原来的1/100
    plt.imshow(res3, 'gray')
    plt.xlabel('降低对比度', fontsize=10, fontproperties=font)


# 直方图均衡化
def balance(path):
    plt.subplot(1, 5, 5)
    plt.imshow(cv.equalizeHist(cv.imread(path, cv.IMREAD_GRAYSCALE)), 'gray')  # 转为均衡图并显示
    plt.xlabel('均衡化图像', fontsize=10, fontproperties=font)


if __name__ == '__main__':
    path = "..\\figures\\img\\Lenna.png"
    img = cv.imread(path, 0)
    origion(img)
    bright(img)
    dark(img)
    low_contrast(img)
    balance(path)
    plt.show()