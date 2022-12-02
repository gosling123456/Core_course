# 在此处自行编程实现(可调用工具包中函数)：绘制图像直方图、图像直方图均衡化算法。此处脚本代码应包含原图、直方图在处理前后的可视化对比显示输出
# -*- coding: utf-8 -*-



import cv2
from matplotlib import pyplot as plt
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
plt.subplots_adjust(left=None,
                    bottom=None,
                    right=None,
                    top=None,
                    wspace=0.5,
                    hspace=0.5)  # 设置子图左右和上下间隔，一般为0.5为佳
path1 = '..\\figures\\img\\Lenna.png'
path2 = '..\\figures\\img\\iris.jpg'

plt.subplot(241)  # 三个参数分别代表的是行、列、序号，序号从左往右从上到下增加，从1开始
plt.imshow(cv2.imread(path1)[:, :, ::-1])  # 用OpenCV读取图片，并将其转化为RGB格式，转化完成后用matplotlib展示出来
plt.xlabel('lenna原图',fontsize=10)  # 设置横坐标名称和字体

plt.subplot(242)
img = cv2.imread(path1, 0)
plt.hist(img.ravel(), 255, [0, 256])  # plt.hist()是matplotlib内置画图函数，可以直接统计并绘制直方图
plt.title("lenna的直方图", fontsize=10)

plt.subplot(243)
plt.hist(cv2.imread(path1, cv2.IMREAD_GRAYSCALE).ravel(), 256)  # 转为均衡图并显示
plt.title('lenna均衡化直方图', fontsize=10)

plt.subplot(244)
plt.imshow(cv2.equalizeHist(cv2.imread(path1, cv2.IMREAD_GRAYSCALE)),'gray')
plt.xlabel('lenna均衡灰度图',fontsize=10)

plt.subplot(245)
plt.imshow(cv2.imread(path2)[:, :, ::-1])  # 用OpenCV读取图片，并将其转化为RGB格式，转化完成后用matplotlib展示出来
plt.xlabel('iris原图',fontsize=10)  # 设置横坐标名称和字体

plt.subplot(246)
img1 = cv2.imread(path2, 0)
plt.hist(img1.ravel(), 255, [0, 256])
plt.title("iris的直方图", fontsize=10)

plt.subplot(247)
plt.hist(cv2.imread(path2, cv2.IMREAD_GRAYSCALE).ravel(), 256)
plt.title('iris的均衡化直方图', fontsize=10)

plt.subplot(248)
plt.imshow(cv2.equalizeHist(cv2.imread(path2, cv2.IMREAD_GRAYSCALE)),'gray')
plt.xlabel('iris均衡灰度图',fontsize=10)

plt.savefig('.\\Graph.jpg')
plt.show()