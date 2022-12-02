import cv2
import matplotlib.pyplot as plt
import numpy as np


def pltshow(a):
    plt.figure("Image")  # 图像窗口名称
    plt.imshow(a[:, :, ::-1])
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title('image')  # 图像题目
    plt.savefig('./plt_picture.jpg')
    plt.show()


def opencvshow(a):
    cv2.namedWindow("Gosling's report1", cv2.WINDOW_GUI_EXPANDED)  # 已修改窗口名
    cv2.imshow("Lenna", a)
    cv2.waitKey(50000)  # 显示图像的暂停时间设置（单位为毫秒），已修改
    cv2.destroyWindow("Lenna")
    cv2.imwrite('opencv_lenna.png', lenna, (cv2.IMWRITE_PNG_COMPRESSION, 5))


def Opencv_color2gray(path):
    img = cv2.imread(path, 0)  # 直接以灰度图片读取
    opencvshow(img)


def Plt_color2gray(a,path):
    img = cv2.imread(path, 0)
    plt.subplot(1,2,1)
    plt.imshow(a[:, :, ::-1])
    plt.xlabel('origin')
    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.xlabel('grayscale')
    plt.show()


def Opencv_takecolor(a):
    blueImg = a[:, :, 0]
    greenImg = a[:, :, 1]
    redImg = a[:, :, 2]
    imgs = np.hstack([blueImg, greenImg, redImg])
    cv2.imshow("mutil_pic", imgs)
    cv2.waitKey(0)  # 显示图像的暂停时间设置（单位为毫秒），请修改


def Plt_takecolor(a):
    blueImg = a[:, :, 0]
    greenImg = a[:, :, 1]
    redImg = a[:, :, 2]
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(blueImg, cmap='gray')  # 蓝色通道转化为灰度
    plt.subplot(1, 3, 2)
    plt.imshow(greenImg, cmap='gray')  # 绿色通道转化为灰度
    plt.subplot(1, 3, 3)
    plt.imshow(redImg, cmap='gray')  # 红色通道转化为灰度
    plt.show()


if __name__ == "__main__":
    path = "..\\figures\\img\\Lenna.png"
    lenna = cv2.imread(path)  # 请据实修改路径
    # print(type(lenna))  # 返回numpy.ndarray 这个 class
    # pltshow(lenna)  # 用matplotlib作图显示照片
    opencvshow(lenna)  # 用opencv作图显示照片
    # Opencv_color2gray(path)  # 用Opencv做灰度图并保存D
    # Plt_color2gray(path)  # 用matplotlib做灰度图并保存
    # Opencv_takecolor(lenna)  # 用opencv提取三通道并转为灰度图显示出来
    # Plt_takecolor(lenna)  # 用matplotlib提取三通道并转为灰度图显示出来
