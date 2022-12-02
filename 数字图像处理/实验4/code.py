# coding:utf-8
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from pylab import mpl

# 设置显示中文字体
from skimage.feature import hog, local_binary_pattern

mpl.rcParams["font.sans-serif"] = ["SimHei"]
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.05)


def delect(path):
    img = cv2.imread(path, 0)  # 使用此图像，据实修改路径
    # 定义矩阵分别用来存放被阈值T1分开的两部分
    G1 = np.zeros(img.shape, np.uint8)
    G2 = np.zeros(img.shape, np.uint8)
    T1 = np.mean(img)  # 用图像均值做初始阈值
    diff = 255
    T0 = 0.01  # 设置的最大阈值差
    while (diff > T0):
        # THRESH_TOZERO	超过thresh的像素不变, 其他设为0
        # THRESH_TOZERO_INV 与 THRESH_TOZERO 相反
        # _, G1 = cv2.threshold(img, T1, 255, cv2.THRESH_TOZERO_INV)
        # _, G2 = cv2.threshold(img, T1, 255, cv2.THRESH_TOZERO)
        loc1 = np.where(G1 > 0.001)  # 可以对二维数组操作，获得G1 部分非0像素的坐标
        loc2 = np.where(G1 > 0.001)
        # g1 = list(filter(lambda a: a > 0, G1.flatten()))#只能对一维列表筛选，得到的是一个筛选对象
        # g2 = list(filter(lambda a: a > 0, G2.flatten()))
        ave1 = np.mean(G1[loc1])  # 获得G1 部分非0像素的均值
        ave2 = np.mean(G2[loc2])  # 获得G2 部分非0像素的均值
        T2 = (ave1 + ave2) / 2

        diff = abs(T2 - T0)
        T0 = T2
    _, img_result = cv2.threshold(img, T0, 255, cv2.THRESH_BINARY)
    plt.imshow(img_result, 'gray')


def ww(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = skimage.util.random_noise(gray, mode='gaussian', var=0.01, mean=0)  # 添加高斯噪声
    smooth = cv2.GaussianBlur(image, (3, 3), 1.3)  # 高斯滤波（平滑）

    plt.subplot(231)
    plt.imshow(image, "gray")
    plt.title("source image")
    plt.xlabel('噪声图')
    plt.subplot(232)
    plt.hist(image.ravel(), 256)
    plt.title("Histogram")
    plt.xlabel('噪声直方图')
    ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 方法选择为THRESH_OTSU
    plt.subplot(233)
    plt.imshow(th1, "gray")
    plt.title("OTSU,threshold is " + str(ret1))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('OTSU分割图')

    plt.subplot(234)
    plt.imshow(smooth, 'gray')
    plt.title('高斯平滑处理')
    plt.subplot(235)
    plt.hist(smooth.ravel(), 256)
    plt.title("Histogram")
    plt.xlabel('滤波后直方图')
    smooth = (255 * smooth).astype(np.uint8)
    ret1, th1 = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 方法选择为THRESH_OTSU
    plt.subplot(236)
    plt.imshow(th1, "gray")
    plt.title("OTSU,threshold is " + str(ret1))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('OTSU分割图')


class Hog_descriptor():
    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img
        self.img = np.sqrt(img / np.max(img))
        self.img = img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size

    def extract(self):
        height, width = self.img.shape
        # 计算图像的梯度大小和方向
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                # cell内的梯度大小
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                # cell内的梯度方向
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                # 转化为梯度直方图格式
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        # 绘制梯度直方图
        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)

        # block组合、归一化
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image


def LBP_feature(path):
    radius = 3
    n_points = 8 * radius

    image = cv2.imread(path)
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.subplot(131)
    # plt.imshow(image1, cmap='gray')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(image.shape)
    # plt.subplot(132)
    # plt.imshow(image, cmap='gray')
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    # plt.subplot(133)
    plt.imshow(lbp,'gray')
    plt.xlabel('LBP特征')

    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp, density=True, bins=n_bins,
                               range=(0, n_bins))  # 绘制直方图，百分比normed=True，像素个数normed=flase
    # plt.plot(lbp_hist)


if __name__ == '__main__':
    path = r'../figures/img/Lenna.png'
    # ww(path)
    # delect(path)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    hog = Hog_descriptor()
    vector, image = hog.extract()
    # 输出图像的特征向量shape
    print(np.array(vector).shape)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.xlabel('HOG特征')
    #
    LBP_feature(path)
    plt.show()
