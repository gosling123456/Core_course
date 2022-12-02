# 请在此处编写图像空域滤波(卷积)的代码,下面的示例代码需要修改
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.font_manager import FontProperties
from pylab import mpl
# 设置显示中文字体
font = FontProperties(fname="..\\figures\\font\\simhei.ttf")
# 设置正常显示符号
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
img = cv2.imread("..\\figures\\img\\Lenna.png")  # 使用此图像，需据实修改路径
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 设置滤波核
kernel1 = 1 / 16 * np.array([[1, 2, 1],
                             [2, 4, 2],
                             [1, 2, 1]])

kernel2 = 1 / 9 * np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])

kernel3 = 1 / 10 * np.array([[1, 1, 1],
                             [1, 2, 1],
                             [1, 1, 1]])

kernel4 = np.array([[-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]])
kernel = [kernel1, kernel2, kernel3, kernel4]
# 使用OpenCV的卷积函数
for i in range(0, 4):
    ImgSmoothed = cv2.filter2D(img, -1, kernel[i], borderType=cv2.BORDER_DEFAULT)
    # 展示结果
    plt.figure()
    plt.subplot(121)
    # plt.axis("off")
    plt.title('原图')
    plt.imshow(img)
    plt.subplot(122)
    # plt.axis("off")
    plt.title('用卷积核' + str(i + 1) + '做空域变换后的图像')
    plt.imshow(ImgSmoothed)
plt.show()
