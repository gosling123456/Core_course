"""5. 添加和抑制噪声实验
编写程序，首先对图像添加椒盐噪声或高斯噪声，然后对加噪图像进行均值滤波、中值滤波和高斯滤波，查看并分析滤波效果(需附结果展示图像)。
提示：
（1）给图像添加椒盐噪声的Python参考代码：
from flatbuffers.builder import np

def addSaltAndPepper(src, percentage):
    # NoiseImg = src   #使用此语句传递的是地址，程序会出错
    # 在此要使用copy函数，否则src和主程序中的img都会跟着改变
    NoiseImg = src.copy()
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        # 注意需要引入random包
        # 产生[0, src.shape[0] - 1]之间随机整数
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        if random.randint(0, 1) == 0:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


（2）给图像添加高斯噪声的Python参考代码：
def addGaussianNoise(src,means,sigma):
    NoiseImg=src.copy()
    NoiseImg=NoiseImg/NoiseImg.max()
    rows=NoiseImg.shape[0]
    cols=NoiseImg.shape[1]
    for i in range(rows):
        for j in range(cols):
            # Python里使用random.gauss 函数添加高斯噪声
            NoiseImg[i,j]=NoiseImg[i,j]+ random.gauss(means,sigma)
            if  NoiseImg[i,j]< 0:
                 NoiseImg[i,j]=0
            elif  NoiseImg[i,j]>1:
                 NoiseImg[i,j]=1
    NoiseImg=np.uint8(NoiseImg*255)
    return NoiseImg
（3）OpenCV工具包中实现均值滤波、中值滤波和高斯滤波的函数：
cv2.blur(img,(3,3))
cv2.medianBlur(img,3)
cv2.GaussianBlur(img,(3,3),1)"""

# 在这里编写你的代码
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import mpl
font = FontProperties(fname="..\\figures\\font\\simhei.ttf")
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
# plt.axis('off')
# plt.figure(figsize=(5,5))
mpl.rcParams["axes.unicode_minus"] = False
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.5,hspace=0.05)  # 设置子图左右和上下间隔，一般为0.5为佳

def addSaltAndPepper(src, percentage):
    # NoiseImg = src   #使用此语句传递的是地址，程序会出错
    # 在此要使用copy函数，否则src和主程序中的img都会跟着改变
    NoiseImg = src.copy()
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        # 注意需要引入random包
        # 产生[0, src.shape[0] - 1]之间随机整数
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        if random.randint(0, 1) == 0:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


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


# 均值滤波
def average(img):
    return cv2.blur(img, (3, 3))


# 中值滤波
def middle(img):
    return cv2.medianBlur(img, 3)


# 高斯滤波
def Goss(img):
    return cv2.GaussianBlur(img, (3, 3), 1)


if __name__ == '__main__':
    # path = 'work/img/Lenna.png'
    path = "..\\figures\\img\\Lenna.png"
    img = cv2.cvtColor(addSaltAndPepper(cv2.imread(path), 0.2), cv2.COLOR_BGR2RGB)
    plt.subplot(241)
    plt.imshow(img)
    plt.title('添加椒盐噪声',fontsize=10, fontproperties=font)

    plt.subplot(242)
    plt.imshow(average(img))
    plt.title('椒盐噪声的均值滤波',fontsize=10, fontproperties=font)

    plt.subplot(243)
    plt.imshow(middle(img))
    plt.title('椒盐噪声的中值滤波',fontsize=10, fontproperties=font)

    plt.subplot(244)
    plt.imshow(Goss(img))
    plt.title('椒盐噪声的高斯滤波',fontsize=10)

    img1 = cv2.cvtColor(addGaussianNoise(cv2.imread(path), 0, 0.1), cv2.COLOR_BGR2RGB)
    plt.subplot(245)
    plt.imshow(img1)
    plt.title('添加高斯噪声',fontsize=10)

    plt.subplot(246)
    plt.imshow(average(img1))
    plt.title('高斯噪声的均值滤波',fontsize=10)

    plt.subplot(247)
    plt.imshow(middle(img1))
    plt.title('高斯噪声的中值滤波',fontsize=10)

    plt.subplot(248)
    plt.imshow(Goss(img1))
    plt.title('高斯噪声的高斯滤波',fontsize=10)
    plt.show()
