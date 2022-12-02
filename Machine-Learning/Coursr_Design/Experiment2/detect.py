import argparse
import os
import random
import warnings

from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
import cv2
import joblib
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch
import torch.nn.functional as F
from torchvision import models

idx_to_labels = np.load('label/idx_to_labels.npy', allow_pickle=True).item()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('device:', device)

from torchvision import transforms

font = ImageFont.truetype('font/SimHei.ttf', 32)

# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                     ])


def process_frame_pth(img, model):
    model = torch.load(model, map_location=torch.device(device))
    model = model.eval().to(device)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
    img_pil = Image.fromarray(img_rgb)  # array 转 PIL
    input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    pred_logits = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
    _, pred_ids = torch.max(pred_logits, 1)
    img = np.array(img_pil)  # PIL 转 array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB转BGR
    # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，线宽，线型
    cv2.putText(img, idx_to_labels[int(pred_ids)], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return img, idx_to_labels[int(pred_ids)]  # 部署
    # return int(pred_ids)        # 返回标签值


def process_frame_pt(img, model):
    model = joblib.load(model)
    img0 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    hist = cv2.calcHist([img0], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
    di = ['non_food', 'food']
    text = di[model.predict([(hist / 255).flatten()])[0]]
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return img, text


# source == 0
def pic_cam_pt_pth(model, source, save):
    cap = cv2.VideoCapture(0)  # 获取摄像头，传入0表示获取系统默认摄像头
    cap.open(0)  # 打开cap
    text = ''
    while cap.isOpened():  # 无限循环，直到break被触发
        success, frame = cap.read()  # 获取画面
        if not success:
            print('Error')
            break
        if model.endswith('.pth'):
            frame, text = process_frame_pth(frame, model)  ## !!!处理帧函数
            print(frame.shape)
        elif model.endswith('.pt'):
            frame, text = process_frame_pt(frame, model)
        cv2.imshow('111', frame)  # 展示处理后的三通道图像
        if cv2.waitKey(100) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
            break
    cap.release()  # 关闭摄像头
    cv2.destroyAllWindows()  # 关闭图像窗口

# source == imgs
def pic_img_pt_pth(model, source, save):
    pic = [source + '/' + i for i in os.listdir(source)]
    for i in pic:
        img = cv2.imread(i)
        if model.endswith('.pt'):
            im, text = process_frame_pt(img, model)
        elif model.endswith('.pth'):
            im, text = process_frame_pth(img, model)
        cv2.imshow("Predict: %s" % text, im)
        if cv2.waitKey(1000) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
            return
        cv2.destroyWindow("Predict: %s" % text)


# sour == *mp4
def pic_video_pt_pth(model, source, save):
    cap = cv2.VideoCapture(source)
    while True:
        res, image = cap.read()  # 读出图片。res表示是否读取到图片，image表示读取到的每一帧图片
        if not res:
            print('图片提取结束')
            break
        if model.endswith('.pt'):
            im, text = process_frame_pt(image, model)
        elif model.endswith('.pth'):
            im, text = process_frame_pth(image, model)
        cv2.imshow('my_window', im)  # 展示处理后的三通道图像
        if cv2.waitKey(100) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
            break
    cap.release()  # 关闭摄像头
    cv2.destroyAllWindows()  # 关闭图像窗口


def main(model, source, save):
    print(model)
    print(source)
    source = source[0]
    print(save)
    if str(source) == '0':
        pic_cam_pt_pth(model=model, source=source, save=save)
    elif os.path.exists(source):
        pic_img_pt_pth(model=model, source=source, save=save)
    elif source.split('.')[0].lower() in ['mp4', 'avi']:
        pic_video_pt_pth(model=model, source=source, save=save)
    else:
        print('检测文件格式错误！')

                                                                    #'Bayes_best.pt'
if __name__ == '__main__':                                          #'TraditionModels/tree_best.pt'
    parser = argparse.ArgumentParser()                              #'models/resnet18_best-0.985.pth'
    """训练模型"""                                                   #'TraditionModels/tree_best.pt'
    parser.add_argument('--model', nargs='+', type=str, default='models/resnet18_best.pth')  # 权重文件
    parser.add_argument('--source', nargs='+', type=str, default='0')  # 测试文件，0 for camera on PC
    parser.add_argument('--save', nargs='+', type=str, default=0)  # 是否要保存视频
    opt = parser.parse_args()
    main(**vars(opt))


