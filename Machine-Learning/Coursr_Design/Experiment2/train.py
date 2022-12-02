import argparse
import datetime
import random
import time
from pathlib import Path
import cv2
import joblib
import numpy
import warnings
import pandas as pd
from torch.optim import lr_scheduler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
# 忽略烦人的红色提示
import warnings

warnings.filterwarnings("ignore")
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim

ROOT = os.getcwd()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class traditional:
    def __init__(self):  # 构造函数
        pass
    def mk_dataset(  # 图像数据化
            self,
            train=ROOT + '//training',
            test=ROOT + '//evaluation',
            validation=ROOT + '//validation'
    ):
        train_path = os.path.join(os.getcwd(), train)
        test_path = os.path.join(os.getcwd(), test)
        vail_path = os.path.join(os.getcwd(), validation)

        label = {'food': 0, 'non_food': 1}
        di = os.listdir(train_path)
        X_train_path = []
        Y_train = []

        for i in di:
            for j in list(os.walk(os.path.join(train_path, i)))[-1][-1]:
                X_train_path.append(os.path.join(train_path, i) + '\\' + j)
                Y_train.append(label[i])

        print('训练集路径', train_path)
        print('训练集图像数量', len(X_train_path))
        print('类别个数', len(di))
        print('各类别名称', di)

        di = os.listdir(test_path)
        X_test_path = []
        Y_test = []
        for i in di:
            for j in list(os.walk(os.path.join(test_path, i)))[-1][-1]:
                X_test_path.append(os.path.join(test_path, i) + '\\' + j)
                Y_test.append(label[i])

        print('\n')
        print('测试集路径', test_path)
        print('测试集图像数量', len(X_test_path))
        print('类别个数', len(di))
        print('各类别名称', di)

        di = os.listdir(vail_path)
        V_path = []
        V_label = []
        for i in di:
            for j in list(os.walk(os.path.join(vail_path, i)))[-1][-1]:
                V_path.append(os.path.join(vail_path, i) + '\\' + j)
                V_label.append(label[i])
        print('\n')
        print('验证集路径', vail_path)
        print('验证集图像数量', len(V_path))
        print('类别个数', len(di))
        print('各类别名称', di)

        XX_train = []  # 训练集
        for i in X_train_path:
            image = cv2.imdecode(np.fromfile(i, dtype=np.uint8), cv2.IMREAD_COLOR)
            # 图像像素大小一致
            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
            # 计算图像直方图并存储至X数组
            hist = cv2.calcHist([img], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
            XX_train.append(((hist / 255).flatten()))

        XX_test = []  # 测试集
        for i in X_test_path:
            # 不使用imread，而是用imdecode以识别中文路径
            image = cv2.imdecode(np.fromfile(i, dtype=np.uint8), cv2.IMREAD_COLOR)
            # 图像像素大小一致
            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
            # 计算图像直方图并存储至X数组
            hist = cv2.calcHist([img], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
            XX_test.append(((hist / 255).flatten()))

        V_data = []  # 验证集
        for i in V_path:
            # 不使用imread，而是用imdecode以识别中文路径
            image = cv2.imdecode(np.fromfile(i, dtype=np.uint8), cv2.IMREAD_COLOR)
            # 图像像素大小一致
            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
            # 计算图像直方图并存储至X数组
            hist = cv2.calcHist([img], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
            V_data.append(((hist / 255).flatten()))

        c = list(zip(XX_train, Y_train))
        random.shuffle(c)
        XX_train, Y_train = zip(*c)
        XX_train, Y_train = list(XX_train), list(Y_train)

        c = list(zip(XX_test, Y_test))
        random.shuffle(c)
        XX_test, Y_test = zip(*c)
        XX_test, Y_test = list(XX_test), list(Y_test)

        c = list(zip(V_data, V_label))
        random.shuffle(c)
        V_data, V_label = zip(*c)
        V_data, V_label = list(V_data), list(V_label)

        Y_train = numpy.array(Y_train)
        Y_test = numpy.array(Y_test)
        V_label = numpy.array(V_label)

        """"降维"""
        from sklearn.decomposition import PCA
        X = XX_train  # 导入数据，维度为65536
        pca = PCA(n_components=1000)  # 降到100维
        pca.fit(X)  # 训练

        XX_train = pca.fit_transform(XX_train)
        XX_test = pca.fit_transform(XX_test)
        V_data = pca.fit_transform(V_data)

        print('数据初始化完成！')
        return X_train_path, X_test_path, XX_train, Y_train, XX_test, Y_test, V_data, V_label, di


    def model_training(self):
        models = {
            # 'SVM': SVC(),
            'tree': DecisionTreeClassifier(),
            'KNN': KNeighborsClassifier(n_neighbors=2),
            'Bayes': BernoulliNB()
        }
        epochs = 500
        #  定义调参的超参数
        C = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8]  # SVM的惩罚系数
        K = [2, 3, 5, 6, 7, 8]  # KNN的近邻参数
        SVM_ACC = {}
        KNN_ACC = {}
        for model in models.keys():
            Acc = 0
            if model == 'SVM':
                for epoch in range(1, epochs + 1):
                    clf = SVC(C=C[epoch // 100]).fit(XX_train, Y_train)
                    acc = round(clf.score(V_data, V_label), 2)
                    if acc > Acc:
                        joblib.dump( clf,model + '_best.pt')
                        print(model + '准确度新高：', acc)
                        Acc = acc
                    if epoch % 100 == 0:
                        print("惩罚系数C=%.1f在验证集上准确率:" % C[(epoch // 100)], acc)
                        KNN_ACC[str(epoch // 100)] = acc

            elif model == 'KNN':
                for epoch in range(epochs):
                    clf = KNeighborsClassifier(n_neighbors=K[epoch // 100]).fit(XX_train, Y_train)
                    acc = round(clf.score(V_data, V_label), 2)
                    if acc > Acc:
                        joblib.dump( clf,model + '_best.pt')
                        print(model+'准确度新高：',acc)
                        Acc = acc
                    if epoch % 100 == 0:
                        print("邻居数K=%.1f在验证集上准确率:" % K[epoch // 100], acc)
                        KNN_ACC[str(epoch // 100)] = acc
            else:
                for epoch in range(epochs):
                    clf = models[model].fit(XX_train, Y_train)
                    acc = round(clf.score(V_data, V_label), 2)
                    print('验证集上准确率：',acc)
                    if acc > Acc:
                        joblib.dump(model + '_best.pt', clf)
                        print(model + '准确度新高：', acc)
                        Acc = acc
            predictions_labels = clf.predict(XX_test)
            print(u'预测结果:')
            print(predictions_labels)
            print(u'算法评价:')
            print(classification_report(Y_test, predictions_labels))

        plt.subplot(1, 2, 1)
        plt.plot([float(i) for i in list(SVM_ACC.keys())], list(SVM_ACC.values()), c='red')
        plt.xlabel('惩罚因子C')
        plt.ylabel('验证集上的准确率')
        plt.title('SVM调参图')
        plt.savefig('SVM.jpg')

        plt.subplot(122)
        plt.plot([float(i) for i in list(KNN_ACC.keys())], list(KNN_ACC.values()), c='red')
        plt.xlabel('邻居值K')
        plt.ylabel('验证集上的准确率')
        plt.title('KNN调参图')
        plt.savefig('KNN.jpg')

        plt.show()


class Deeplearning():
    def train_one_batch(self,images, labels, optimizer, model, epoch, batch_idx):
        '''
        运行一个 batch 的训练，返回当前 batch 的训练日志
        '''
        # images, labels = next(iter(train_loader))  # iter()作用将train_loader作用成一个迭代器，我的理解就是可以根据前面的找到后面的，是有序的。
        #                                            # 以张量为单位,一个image就是一个32张*3通道*224*224大小的数据
        # images = images.to(device)  # 把图像数据传到GPU或者CPU里面
        # labels = labels.to(device)  # 把标签数据传到GPU或者CPU里面
        # outputs = model(images)     # 以image为单位进入网络进行前向预测得到这32个图片在2各类别上的概率（张量）是一个32*2的张量，32列表示32个图像，2表示分属两类的概率
        # loss = criterion(outputs, labels)  # 返回每个样本的平均交叉熵损失函数
        # """反向传播三部曲"""
        # optimizer.zero_grad()  # 清除梯度至0
        # loss.backward()  # 反向传播求梯度，微调使模型损失函数最小化
        # optimizer.step()  # 优化更新迭代
        # _, preds = torch.max(outputs, 1)  # _是每一列具体的值，但是I don't care,我要的值是最大值在outputs里每一列
        # 最大值的下标preds，它反映了这一组图像每一张被预测成什么
        criterion = nn.CrossEntropyLoss()
        # 获得一个 batch 的数据和标注
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)  # 输入模型，执行前向预测
        loss = criterion(outputs, labels)  # 计算当前 batch 中，每个样本的平均交叉熵损失函数值

        # 优化更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 获取当前 batch 的标签类别和预测类别
        _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
        preds = preds.cpu().numpy()
        loss = loss.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        log_train = {}
        log_train['epoch'] = epoch
        log_train['batch'] = batch_idx
        # 计算分类评估指标
        log_train['train_loss'] = loss
        log_train['train_accuracy'] = accuracy_score(labels, preds)
        log_train['train_precision'] = precision_score(labels, preds, average='macro')
        log_train['train_recall'] = recall_score(labels, preds, average='macro')
        log_train['train_f1-score'] = f1_score(labels, preds, average='macro')

        return log_train


    def evaluate_testset(self,test_loader, model, epoch):
        '''
        在整个测试集上评估，返回分类评估指标日志
        '''
        criterion = nn.CrossEntropyLoss()
        loss_list = []
        labels_list = []
        preds_list = []

        with torch.no_grad():
            for images, labels in test_loader:  # 生成一个 batch 的数据和标注
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)  # 输入模型，执行前向预测

                # 获取整个测试集的标签类别和预测类别
                _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
                preds = preds.cpu().numpy()
                loss = criterion(outputs, labels)  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
                loss = loss.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                loss_list.append(loss)
                labels_list.extend(labels)
                preds_list.extend(preds)

        log_test = {}
        log_test['epoch'] = epoch

        # 计算分类评估指标
        log_test['test_loss'] = np.mean(loss)
        log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
        log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
        log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
        log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')

        return log_test


    def model_fine_tune(self,
            train=ROOT + '//training',
            test=ROOT + '//evaluation',
            validation=ROOT + '//validation',
            device=device

    ):
        global lr_scheduler
        print('device', device)
        train_transform = transforms.Compose([transforms.RandomResizedCrop(256),  # 随压缩成256*256大小
                                              transforms.RandomHorizontalFlip(),  # 数据增强，提高模型的鲁棒性
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],  # imagenet上(R,G,B)的均值和方差，
                                                                   [0.229, 0.224, 0.225])  # 因此用这一组数据正则化
                                              ])

        # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
        test_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(
                                                 mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                                             ])
        train_path = os.path.join(os.getcwd(), train)
        test_path = os.path.join(os.getcwd(), test)

        train_dataset = datasets.ImageFolder(train_path, train_transform)

        # 载入测试集
        test_dataset = datasets.ImageFolder(test_path, test_transform)

        print('训练集图像数量', len(train_dataset))
        print('类别个数', len(train_dataset.classes))
        print('各类别名称', train_dataset.classes)

        print('测试集图像数量', len(test_dataset))
        print('类别个数', len(test_dataset.classes))
        print('各类别名称', test_dataset.classes)

        # 各类别名称
        class_names = train_dataset.classes
        n_class = len(class_names)
        idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}
        np.save('idx_to_labels.npy', idx_to_labels)
        np.save('labels_to_idx.npy', train_dataset.class_to_idx)

        BATCH_SIZE = 32

        # 训练集的数据加载器
        train_loader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,  # 每次计算32个样本
                                  shuffle=True,  # 在每个epochs计算式打乱数据
                                  num_workers=4  # 进程数，相当于在同一时刻计算4*32=108张图片
                                  )

        # 测试集的数据加载器
        test_loader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=4
                                 )

        """上述以dataloader为单位将数据喂给模型"""

        modelS = {  # 可以作为微调的模型
            'resnet18': models.resnet18(pretrained=True),
            # 'resnet34' : models.resnet34(pretrained=True),
            # 'resnet101' : models.resnet101(pretrained=True),
            # 'resnet152' : models.resnet152(pretrained=True)
            # 'models.alexnet': models.alexnet(pretrained=True),
            # 'models.squeezenet1_0': models.squeezenet1_0(pretrained=True),
            # 'models.vgg16': models.vgg16(pretrained=True),
            # 'models.densenet161': models.densenet161(pretrained=True),
            # 'models.inception_v3': models.inception_v3(pretrained=True),
            # 'models.googlenet': models.googlenet(pretrained=True),
            # 'models.shufflenet_v2_x1_0': models.shufflenet_v2_x1_0(pretrained=True),
            # 'models.mobilenet_v2': models.mobilenet_v2(pretrained=True),
            # 'models.resnext50_32x4d': models.resnext50_32x4d(pretrained=True),
            # 'models.wide_resnet50_2': models.wide_resnet50_2(pretrained=True),
            # 'models.mnasnet1_0': models.mnasnet1_0(pretrained=True)
        }
        for name, model in modelS.items():
            print(name)
            try:
                model.fc = nn.Linear(model.fc.in_features, n_class)  # 改变全连接层，映射到n_class维度，这里是二分类，n_class=2，model.fc是整个网络
                optimizer = optim.Adam(model.fc.parameters())  # 构造优化器，用来保存当前的状态，并能够根据计算得到的梯度来更新参数
                model = model.to(device)  # 这一步很奇妙，就18个字母，就能把这么庞大的，几百层的深度神经网络加到GPU/CPU
                # 交叉熵损失函数
                criterion = nn.CrossEntropyLoss()  # 损失函数是交叉熵
                # 训练轮次 Epoch
                EPOCHS = 300  # 每个模型，训练300轮
                lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

                # 训练开始之前，记录日志
                epoch = 0
                batch_idx = 0
                best_test_accuracy = 0

                # 训练日志-训练集
                df_train_log = pd.DataFrame()
                log_train = {}
                log_train['epoch'] = 0
                log_train['batch'] = 0

                # 训练日志-测试集
                df_test_log = pd.DataFrame()
                log_test = {}
                log_test['epoch'] = 0
                log_test.update(self.evaluate_testset(test_loader, model, epoch))
                df_test_log = df_test_log.append(log_test, ignore_index=True)

                for epoch in range(1, EPOCHS + 1):

                    print(f'Epoch {epoch}/{EPOCHS}')

                    ## 训练阶段
                    model.train()
                    for images, labels in tqdm(train_loader):  # 获得一个 batch 的数据和标注
                        batch_idx += 1
                        log_train = self.train_one_batch(images, labels, optimizer, model, epoch, batch_idx)
                        df_train_log = df_train_log.append(log_train, ignore_index=True)

                    lr_scheduler.step()

                    ## 测试阶段
                    model.eval()
                    log_test = self.evaluate_testset(test_loader, model, epoch)
                    df_test_log = df_test_log.append(log_test, ignore_index=True)

                    # 保存最新的最佳模型文件
                    if log_test['test_accuracy'] > best_test_accuracy:
                        # 删除旧的最佳模型文件(如有)

                        old_best_checkpoint_path = 'checkpoints/' + name + '_best-{:.3f}.pth'.format(best_test_accuracy)
                        if os.path.exists(old_best_checkpoint_path):
                            os.remove(old_best_checkpoint_path)
                        # 保存新的最佳模型文件
                        new_best_checkpoint_path = 'checkpoints/' + name + '_best-{:.3f}.pth'.format(
                            log_test['test_accuracy'])
                        torch.save(model, new_best_checkpoint_path)
                        print('保存新的最佳模型', 'checkpoints/' + name + '_best-{:.3f}.pth'.format(best_test_accuracy))
                        print('保存新的最佳模型', 'checkpoints/' + name + '_best-{:.3f}.pth')
                        best_test_accuracy = log_test['test_accuracy']

                df_train_log.to_csv('训练日志-训练集.csv', index=False)
                df_test_log.to_csv('训练日志-测试集.csv', index=False)
            except:
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """训练模型"""
    parser.add_argument('--train', nargs='+', type=str, default=ROOT + '\\' + 'training')  # 训练集
    parser.add_argument('--test', nargs='+', type=str, default=ROOT + '\\' + 'evaluation')  # 测试集
    parser.add_argument('--validation', nargs='+', type=str, default=ROOT + '\\' + 'validation')  # 验证集
    opt = parser.parse_args()
    T = traditional()
    D = Deeplearning()
    X_train_path, X_test_path, XX_train, Y_train, XX_test, Y_test, V_data, V_label, di = T.mk_dataset(**vars(opt))
    # print('------------------------微调--------------------------------')
    D.model_fine_tune(**vars(opt))
    # print('------------------------训练--------------------------------')
    MODEL = ['Bayes_best.pt','KNN_best.pt','SVM_best.pt','tree_best.pt']

    T.model_training()
    for i in MODEL:
        T.model_test(i)
