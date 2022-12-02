import numpy as np
import joblib
from sklearn import svm
import matplotlib.pyplot as plt

x = [[1, 2], [4, 5], [18, 9], [12, 6], [2, 3], [13, 18]]
x = np.array(x)
y = [1, 1, 0, 0, 1, 0]
y = np.array(y)
# 训练模型
model = svm.SVC(C=10, kernel='linear')
# 错误术语的惩罚参数C。
# C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，
# 这样对训练集测试时准确率很高，但泛化能力弱。
# C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
# kernel ： string，optional(default =‘rbf’)
# 核函数类型，str类型，默认为’rbf’。可选参数为：
# ’linear’：线性核函数
# ‘poly’：多项式核函数
# ‘rbf’：径像核函数/高斯核
# ‘sigmod’：sigmod核函数
# ‘precomputed’：核矩阵
# precomputed表示自己提前计算好核函数矩阵，这时候算法内部就不再用核函数去计算核矩阵，
# 而是直接用你给的核矩阵，核矩阵需要为n*n的。
model.fit(x, y)
# x：输入数据。如果模型只有一个输入，那么x的类型是numpy
# array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array
# y：标签，numpy array
joblib.dump(model, 'svm.m')  # 保存模型

# 预测类别
a = [[8, 6]]
a_pre = model.predict(a)
print("a_pre:", a_pre)
# 对应的支持向量
Support_vector = model.support_vectors_
print("Support_vector:", Support_vector)
# 线性分类对应的参数
w = model.coef_
print("w:", w)
b = model.intercept_
print("b:", b)
# 训练集散点图
plt.scatter(x[:, 0], x[:, 1])

if w[0, 1] != 0:
    xx = np.arange(0, 20, 0.1)
    # 最佳分类线
    yy = -w[0, 0] / w[0, 1] * xx - b / w[0, 1]
    plt.scatter(xx, yy, s=4)
    # 支持向量
    b1 = Support_vector[0, 1] + w[0, 0] / w[0, 1] * Support_vector[0, 0]
    b2 = Support_vector[1, 1] + w[0, 0] / w[0, 1] * Support_vector[1, 0]
    yy1 = -w[0, 0] / w[0, 1] * xx + b1
    plt.scatter(xx, yy1, s=4)
    yy2 = -w[0, 0] / w[0, 1] * xx + b2
    plt.scatter(xx, yy2, s=4)
else:
    xx = np.ones(100) * (-b) / w[0, 0]
    yy = np.arange(0, 10, 0.1)
    plt.scatter(xx, yy)
plt.show()