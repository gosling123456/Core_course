import json
import jieba
import gensim
import joblib
import warnings
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import shuffle
from sklearn import svm, manifold
from sklearn import tree, preprocessing
from gensim.models.doc2vec import Doc2Vec
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def stop_words():
    with open('stopword.txt', encoding='utf-8') as file:
        stopwords = file.read().split("\n")
    return stopwords


def read_json(path):
    text = []
    f_read = open(path, 'r', encoding='utf8', errors='ignore')
    for line in f_read:
        line = line.replace('\\u0009', '').replace('\\n', '')
        obj = json.loads(line)
        sent = obj['contentClean']
        text.append(sent)
    return text


def shuffle_data(x_train, y_train):
    c = list(zip(x_train, y_train))
    shuffle(c)
    x_train, y_train = zip(*c)  # 打乱顺序
    x_train = list(x_train)
    y_train = list(y_train)  # 打乱结果转换成列表
    return x_train, y_train


class TF_IDF:
    def __init__(self):
        self.corpus = read_json('从政.json') + read_json('体育.json') + read_json('国际.json') + read_json('经济.json')
        self.labels = [0] * 500 + [1] * 500 + [2] * 500 + [3] * 500
        self.corpus, self.labels = shuffle_data(self.corpus, self.labels)
        self.vectorizer = TfidfVectorizer(stop_words=stop_words(), max_features=1000)
        self.X = self.vectorizer.fit_transform(self.corpus).toarray()
        self.Y = np.array(self.labels)
        self.X_train = self.X[:int(self.X.shape[0] * 0.8)]  # 训练集5折交叉（训练集：测试集 = 4：1）
        self.Y_train = self.Y[:int(self.Y.shape[0] * 0.8)]
        self.X_test = self.X[int(self.X.shape[0] * 0.8):]  # 测试集5折交叉（训练集：测试集 = 4：1）
        self.Y_test = self.Y[int(self.Y.shape[0] * 0.8):]

    def TF_IDF_seen(self, model):

        tsne = manifold.TSNE(n_components=2, perplexity=20.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                             init="pca", random_state=0).fit_transform(self.X)
        model_result = {
            "AgglomerativeClustering": AgglomerativeClustering(n_clusters=4),
            "Kmeans": KMeans(n_clusters=4, random_state=10),
            "DBSCAN": DBSCAN(eps=0.2)

        }
        result = model_result[model].fit_predict(self.X).tolist()  # list中最小值是-1
        print(len(set(result)))
        for i in range(tsne.shape[0]):
            # print(tsne)
            x = np.array(tsne[i, 0])
            y = np.array(tsne[i, 1])
            plt.scatter(x, y, c=color[result[i]])
        plt.xlabel('n_class: %d' % len(set(result)))
        plt.title('TF-IDF_' + model + ' 聚类')
        # plt.legend(['第一类', '第二类', '第三类', '第四类'], fontsize=12)
        plt.savefig('figures/TF-IDF/' + model + ' 聚类.jpg')
        plt.show()

    def classify(self, model):
        model_list = {
            "SVM": svm.SVC(kernel='linear', probability=True),
            "Bayes": BernoulliNB(alpha=0.001),
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "Tree": tree.DecisionTreeClassifier(criterion='entropy')
        }
        clf = model_list[model]
        try:
            clf = joblib.load('models/TF_IDF/' + model + '1.m')
        except:
            print('开始训练模型..........')
            clf.fit(self.X_train, self.Y_train)
            joblib.dump(clf, 'models/TF_IDF/' + model + '1.m')
            print('模型训练结束。')
        print('开始预测........')
        correct = 0
        Precision = []
        results = []
        for i in range(int(self.X.shape[0] * 0.8), self.X.shape[0]):
            mark = clf.predict(self.X[i:i + 1])
            result = mark[0]
            if self.Y[i] == mark[0]:
                correct += 1
            results.append(result)
            Precision.append(correct / (i - 1600 + 1))
        print('预测结果为：', results)
        print('预测结束。利用%s算法在%d个样本中，一共有%d个样本被预测正确' % (model, int(self.X.shape[0] * 0.2), correct), end='\t')
        print('Accuracy:%.2f%%' % (correct / 400 * 100))
        num = [i for i in range(1, 401)]

        plt.plot(num, Precision, c='red')
        # plt.scatter(num,Precision, c='red')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlabel("累计测试样本个数", fontdict={'size': 16})
        plt.ylabel("实时准确率", fontdict={'size': 16})
        plt.title('TF-IDF_' + model + ' Precision', fontdict={'size': 20})
        plt.savefig("figures/TF-IDF/" + model + ' Precision.jpg')
        plt.show()


class Doc2vec:
    def __init__(self):
        self.corpus1 = read_json('从政.json') + read_json('体育.json') + read_json('国际.json') + read_json('经济.json')
        self.labels = [0] * 500 + [1] * 500 + [2] * 500 + [3] * 500
        self.vectorizer = TfidfVectorizer(stop_words=stop_words(), max_features=1000)
        self.X = self.vectorizer.fit_transform(self.corpus1).toarray()
        self.Y = np.array(self.labels)
        self.corpus, self.labels = shuffle_data(self.corpus1, self.labels)
        self.labels = np.array(self.labels)
        self.X_train = self.corpus[:1600]
        self.X_train, self.d_model = self.convert_data(self.X_train)
        self.Y_train = self.labels[:1600]
        self.X_test = self.corpus[1600:]
        self.Y_test = self.labels[1600:]
        # a = self.Y_train.tolist()
        # label0 = 0
        # label1 = 0
        # label2 = 0
        # label3 = 0
        # for i in a:
        #     if i == 0:
        #         label0 += 1
        #     elif i == 1:
        #         label1 += 1
        #     elif i == 2:
        #         label2 += 1
        #     elif i == 3:
        #         label3 += 1
        # print('0:',label0)
        # print('1',label1)
        # print('2',label2)
        # print('3',label3)
        self.data = []
        for sent in self.corpus1:
            processed_sent = jieba.cut(sent.strip(' '))
            self.data.append(list(processed_sent))

    def convert_data(self, data):
        train_text = []
        for i, sent in enumerate(data):
            tagged_doc = gensim.models.doc2vec.TaggedDocument(sent, tags=[i])
            train_text.append(tagged_doc)
        d_model = Doc2Vec(train_text, min_count=5, windows=3, vector_size=1000, sample=0.001, nagetive=5)
        try:
            d_model = gensim.models.doc2vec.Doc2Vec.load("doc2vec_model")
        except:
            d_model.train(train_text, total_examples=d_model.corpus_count, epochs=10)
            d_model.save("doc2vec_model")  # 保存模型
            d_model = gensim.models.doc2vec.Doc2Vec.load("doc2vec_model")  # 加载模型
        return d_model.docvecs.vectors_docs, d_model

    def classify(self, model):
        model_list = {
            "SVM": svm.SVC(kernel='linear', probability=True),
            "Bayes": BernoulliNB(alpha=0.001),
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "Tree": tree.DecisionTreeClassifier(criterion='entropy')
        }
        try:
            clf = joblib.load('models/Doc2vec/' + model + '2.m')
        except:
            print('开始训练模型..........')
            clf = model_list[model]
            clf.fit(self.X_train, self.Y_train)
            joblib.dump(clf, 'models/Doc2vec/' + model + '2.m')
            print('模型训练结束。')
        print('开始预测........')
        correct = 0
        Precision = []
        results = []
        for i in range(self.X_train.shape[0], self.labels.shape[0]):
            v1 = self.d_model.infer_vector(jieba.cut(self.corpus[i]))
            result = clf.predict([v1])[0]
            if result == self.labels[i]:
                correct += 1
            results.append(result)
            Precision.append(correct / (i - 1600 + 1))
        print('预测结果为：', results)
        print('预测结束。利用%s算法在%d个样本中，一共有%d个样本被预测正确' % (model, len(self.labels) - len(self.X_train), correct), end='\t')
        print('Accuracy:%.2f%%' % (correct / 400 * 100))

        num = [i for i in range(1, 401)]

        plt.plot(num, Precision, c='red')
        # plt.scatter(num,Precision, c='red')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlabel("累计测试样本个数", fontdict={'size': 16})
        plt.ylabel("实时准确率", fontdict={'size': 16})
        plt.title('Doc2vec_' + model + ' Precision', fontdict={'size': 20})
        plt.savefig("figures/Doc2vec/" + model + ' Precision.jpg')
        plt.show()

    def Doc2vec_seen(self, model):
        text = self.corpus1
        text_vecs, _ = self.convert_data(text)
        tsne = manifold.TSNE(n_components=2, perplexity=20.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                             init="pca", random_state=0).fit_transform(text_vecs)
        model_result = {
            "AgglomerativeClustering": AgglomerativeClustering(n_clusters=4),
            "Kmeans": KMeans(n_clusters=4, random_state=10),
            "DBSCAN": DBSCAN(eps=0.4
                             # , min_samples=4, metric='euclidean', algorithm='auto', leaf_size=30, p=None,
                             # n_jobs=1
                             )

        }

        result = model_result[model].fit_predict(text_vecs).tolist()  # list
        for i in range(tsne.shape[0]):
            x = np.array(tsne[i, 0])
            y = np.array(tsne[i, 1])
            plt.scatter(x, y, c=color[result[i]])
        plt.xlabel('n_class: %d' % len(set(result)))
        plt.title('Doc2vec_' + model + ' 聚类')
        plt.savefig("figures/Doc2Vec/" + model + '聚类.jpg')
        plt.show()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")  # 不显示警告
    color = ['r', 'g', 'b', 'c', 'k', 'm', 'purple', 'pink', 'yellow']
    #
    # print(
    #     '---------------------------------------------------------------------------------------------------------------------------')
    # print('TF-IDF专区\t\t\t')
    tf = TF_IDF()
    # tf.classify('SVM')
    # tf.classify('KNN')
    # tf.classify('Bayes')
    # tf.classify('Tree')
    # tf.TF_IDF_seen('Kmeans')
    # tf.TF_IDF_seen('AgglomerativeClustering')
    # tf.TF_IDF_seen('DBSCAN')
    # for i in range(10):
    #     print('-----------------------------------------------------------------------------')
    #     print('\n\n---------------------------------------------------------------------------')
    #     print('Doc2vec专区\n\n\n')
    doc = Doc2vec()
    # doc.classify('SVM')
    # doc.classify('KNN')
    # doc.classify('Bayes')
    # doc.classify('Tree')
    # doc.Doc2vec_seen('Kmeans')
    # doc.Doc2vec_seen('AgglomerativeClustering')
    # doc.Doc2vec_seen('DBSCAN')
