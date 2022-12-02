import json

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# 读取json文件
def read_json(path):
    text = []
    f_read = open(path, 'r', encoding='utf8', errors='ignore')
    for line in f_read:
        line = line.replace('\\u0009', '').replace('\\n', '')
        obj = json.loads(line)
        sent = obj['contentClean']
        text.append(sent)
    return text


corpus = read_json('国际.json') + read_json('从政.json') + read_json('经济.json') + read_json('体育.json')
# labels = [0]*500+[1]*500+[2]*500+[3]*500
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(corpus).toarray()
# print(vectorizer.get_feature_names())
# print('\n')
# print('tf-idf向量化的结果为：\n', X)


from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

#
# kmeans = KMeans(n_clusters=4).fit(X)
# result = kmeans.predict(X)
# result = result.tolist()
# res = ' '.join(str(i) for i in result)
# print(res.count('0'))
# print(res.count('1'))
# print(res.count('2'))
# print(res.count('3'))
color = ['r', 'g', 'b', 'c', 'k', 'm', 'purple', 'pink', 'yellow']


def show(model):
    tsne = manifold.TSNE(n_components=3, perplexity=20.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                         init="pca", random_state=0).fit_transform(X)
    model_result = {
        "AgglomerativeClustering": AgglomerativeClustering(n_clusters=4),
        "Kmeans": KMeans(n_clusters=4, random_state=10),
        "DBSCAN": DBSCAN(eps=0.2)

    }
    result = model_result[model].fit_predict(X).tolist()  # list中最小值是-1
    print(len(set(result)))

    # print(tsne)
    x = np.array(tsne[:, 0])
    y = np.array(tsne[:, 1])
    z = np.array(tsne[:, 2])
    # plt.scatter(x, y, c=color[result[i]])
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(x.shape[0]):
        ax.scatter(x[i], y[i], z[i],c=color[result[i]])
    # plt.xlabel('n_class: %d' % len(set(result)))
    plt.title('TF-IDF_' + model + ' 聚类')
    # plt.legend(['第一类', '第二类', '第三类', '第四类'], fontsize=12)
    plt.savefig('figures/TF-IDF/' + model + ' 聚类.jpg')
    plt.show()


# print(kmeans.cluster_centers_)
show("Kmeans")
