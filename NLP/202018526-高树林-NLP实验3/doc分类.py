import json
def read_json(path):
    text = []
    f_read = open(path, 'r', encoding='utf8', errors='ignore')
    for line in f_read:
        line = line.replace('\\u0009', '').replace('\\n', '')
        obj = json.loads(line)
        sent = obj['contentClean']
        text.append(sent)
    return text


text = read_json('国际.json')+read_json('从政.json')+read_json('经济.json')+read_json('体育.json')


import jieba
processed_text=[]
for sent in text:
    processed_sent=jieba.cut(sent.strip(' '))
    processed_text.append(list(processed_sent))
print(processed_text[0])


import gensim
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
# 生成固定格式的训练文档集合
train_text=[]
for i,sent in enumerate(processed_text):
    #改变成Doc2vec所需要的输入样本格式，
    #由于gensim里Doc2vec模型需要的输入为固定格式，输入样本为：[句子，句子序号],这里需要
    tagged_doc=gensim.models.doc2vec.TaggedDocument(sent,tags=[i])
    train_text.append(tagged_doc)
#print(tagged_doc)
d_model=Doc2Vec(train_text,min_count=5,windows=3,vector_size=100,sample=0.001,nagetive=5)
d_model.train(train_text,total_examples=d_model.corpus_count,epochs=10)
# 保存模型，以便重用
d_model.save("doc2vec_model") #保存模型


import gensim
from gensim.models.doc2vec import Doc2Vec
#load doc2vec model...
d_model= gensim.models.doc2vec.Doc2Vec.load("doc2vec_model")
#load train vectors...
text_vecs= d_model.docvecs.vectors_docs
print("专利向量的个数为",len(text_vecs))
#print(text_vecs[0])


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4).fit(text_vecs)
result = kmeans.predict(text_vecs)
result = result.tolist()
res = ' '.join(str(i) for i in result)
print(res.count('0'))
print(res.count('1'))
print(res.count('2'))
print(res.count('3'))