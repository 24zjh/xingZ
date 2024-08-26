import numpy as np
import jieba
from gensim.models import Word2Vec, TfidfModel
from gensim import corpora
from sklearn.cluster import KMeans

# 示例中文文本数据
texts = [
    "神经网络经过大规模数据集训练以提高其准确性。",
    "机器学习模型在调优后达到了较高的准确率。",
    "数据预处理步骤对于取得良好结果至关重要。",
    "神经网络是一种模仿人脑的机器学习模型。"
]

# 1. 中文分词
cut_word_list = [list(jieba.cut(text)) for text in texts]
dictionary = corpora.Dictionary(cut_word_list)
corpus = [dictionary.doc2bow(text) for text in cut_word_list]

# 2. 使用Word2Vec训练词向量
word2vec_model = Word2Vec(cut_word_list, vector_size=200, window=5, min_count=1, workers=4)
word2vec_model.save('word2vec.model')
word2vec_model.train(cut_word_list, total_examples=word2vec_model.corpus_count, epochs=10)
wv = word2vec_model.wv

# 3. 结合TF-IDF进行向量加权
tfidf_model = TfidfModel(corpus)
corpus_tfidf = [tfidf_model[doc] for doc in corpus]
corpus_id_tfidf = list(map(dict, corpus_tfidf))
word_id = dictionary.token2id

def get_tfidf_vec(content):
    text_vec = np.zeros((len(content), 200))
    for ind, text in enumerate(content):
        vec = np.zeros((1, 200))
        for w in text:
            try:
                if word_id.get(w, False):
                    vec += (wv[w] * corpus_id_tfidf[ind][word_id[w]])
                else:
                    vec += wv[w]
            except:
                pass
        text_vec[ind] = vec / len(text)
    return text_vec

tfidf_vectors = get_tfidf_vec(cut_word_list)

# 4. 文本聚类
kmeans = KMeans(n_clusters=2).fit(tfidf_vectors)
labels = kmeans.labels_

# 5. 将文本按聚类标签进行分类
clustered_texts = {}
for i, label in enumerate(labels):
    if label not in clustered_texts:
        clustered_texts[label] = []
    clustered_texts[label].append(texts[i])

# 输出聚类结果
for label, cluster_texts in clustered_texts.items():
    print(f"聚类 {label}:")
    for text in cluster_texts:
        print(f" - {text}")
