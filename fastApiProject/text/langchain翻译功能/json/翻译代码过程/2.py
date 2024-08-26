import os
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

# 设置 API 密钥
os.environ["DASHSCOPE_API_KEY"] = "sk-39a2d85e79a5493a8859b60b725e5e55"
os.environ["IFLYTEK_SPARK_APP_ID"] = "7fc4114e"
os.environ["IFLYTEK_SPARK_API_KEY"] = "0537b19ffd7b25bf4e3114325898e284"
os.environ["IFLYTEK_SPARK_API_SECRET"] = "OTkxMTNkMDI5ODkwYjk0YTQ3YmRkYTUx"
os.environ["QIANFAN_AK"] = "7aSK7bReecqYMjpHtHHowyRM"
os.environ["QIANFAN_SK"] = "8VCRFZhDfIWHa6UYHNTa7hWaYBk2DTsG"

# 初始化嵌入模型
embeddings = QianfanEmbeddingsEndpoint()


# 函数：提取 JSON 文件中的内容并转换为向量
def extract_and_embed(json_files, embeddings_model):
    all_texts = []
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            texts = list(data["cn"]["message"].keys())
            all_texts.extend(texts)
    text_embeddings = embeddings_model.embed_documents(all_texts)
    return text_embeddings, all_texts


# 函数：计算文本向量的相似度并进行聚类
def cluster_texts(all_texts, num_clusters=5):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                       min_df=2, stop_words='english',
                                       use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

    n_components = min(20, tfidf_matrix.shape[1])  # 设置 n_components 为特征数量与20中较小的一个
    svd = TruncatedSVD(n_components=n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(lsa_matrix)
    clusters = kmeans.predict(lsa_matrix)
    return clusters


# 函数：使用 LDA 进行主题建模
def topic_modeling(all_texts, clusters):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tfidf_matrix)
    topics = lda.transform(tfidf_matrix)
    return topics


# 示例 JSON 文件列表
json_files = ['AboutView_translation.json', 'destination_translation.json', 'indexBanner_translation.json',
              'tourismProduct_translation.json']
# 提取并嵌入文本内容
text_embeddings, all_texts = extract_and_embed(json_files, embeddings)
# 对嵌入的文本向量进行聚类
clusters = cluster_texts(all_texts)
print(clusters)