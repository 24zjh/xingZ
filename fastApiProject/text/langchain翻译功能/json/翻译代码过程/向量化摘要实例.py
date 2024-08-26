import os
import json
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 设置 API 密钥，这些密钥用于访问不同的语言模型服务
os.environ["DASHSCOPE_API_KEY"] = "sk-39a2d85e79a5493a8859b60b725e5e55"  # Tongyi 模型的 API 密钥
os.environ["IFLYTEK_SPARK_APP_ID"] = "7fc4114e"  # Spark 模型的应用 ID
os.environ["IFLYTEK_SPARK_API_KEY"] = "0537b19ffd7b25bf4e3114325898e284"  # Spark 模型的 API 密钥
os.environ["IFLYTEK_SPARK_API_SECRET"] = "OTkxMTNkMDI5ODkwYjk0YTQ3YmRkYTUx"  # Spark 模型的 API 密钥
os.environ["QIANFAN_AK"] = "7aSK7bReecqYMjpHtHHowyRM"  # Qianfan 模型的 API 密钥
os.environ["QIANFAN_SK"] = "8VCRFZhDfIWHa6UYHNTa7hWaYBk2DTsG"  # Qianfan 模型的密钥


# 函数：提取 JSON 文件中的内容
def extract_texts_from_json(json_files):
    all_texts = []
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            texts = list(data["cn"]["message"].keys())
            all_texts.extend(texts)
    return all_texts


# 函数：预处理文本数据
def preprocess_texts(texts):
    # 这里可以根据需要添加更多的预处理步骤
    return texts


# 函数：使用 TF-IDF 进行特征提取
def extract_features(texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix


# 函数：对文本进行聚类
def cluster_texts(tfidf_matrix, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)
    clusters = kmeans.predict(tfidf_matrix)
    return clusters


# 函数：生成摘要
def generate_summary(texts, clusters, tfidf_matrix):
    summary = []
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        cluster_indices = np.where(clusters == cluster)[0]
        cluster_texts = [texts[i] for i in cluster_indices]
        cluster_tfidf = tfidf_matrix[cluster_indices]
        # 选择与簇中心最接近的文本作为摘要
        centroid = cluster_tfidf.mean(axis=0)
        # 将 centroid 和 cluster_tfidf 转换为 numpy 数组
        centroid = np.asarray(centroid)
        cluster_tfidf = np.asarray(cluster_tfidf.todense())
        similarities = cosine_similarity(centroid, cluster_tfidf)
        most_representative_idx = cluster_indices[similarities.argmax()]
        summary.append(texts[most_representative_idx])
    return summary


# 主函数：执行文本摘要流程
def main():
    json_files = ['AboutView_translation.json', 'destination_translation.json', 'indexBanner_translation.json',
                  'tourismProduct_translation.json']
    texts = extract_texts_from_json(json_files)
    preprocessed_texts = preprocess_texts(texts)
    tfidf_matrix = extract_features(preprocessed_texts)
    clusters = cluster_texts(tfidf_matrix)
    summary = generate_summary(preprocessed_texts, clusters, tfidf_matrix)

    print("生成的摘要：")
    for sentence in summary:
        print(sentence)


if __name__ == "__main__":
    main()
