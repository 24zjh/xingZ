import os
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from langchain_community.llms import Tongyi, QianfanLLMEndpoint, SparkLLM
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.chat_models import QianfanChatEndpoint, ChatSparkLLM
from langchain.vectorstores import Chroma
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

# 设置 API 密钥，这些密钥用于访问不同的语言模型服务
os.environ["DASHSCOPE_API_KEY"] = "sk-39a2d85e79a5493a8859b60b725e5e55"  # Tongyi 模型的 API 密钥
os.environ["IFLYTEK_SPARK_APP_ID"] = "7fc4114e"  # Spark 模型的应用 ID
os.environ["IFLYTEK_SPARK_API_KEY"] = "0537b19ffd7b25bf4e3114325898e284"  # Spark 模型的 API 密钥
os.environ["IFLYTEK_SPARK_API_SECRET"] = "OTkxMTNkMDI5ODkwYjk0YTQ3YmRkYTUx"  # Spark 模型的 API 密钥
os.environ["QIANFAN_AK"] = "7aSK7bReecqYMjpHtHHowyRM"  # Qianfan 模型的 API 密钥
os.environ["QIANFAN_SK"] = "8VCRFZhDfIWHa6UYHNTa7hWaYBk2DTsG"  # Qianfan 模型的密钥

# 初始化嵌入模型，用于将文本内容转换为嵌入向量表示
embeddings = QianfanEmbeddingsEndpoint()


# 函数：提取 JSON 文件中的内容并转换为向量
def extract_and_embed(json_files, embeddings_model):
    all_texts = []  # 用于存储所有提取的文本内容
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 读取 JSON 文件内容
            # 提取 "cn" -> "message" 中的所有键（即待翻译的文本内容）
            texts = list(data["cn"]["message"].keys())
            all_texts.extend(texts)  # 将所有文本内容添加到 all_texts 列表中

    # 使用嵌入模型将所有文本转换为向量表示
    text_embeddings = embeddings_model.embed_documents(all_texts)
    return text_embeddings, all_texts  # 返回文本的向量表示和原始文本内容


# 函数：计算文本向量的相似度并进行聚类
def cluster_texts(text_embeddings, num_clusters=5):
    # 使用 K-means 聚类算法对文本向量进行聚类
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(text_embeddings)  # 根据文本向量进行聚类训练
    clusters = kmeans.predict(text_embeddings)  # 获取每个文本所属的聚类类别
    return clusters  # 返回聚类结果


# 函数：使用 LDA 进行主题建模
def topic_modeling(text_embeddings, clusters):
    # 使用 LDA（Latent Dirichlet Allocation）进行主题建模
    lda = LatentDirichletAllocation(n_components=5)
    lda.fit(text_embeddings)  # 基于文本向量训练 LDA 模型
    topics = lda.transform(text_embeddings)  # 获取文本向量的主题分布
    return topics  # 返回主题分布

json_files = ['AboutView_translation.json', 'destination_translation.json', 'indexBanner_translation.json','tourismProduct_translation.json']  # 示例 JSON 文件列表
# 提取并嵌入文本内容
text_embeddings, all_texts = extract_and_embed(json_files, embeddings)
# 对嵌入的文本向量进行聚类
clusters = cluster_texts(text_embeddings)


