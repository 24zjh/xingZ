import os
import time
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from langchain_community.llms import Tongyi, QianfanLLMEndpoint, SparkLLM
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.chat_models import QianfanChatEndpoint, ChatSparkLLM
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

# 设置 API 密钥
# 设置 API 密钥，这些密钥用于访问不同的语言模型服务
os.environ["DASHSCOPE_API_KEY"] = "sk-39a2d85e79a5493a8859b60b725e5e55"  # Tongyi 模型的 API 密钥
os.environ["IFLYTEK_SPARK_APP_ID"] = "7fc4114e"  # Spark 模型的应用 ID
os.environ["IFLYTEK_SPARK_API_KEY"] = "0537b19ffd7b25bf4e3114325898e284"  # Spark 模型的 API 密钥
os.environ["IFLYTEK_SPARK_API_SECRET"] = "OTkxMTNkMDI5ODkwYjk0YTQ3YmRkYTUx"  # Spark 模型的 API 密钥
os.environ["QIANFAN_AK"] = "7aSK7bReecqYMjpHtHHowyRM"  # Qianfan 模型的 API 密钥
os.environ["QIANFAN_SK"] = "8VCRFZhDfIWHa6UYHNTa7hWaYBk2DTsG"  # Qianfan 模型的密钥

# 初始化嵌入模型
embeddings = QianfanEmbeddingsEndpoint()


# 函数：提取 JSON 文件中的内容并转换为向量
def extract_and_embed(json_files, embeddings_model):
    all_texts = []
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            texts = list(data["cn"]["message"].keys())
            # print(texts)
            all_texts.extend(texts)

    text_embeddings = embeddings_model.embed_documents(all_texts)
    print(all_texts)
    return text_embeddings, all_texts


# 函数：计算文本向量的相似度并进行聚类
def cluster_texts(text_embeddings, num_clusters=1):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(text_embeddings)
    clusters = kmeans.predict(text_embeddings)
    return clusters


# 归一化嵌入向量，确保所有值为非负
def normalize_embeddings(embeddings):
    embeddings = np.maximum(embeddings, 0)
    return embeddings


# 函数：使用 NMF 进行主题建模
def topic_modeling_nmf(text_embeddings, clusters):
    text_embeddings = normalize_embeddings(text_embeddings)
    nmf = NMF(n_components=5)
    topics = nmf.fit_transform(text_embeddings)
    return topics

def apply_translation_chain(texts, clusters, translation_models):
    translated_texts = []
    num_models = len(translation_models)
    for text, cluster in zip(texts, clusters):
        model_index = cluster % num_models  # 确保索引在模型范围内
        model = translation_models[model_index]
        translated_text = model.invoke(str(text))  # 使用 invoke 方法并确保输入为字符串
        translated_texts.append(translated_text)
        time.sleep(1)  # 控制请求频率，避免超出API限制
    return translated_texts

# 示例：处理多个 JSON 文件并生成翻译
json_files = ['AboutView_translation.json', 'destination_translation.json', 'indexBanner_translation.json', 'tourismProduct_translation.json']
text_embeddings, all_texts = extract_and_embed(json_files, embeddings)
print(all_texts)
print(text_embeddings)
clusters = cluster_texts(text_embeddings)
print(clusters)
topics = topic_modeling_nmf(text_embeddings, clusters)

# 使用受支持的模型替代
translation_models = [ChatTongyi(), QianfanChatEndpoint(endpoint="ERNIE-Bot-turbo"), ChatSparkLLM()]

translated_texts = apply_translation_chain(all_texts, clusters, translation_models)

# 输出原始文本和翻译后的文本
for original, translated in zip(all_texts, translated_texts):
    print(f"原文: {original}")
    print(f"翻译: {translated}")
