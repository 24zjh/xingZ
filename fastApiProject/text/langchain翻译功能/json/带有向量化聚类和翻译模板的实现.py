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


# 函数：根据聚类结果选择合适的翻译模型
def apply_translation_chain(texts, clusters, translation_models):
    translated_texts = []  # 用于存储翻译后的文本
    for text, cluster in zip(texts, clusters):
        model = translation_models[cluster]  # 根据聚类结果选择对应的翻译模型
        translated_text = model.translate(text)  # 使用选定的模型进行翻译
        translated_texts.append(translated_text)  # 将翻译结果添加到列表中
    return translated_texts  # 返回翻译后的文本列表


# 示例：处理多个 JSON 文件并生成翻译
# 示例：处理多个 JSON 文件并生成翻译
json_files = ['AboutView_translation.json', 'destination_translation.json', 'indexBanner_translation.json','tourismProduct_translation.json']  # 示例 JSON 文件列表
# 提取并嵌入文本内容
text_embeddings, all_texts = extract_and_embed(json_files, embeddings)
# 对嵌入的文本向量进行聚类
clusters = cluster_texts(text_embeddings)
# 定义多个翻译模型
translation_models = [ChatTongyi(), QianfanChatEndpoint(), ChatSparkLLM()]
# 根据聚类结果选择合适的翻译模型进行翻译
translated_texts = apply_translation_chain(all_texts, clusters, translation_models)

# 输出原始文本和翻译后的文本
for original, translated in zip(all_texts, translated_texts):
    print(f"原文: {original}")
    print(f"翻译: {translated}")



# 以下是代码的详细注释，以帮助你理解每个部分的功能和用途：
#
# ```python
# import os
# import json
# from sklearn.cluster import KMeans
# from sklearn.decomposition import LatentDirichletAllocation
# from langchain_community.llms import Tongyi, QianfanLLMEndpoint, SparkLLM
# from langchain_community.chat_models.tongyi import ChatTongyi
# from langchain_community.chat_models import QianfanChatEndpoint, ChatSparkLLM
# from langchain.vectorstores import Chroma
# from langchain_community.embeddings import QianfanEmbeddingsEndpoint
#
# # 设置 API 密钥，这些密钥用于访问不同的语言模型服务
# os.environ["DASHSCOPE_API_KEY"] = "sk-39a2d85e79a5493a8859b60b725e5e55"  # Tongyi 模型的 API 密钥
# os.environ["IFLYTEK_SPARK_APP_ID"] = "7fc4114e"  # Spark 模型的应用 ID
# os.environ["IFLYTEK_SPARK_API_KEY"] = "0537b19ffd7b25bf4e3114325898e284"  # Spark 模型的 API 密钥
# os.environ["IFLYTEK_SPARK_API_SECRET"] = "OTkxMTNkMDI5ODkwYjk0YTQ3YmRkYTUx"  # Spark 模型的 API 密钥
# os.environ["QIANFAN_AK"] = "7aSK7bReecqYMjpHtHHowyRM"  # Qianfan 模型的 API 密钥
# os.environ["QIANFAN_SK"] = "8VCRFZhDfIWHa6UYHNTa7hWaYBk2DTsG"  # Qianfan 模型的密钥
#
# # 初始化嵌入模型，用于将文本内容转换为嵌入向量表示
# embeddings = QianfanEmbeddingsEndpoint()
#
#
# # 函数：提取 JSON 文件中的内容并转换为向量
# def extract_and_embed(json_files, embeddings_model):
#     all_texts = []  # 用于存储所有提取的文本内容
#     for file in json_files:
#         with open(file, 'r', encoding='utf-8') as f:
#             data = json.load(f)  # 读取 JSON 文件内容
#             # 提取 "cn" -> "message" 中的所有键（即待翻译的文本内容）
#             texts = list(data["cn"]["message"].keys())
#             all_texts.extend(texts)  # 将所有文本内容添加到 all_texts 列表中
#
#     # 使用嵌入模型将所有文本转换为向量表示
#     text_embeddings = embeddings_model.embed_documents(all_texts)
#     return text_embeddings, all_texts  # 返回文本的向量表示和原始文本内容
#
#
# # 函数：计算文本向量的相似度并进行聚类
# def cluster_texts(text_embeddings, num_clusters=5):
#     # 使用 K-means 聚类算法对文本向量进行聚类
#     kmeans = KMeans(n_clusters=num_clusters)
#     kmeans.fit(text_embeddings)  # 根据文本向量进行聚类训练
#     clusters = kmeans.predict(text_embeddings)  # 获取每个文本所属的聚类类别
#     return clusters  # 返回聚类结果
#
#
# # 函数：使用 LDA 进行主题建模
# def topic_modeling(text_embeddings, clusters):
#     # 使用 LDA（Latent Dirichlet Allocation）进行主题建模
#     lda = LatentDirichletAllocation(n_components=5)
#     lda.fit(text_embeddings)  # 基于文本向量训练 LDA 模型
#     topics = lda.transform(text_embeddings)  # 获取文本向量的主题分布
#     return topics  # 返回主题分布
#
#
# # 函数：根据聚类结果选择合适的翻译模型
# def apply_translation_chain(texts, clusters, translation_models):
#     translated_texts = []  # 用于存储翻译后的文本
#     for text, cluster in zip(texts, clusters):
#         model = translation_models[cluster]  # 根据聚类结果选择对应的翻译模型
#         translated_text = model.translate(text)  # 使用选定的模型进行翻译
#         translated_texts.append(translated_text)  # 将翻译结果添加到列表中
#     return translated_texts  # 返回翻译后的文本列表
#
#
# # 示例：处理多个 JSON 文件并生成翻译
# json_files = ['homepage.json', 'profile.json', 'recommendation.json']  # 示例 JSON 文件列表
# # 提取并嵌入文本内容
# text_embeddings, all_texts = extract_and_embed(json_files, embeddings)
# # 对嵌入的文本向量进行聚类
# clusters = cluster_texts(text_embeddings)
# # 使用 LDA 对文本向量进行主题建模
# topics = topic_modeling(text_embeddings, clusters)
# # 定义多个翻译模型
# translation_models = [ChatTongyi(), QianfanChatEndpoint(), ChatSparkLLM()]
# # 根据聚类结果选择合适的翻译模型进行翻译
# translated_texts = apply_translation_chain(all_texts, clusters, translation_models)
#
# # 输出原始文本和翻译后的文本
# for original, translated in zip(all_texts, translated_texts):
#     print(f"原文: {original}")
#     print(f"翻译: {translated}")
# ```
#
# ### 详细注释说明
#
# 1. **导入模块与库**:
#    - `os`, `json` 用于文件操作和数据解析。
#    - `KMeans`, `LatentDirichletAllocation` 来自 `sklearn`，用于聚类和主题建模。
#    - `Tongyi`, `QianfanLLMEndpoint`, `SparkLLM`, `ChatTongyi`, `QianfanChatEndpoint`, `ChatSparkLLM` 是来自 `LangChain` 的不同模型和嵌入技术，用于处理和生成文本。
#
# 2. **API 密钥设置**:
#    - 这些环境变量用于存储各个模型的 API 密钥，确保在调用这些服务时能够成功验证身份。
#
# 3. **嵌入模型初始化**:
#    - `QianfanEmbeddingsEndpoint` 是一个嵌入模型，用于将文本转换为向量表示。向量表示捕捉了文本的语义信息，使得不同的文本片段可以在向量空间中进行比较。
#
# 4. **`extract_and_embed` 函数**:
#    - 从指定的 JSON 文件中提取所有待翻译的文本内容，然后使用嵌入模型将这些文本转换为向量表示。
#    - 通过 `json.load` 读取文件并解析成字典格式，从中提取 `"cn"` -> `"message"` 中的所有键（即待翻译的中文文本）。
#
# 5. **`cluster_texts` 函数**:
#    - 使用 K-means 聚类算法将文本向量进行聚类。每个聚类表示相似的文本片段，它们可能属于同一主题或具有类似的翻译需求。
#
# 6. **`topic_modeling` 函数**:
#    - 使用 LDA（Latent Dirichlet Allocation）进行主题建模，从聚类的文本向量中提取主题。这帮助理解文本内容，并为每个聚类提供上下文参考。
#
# 7. **`apply_translation_chain` 函数**:
#    - 根据聚类结果，选择合适的翻译模型对文本进行翻译。通过将每个文本与其对应的聚类索引进行匹配，选择最适合该类别的翻译模型。
#
# 8. **主流程**:
#    - 从多个 JSON 文件中提取文本并嵌入为向量。
#    - 对嵌入向量进行聚类分析，并进行主题建模。
#    - 根据聚类结果选择适当的翻译模型进行翻译。
#    - 输出每个文本的原文和翻译结果。
#
# 通过这些详细注释，你可以更好地理解代码的每个部分是如何工作的，以及它们如何协同完成从文本提取到翻译的整个流程。