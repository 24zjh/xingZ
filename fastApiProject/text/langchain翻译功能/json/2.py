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
def cluster_texts(text_embeddings, num_clusters=4):  # 假设有3个主题
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(text_embeddings)
    clusters = kmeans.predict(text_embeddings)
    return clusters

# 归一化嵌入向量，确保所有值为非负
def normalize_embeddings(embeddings):
    embeddings = np.maximum(embeddings, 0)
    return embeddings

# 函数：使用 NMF 进行主题建模
def topic_modeling_nmf(text_embeddings):
    text_embeddings = normalize_embeddings(text_embeddings)
    nmf = NMF(n_components=4)  # 3个主题
    topics = nmf.fit_transform(text_embeddings)
    return topics





from langchain_core.prompts import ChatPromptTemplate
system_prompt = f"""您是一名英语翻译助理，你将收到以下数据,类似类型的数据:
 "cn": {
        "message": {
            "旅游网站": Your translation here,
            "首页": "Your translation here",
            "目的地": "Your translation here",
            "旅游产品": "Your translation here",
            "关于我们": "Your translation here",
            "更多内容": "Your translation here",
            "最美旅游产品推荐": "Your translation here",
        }
    }
根据用户输入。 将您的响应作为带有
"en": {
        "message": {
            "你好": "Hello",
            "旅游网站": "Travel Website",
            "首页": "Homepage",
            "目的地": "Destination",
            "旅游产品": "Travel Products",
            "关于我们": "About Us",
            "更多内容": "More Content",
            "最美旅游产品推荐": "Best Travel Product Recommendation"
        }
    }
的翻译 JSON blob 返回."""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)

from langchain_core.output_parsers import JsonOutputParser

#我们可以使用`JsonOutputParser`将其解析为Python字典:
#如您所见,`JsonOutputParser`将JSON字符串解析为了一个Python字典对象。

parser = JsonOutputParser()

json_str = """```json
    "en": {
        "message": {
            "旅游网站": "Your translation here",
            "首页": "Your translation here",
            "目的地": "Your translation here",
            "旅游产品": "Your translation here",
            "关于我们": "Your translation here",
            "更多内容": "Your translation here",
            "最美旅游产品推荐": "Your translation here",
        }
    }
```"""
output_dict =parser.parse(json_str)
from operator import itemgetter
model = ChatTongyi()
chain = prompt | model | JsonOutputParser()
print(chain.invoke({"input": "你好"}))


# 函数：根据聚类结果选择合适的翻译模型
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
clusters = cluster_texts(text_embeddings)
print(clusters)
topics = topic_modeling_nmf(text_embeddings)
print(topics)
# 使用受支持的模型替代




translation_models = [ChatTongyi(), QianfanChatEndpoint(endpoint="ERNIE-Bot-turbo"), ChatSparkLLM()]
print(1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111)
translated_texts = apply_translation_chain(all_texts, clusters, ChatTongyi())
print(2222222222222222222222222222222222222222222222222222222222222222222222222222222222)
# 输出原始文本和翻译后的文本
for original, translated in zip(all_texts, translated_texts):
    print(f"原文: {original}")
    print(f"翻译: {translated}")
