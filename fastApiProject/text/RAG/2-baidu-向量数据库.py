import os

from langchain_community.llms import QianfanLLMEndpoint

os.environ["QIANFAN_AK"] = "7aSK7bReecqYMjpHtHHowyRM"
os.environ["QIANFAN_SK"] = "8VCRFZhDfIWHa6UYHNTa7hWaYBk2DTsG"

llm_wenxin = QianfanLLMEndpoint()

from langchain_community.embeddings import QianfanEmbeddingsEndpoint
import numpy as np

embeddings_qf = QianfanEmbeddingsEndpoint()

query1 = "狗"
query2 = "猫"
query3 = "雨"

# 通过对应的 embedding 类生成 query 的 embedding。
emb1 = embeddings_qf.embed_query(query1)
emb2 = embeddings_qf.embed_query(query2)
emb3 = embeddings_qf.embed_query(query3)

print(emb1)
print(emb2)
print(emb3)