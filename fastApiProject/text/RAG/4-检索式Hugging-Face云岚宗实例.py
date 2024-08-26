import os
from langchain_community.llms import Tongyi, SparkLLM, QianfanLLMEndpoint
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import numpy as np

# 设置API密钥的环境变量
os.environ["DASHSCOPE_API_KEY"] = "sk-39a2d85e79a5493a8859b60b725e5e55"
os.environ["IFLYTEK_SPARK_APP_ID"] = "7fc4114e"
os.environ["IFLYTEK_SPARK_API_KEY"] = "0537b19ffd7b25bf4e3114325898e284"
os.environ["IFLYTEK_SPARK_API_SECRET"] = "OTkxMTNkMDI5ODkwYjk0YTQ3YmRkYTUx"
os.environ["QIANFAN_AK"] = "7aSK7bReecqYMjpHtHHowyRM"
os.environ["QIANFAN_SK"] = "8VCRFZhDfIWHa6UYHNTa7hWaYBk2DTsG"

# 初始化模型
model_ty = Tongyi(temperature=0.1)
model_qf = QianfanLLMEndpoint(temperature=0.1)

# 导入文本分割器
loader_txt = TextLoader(r'../langchain案例/云岚宗.txt', encoding='utf8')
docs_txt = loader_txt.load()

# 使用递归字符文本分割器
text_splitter_txt = RecursiveCharacterTextSplitter(
    chunk_size=384,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", "", "。", "，"]
)

# 分割文本
documents_txt = text_splitter_txt.split_documents(docs_txt)

# 输出分割后的文档长度和内容
print(len(documents_txt))
print(documents_txt[0].page_content)

# 使用HuggingFace嵌入模型
embeddings_hf = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")

# 启用加速器
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# 嵌入查询
query1 = "狗"
query2 = "猫"
query3 = "雨"

emb1 = np.array(embeddings_hf.embed_query(query1))
emb2 = np.array(embeddings_hf.embed_query(query2))
emb3 = np.array(embeddings_hf.embed_query(query3))

# 打印嵌入向量的点积
print(np.dot(emb1, emb2))
print(np.dot(emb3, emb2))
print(np.dot(emb1, emb3))

# 创建和持久化Chroma向量数据库
vectordb = Chroma.from_documents(
    documents=documents_txt,
    embedding=embeddings_hf,
    persist_directory="C:\\langChainChromaSQL\\text1"
)
vectordb.persist()

# 加载持久化的向量数据库
vectordb_load = Chroma(
    persist_directory="C:\\langChainChromaSQL\\text1",
    embedding_function=embeddings_hf
)

# 打印向量数据库中的文档数量
print(vectordb_load._collection.count())

# 进行相似性搜索和最大边缘相关搜索
print(vectordb_load.similarity_search("云岚宗"))
print(vectordb_load.max_marginal_relevance_search("云岚宗"))

# 创建聊天提示模板
prompt = ChatPromptTemplate.from_template("""
使用下面的语料来回答本模板最末尾的问题。如果你不知道问题的答案，直接回答 "我不知道"，禁止随意编造答案。
为了保证答案尽可能简洁，你的回答必须不超过三句话，你的回答中不可以带有星号。
请注意！在每次回答结束之后，你都必须接上 "感谢你的提问" 作为结束语
以下是一对问题和答案的样例：
    请问：秦始皇的原名是什么
    秦始皇原名嬴政。感谢你的提问。

以下是语料：
<context>
{context}
</context>感谢你的提问。

Question: {input}""")

# 创建文档链和检索链
document_chain = create_stuff_documents_chain(model_ty, prompt)
retriever = vectordb_load.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 调用检索链获取答案
response = retrieval_chain.invoke({
    "input": "萧炎被谁退婚了?"
})

print(response["answer"])


response = retrieval_chain.invoke({
    "input": "他结婚了吗?"
})

print(response["answer"])