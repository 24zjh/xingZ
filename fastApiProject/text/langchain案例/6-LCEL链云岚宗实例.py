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
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
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


from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

model_qf = QianfanLLMEndpoint(temperature=0.1)
loader_txt = TextLoader(r'./云岚宗.txt', encoding='utf8')
docs_txt = loader_txt.load()
text_splitter_txt = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=0,
                                                   separators=["\n\n", "\n", " ", "", "。", "，"])
documents_txt = text_splitter_txt.split_documents(docs_txt)
embeddings_qf = QianfanEmbeddingsEndpoint()
vectordb = Chroma.from_documents(documents=documents_txt, embedding=embeddings_qf)

prompt = ChatPromptTemplate.from_template("""使用下面的语料来回答本模板最末尾的问题。如果你不知道问题的答案，直接回答 "我不知道"，禁止随意编造答案。
        为了保证答案尽可能简洁，你的回答必须不超过三句话，你的回答中不可以带有星号。
        请注意！在每次回答结束之后，你都必须接上 "感谢你的提问" 作为结束语
        以下是一对问题和答案的样例：
            请问：秦始皇的原名是什么
            秦始皇原名嬴政。感谢你的提问。

        以下是语料：
<context>
{context}
</context>

Question: {input}""")
# 创建检索链
document_chain = create_stuff_documents_chain(model_qf, prompt)

retriever = vectordb.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({
    "input": "纳兰桀是谁?"
})
print(response["answer"])



from langchain_core.runnables import RunnableParallel, RunnablePassthrough

llm = Tongyi()

chain_ce = {"context": retriever,"input":RunnablePassthrough()}|prompt|llm


print(chain_ce.invoke("纳兰嫣然来干嘛的？"))

# 改写上面《云岚宗》的rag案例，通过itemgetter实现参数的传入

from operator import itemgetter

chain_ce2 = (
        {
            "context": itemgetter("input")|retriever,
            "input":itemgetter("input")
        }
        |prompt
        |llm
)

print(chain_ce2.invoke({"input":"箫炎是谁？"}))
