#导入语言模型
import os
from langchain_community.llms import Tongyi
from langchain_community.llms import SparkLLM
from langchain_community.llms import QianfanLLMEndpoint

import pandas as pd
#导入模版
from langchain.prompts import PromptTemplate

#导入聊天模型
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain_community.chat_models import ChatSparkLLM
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.chat_models import QianfanChatEndpoint

from langchain.chains import LLMChain

#输入三个模型各自的key

os.environ["DASHSCOPE_API_KEY"] = "sk-39a2d85e79a5493a8859b60b725e5e55"

os.environ["IFLYTEK_SPARK_APP_ID"] = "7fc4114e"
os.environ["IFLYTEK_SPARK_API_KEY"] = "0537b19ffd7b25bf4e3114325898e284"
os.environ["IFLYTEK_SPARK_API_SECRET"] = "OTkxMTNkMDI5ODkwYjk0YTQ3YmRkYTUx"

os.environ["QIANFAN_AK"] = "7aSK7bReecqYMjpHtHHowyRM"
os.environ["QIANFAN_SK"] = "8VCRFZhDfIWHa6UYHNTa7hWaYBk2DTsG"


model_ty = Tongyi(temperature=0.1)
model_qf = QianfanLLMEndpoint(temperature=0.1)
chat_ty = ChatTongyi()


from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser #仅仅是让输出对象成为字符串

prompt = ChatPromptTemplate.from_template("告诉我一个有关 {topic}的笑话")
model = ChatTongyi()
output_parser = StrOutputParser()

chain = prompt | model | output_parser


# ## 操作单变量输入和输出
#
# RunnableParallel 可用于操作一个 Runnable 的输出，使其匹配序列中下一个 Runnable 的输入格式。
#
# 在这里，prompt 的输入预期是一个带有“context”和“question”键的映射。用户输入仅是问题。因此，我们需要使用检索器获取上下文，并将用户输入作为“question”键传递。
#
# * 代码中每个 "|" 前后的元素都可看作是一个Runnable。
#
# 其中的 {"context": retriever, "question": RunnablePassthrough()} 就是RunnableParallel，它的作用就是将输入组成 context 和 question 为key的字典格式，传递给 prompt。
#
# RunnableParallel 的使用可以有以下三种形式，三种形式等价：
#
# * RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
# * {"context": retriever,"question": RunnablePassthrough()}
# * RunnableParallel(context=retriever, question=RunnablePassthrough())

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

vectorstore = Chroma.from_texts(
["小明在华为工作"], embedding=QianfanEmbeddingsEndpoint()
)
retriever = vectorstore.as_retriever()
template ="""仅根据以下上下文回答问题：
{context}

问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatTongyi()

retrieval_chain =(
{"context": retriever,"question": RunnablePassthrough()}
| prompt
| model
| StrOutputParser()
)


