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

#输入三个模型各自的key

os.environ["DASHSCOPE_API_KEY"] = "sk-39a2d85e79a5493a8859b60b725e5e55"

os.environ["IFLYTEK_SPARK_APP_ID"] = "7fc4114e"
os.environ["IFLYTEK_SPARK_API_KEY"] = "0537b19ffd7b25bf4e3114325898e284"
os.environ["IFLYTEK_SPARK_API_SECRET"] = "OTkxMTNkMDI5ODkwYjk0YTQ3YmRkYTUx"

os.environ["QIANFAN_AK"] = "7aSK7bReecqYMjpHtHHowyRM"
os.environ["QIANFAN_SK"] = "8VCRFZhDfIWHa6UYHNTa7hWaYBk2DTsG"

model_ty = Tongyi(temperature=0.1)
model_qf = QianfanLLMEndpoint(temperature=0.1)

print(model_ty)
print(model_qf)


from langchain.document_loaders import PyPDFLoader


from langchain_community.document_loaders import WebBaseLoader
WEB_URL = "https://news.ifeng.com/c/8Y3TlIcTsj0"
loader_html = WebBaseLoader(WEB_URL)
docs_html = loader_html.load()

print(len(docs_html))