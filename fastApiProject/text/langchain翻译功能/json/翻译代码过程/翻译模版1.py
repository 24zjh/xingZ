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
# 设置 API 密钥
os.environ["DASHSCOPE_API_KEY"] = "sk-39a2d85e79a5493a8859b60b725e5e55"
os.environ["IFLYTEK_SPARK_APP_ID"] = "7fc4114e"
os.environ["IFLYTEK_SPARK_API_KEY"] = "0537b19ffd7b25bf4e3114325898e284"
os.environ["IFLYTEK_SPARK_API_SECRET"] = "OTkxMTNkMDI5ODkwYjk0YTQ3YmRkYTUx"
os.environ["QIANFAN_AK"] = "7aSK7bReecqYMjpHtHHowyRM"
os.environ["QIANFAN_SK"] = "8VCRFZhDfIWHa6UYHNTa7hWaYBk2DTsG"



model_ty = Tongyi(temperature=0.1)

from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["word","language"],
    template="我希望你能充当一名翻译专家。将{word}的对应{language}语言翻译出来",
)

chain = LLMChain(llm=model_ty, prompt=prompt)


print(chain.invoke({"word":"彩色袜子","language":"俄罗斯"}))



