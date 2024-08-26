# 存在问题，切换大模型，无法有相同的结果


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
from langchain.prompts import PromptTemplate
from langchain.chains import LLMRequestsChain, LLMChain
import os
from langchain_community.llms import SparkLLM
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

from langchain.output_parsers import StructuredOutputParser, ResponseSchema






list =  {
    "cn": {
        "message": {
            "旅游网站": "",
            "首页": "",
            "目的地": "Your translation here",
            "旅游产品": "Your translation here",
            "关于我们": "Your translation here",
            "更多内容": "Your translation here",
            "欢迎来到旅游主页": "Your translation here",
            "我们提供全球各地最美的旅游景点，带你畅游世界的旅游美景": "Your translation here",
            "查看更多": "Your translation here",
            "预定旅行": "Your translation here",
            "专业团队": "Your translation here",
        }
    }
}

# 清空所有值但保留键
# 提取键并将它们存储为一个集合
keys_set = set(list['cn']['message'].keys())

# 输出集合
print(keys_set)





response_schemas = [
    ResponseSchema(name='message', description="key是翻译前的内容，value是翻译后内容")
]


output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
llm_spark = SparkLLM(temperature=0.1)
model_ty = Tongyi(temperature=0.1)
llm_wenxin = QianfanLLMEndpoint(temperature=0.1)

from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["words", "language"],
    template="将以下内容翻译成{language}，并使用以下结构输出：{format_instructions}。需要翻译的内容如下：{words}",
    partial_variables={"format_instructions": format_instructions}
)



_input = prompt.format_prompt(words={"翻译前的内容": keys_set}, language="英语")
output = model_ty.invoke(_input.to_string())


# 假设模型返回的输出符合正确的结构
parsed_output = output_parser.parse(output)
print(parsed_output)


#
# chain = LLMChain(llm=model_ty, prompt=prompt)
# print(chain.invoke({"words":list,"language":"英语"}))
