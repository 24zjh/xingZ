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


print(keys_set)
# 输出逗号分隔的格式
formatted_keys = ', '.join(keys_set)
print(formatted_keys)





# response_schemas = [
#     ResponseSchema(name='message', description="key是翻译前的内容，value是翻译后内容")
# ]

from langchain.output_parsers import CommaSeparatedListOutputParser
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()


# output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# format_instructions = output_parser.get_format_instructions()
llm_spark = SparkLLM(temperature=0.1)
model_ty = Tongyi(temperature=0.1)
llm_wenxin = QianfanLLMEndpoint(temperature=0.1)

from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["words", "language"],
    template="将以下内容翻译成{language},并使用以下结构输出:{format_instructions},需要翻译的内容如下：{words}",
    partial_variables={"format_instructions": format_instructions}
)


llm_spark = SparkLLM(temperature=0.1)
model_ty = Tongyi(temperature=0.1)
llm_wenxin = QianfanLLMEndpoint(temperature=0.1)


_input = prompt.format_prompt(words= keys_set, language="英语")
output = llm_spark.invoke(_input)


# 假设模型返回的输出符合正确的结构
parsed_output = output_parser.parse(output)
print(parsed_output)

def clean_output(output_list):
    cleaned_output = []
    for item in output_list:
        # 移除可能存在的反引号、单引号和双引号，并修正输出格式
        cleaned_item = item.strip('`').strip("'").strip('"')
        cleaned_output.append(cleaned_item)
    return cleaned_output

try:
    # 假设模型返回的输出符合正确的结构
    parsed_output = output_parser.parse(output)

    # 检查并修复输出中的引号（反引号、单引号、双引号）
    final_output = clean_output(parsed_output)

except Exception as e:
    print(f"An error occurred while processing the output: {e}")
    final_output = parsed_output  # 如果处理失败，使用原始输出

print(final_output)

final_dict = {key: value for key, value in zip(keys_set, final_output)}

# 将最终的字典格式化为 JSON
final_json = {
    "cn": {
        "message": final_dict
    }
}

# 输出最终的 JSON 数据
print(json.dumps(final_json, ensure_ascii=False, indent=4))




#
# chain = LLMChain(llm=model_ty, prompt=prompt)
# print(chain.invoke({"words":list,"language":"英语"}))
