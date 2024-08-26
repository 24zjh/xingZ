#导入语言模型
import os
from langchain_community.llms import Tongyi
from langchain_community.llms import SparkLLM
from langchain_community.llms import QianfanLLMEndpoint

from langchain_community.chat_models import ChatSparkLLM
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.chat_models import QianfanChatEndpoint

from langchain.chains import LLMChain

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser #仅仅是让输出对象成为字符串

from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

from operator import itemgetter



from langchain.prompts import PromptTemplate
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# 导入所需的库和模块
import os
from langchain_community.llms import Tongyi  # 导入同一语言模型（Tongyi）
from langchain_community.llms import SparkLLM  # 导入 Spark 语言模型（SparkLLM）
from langchain_community.llms import QianfanLLMEndpoint  # 导入 Qianfan 语言模型（QianfanLLMEndpoint）

import pandas as pd  # 导入 pandas 库用于数据处理
from langchain.prompts import PromptTemplate  # 导入 PromptTemplate 用于创建提示模板

# 导入聊天模型和消息模板
from langchain.prompts.chat import (
    ChatPromptTemplate,  # 用于创建聊天提示模板
    SystemMessagePromptTemplate,  # 用于系统消息的模板
    AIMessagePromptTemplate,  # 用于 AI 消息的模板
    HumanMessagePromptTemplate,  # 用于人类消息的模板
)
from langchain.schema import (
    AIMessage,  # 表示 AI 消息的类
    HumanMessage,  # 表示人类消息的类
    SystemMessage  # 表示系统消息的类
)

from langchain_community.chat_models import ChatSparkLLM  # 导入 SparkLLM 聊天模型
from langchain_community.chat_models.tongyi import ChatTongyi  # 导入 Tongyi 聊天模型
from langchain_community.chat_models import QianfanChatEndpoint  # 导入 Qianfan 聊天模型

from langchain.chains import LLMChain  # 导入 LLMChain 用于构建语言模型链

# 设置三个模型各自的 API 密钥
os.environ["DASHSCOPE_API_KEY"] = "sk-39a2d85e79a5493a8859b60b725e5e55"  # 设置 Tongyi 模型的 API 密钥
os.environ["IFLYTEK_SPARK_APP_ID"] = "7fc4114e"  # 设置 Spark 模型的应用 ID
os.environ["IFLYTEK_SPARK_API_KEY"] = "0537b19ffd7b25bf4e3114325898e284"  # 设置 Spark 模型的 API 密钥
os.environ["IFLYTEK_SPARK_API_SECRET"] = "OTkxMTNkMDI5ODkwYjk0YTQ3YmRkYTUx"  # 设置 Spark 模型的 API 密钥
os.environ["QIANFAN_AK"] = "7aSK7bReecqYMjpHtHHowyRM"  # 设置 Qianfan 模型的 API 密钥
os.environ["QIANFAN_SK"] = "8VCRFZhDfIWHa6UYHNTa7hWaYBk2DTsG"  # 设置 Qianfan 模型的密钥

# 导入其他必要的模块
from langchain.prompts import ChatPromptTemplate  # 再次导入 ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser  # 导入字符串输出解析器，用于将输出对象转换为字符串
from langchain_core.runnables import RunnablePassthrough  # 导入 RunnablePassthrough，用于数据传递
from langchain.vectorstores import Chroma  # 导入 Chroma，用于向量存储
from langchain_community.embeddings import QianfanEmbeddingsEndpoint  # 导入 Qianfan Embeddings 模型

from operator import itemgetter  # 导入 itemgetter，用于提取对象中的数据

# 初始化语言模型和聊天模型
model_ty = Tongyi(temperature=0.1)  # 初始化 Tongyi 模型，设置温度为 0.1
model_qf = QianfanLLMEndpoint(temperature=0.1)  # 初始化 Qianfan 模型，设置温度为 0.1
chat_ty = ChatTongyi()  # 初始化 Tongyi 聊天模型

# 创建提示模板
physics_template = """
你是一个非常聪明的综合语言翻译专家
你擅长以简洁易懂和规范的的方式翻译语言问题。
当你不知道答案时，你会承认不知道。

这是一个代翻译的数据：
{query}"""

math_template = """
你是一个非常优秀的俄罗斯语言翻译专家
你擅长以简洁易懂和规范的的方式翻译语言问题。
当你不知道答案时，你会承认不知道。

这是一个代翻译的数据：
{query}"""

# 初始化 Qianfan 的嵌入模型
embeddings = QianfanEmbeddingsEndpoint()

# 将物理和数学的提示模板转换为嵌入向量
prompt_templates = [physics_template, math_template]  # 存储提示模板
prompt_embeddings = embeddings.embed_documents(prompt_templates)  # 对模板进行向量化处理


# 向量存储，主题，向量化存储数据的关联性



# 定义一个路由函数，用于根据输入问题选择最合适的模板
def prompt_router(input):
    query_embedding = embeddings.embed_query(input["query"])  # 对输入的查询进行向量化
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]  # 计算输入与模板的相似度
    most_similar = prompt_templates[similarity.argmax()]  # 找到与输入最相似的模板
    print("使用综合语言" if most_similar == math_template else "使用俄罗斯语言")  # 输出选择的模板类型
    return PromptTemplate.from_template(most_similar)  # 返回相应的模板

# 创建链式操作，将输入通过多个步骤处理
chain = (
    {"query": RunnablePassthrough()}  # 将查询数据传递下去
    | RunnableLambda(prompt_router)  # 通过路由函数选择适当的提示模板
    | ChatTongyi()  # 使用 Tongyi 聊天模型进行处理
    | StrOutputParser()  # 将模型的输出解析为字符串格式
)


print(chain.invoke("什么是黑洞"))