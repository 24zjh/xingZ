import os
from langchain_community.llms import Tongyi, QianfanLLMEndpoint
from langchain_community.chat_models import ChatTongyi, QianfanChatEndpoint
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.vectorstores import Chroma
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.utils.math import cosine_similarity

# 设置 API 密钥
os.environ["DASHSCOPE_API_KEY"] = "sk-39a2d85e79a5493a8859b60b725e5e55"
os.environ["QIANFAN_AK"] = "7aSK7bReecqYMjpHtHHowyRM"
os.environ["QIANFAN_SK"] = "8VCRFZhDfIWHa6UYHNTa7hWaYBk2DTsG"

# 初始化模型
chat_ty = ChatTongyi()
embeddings = QianfanEmbeddingsEndpoint()

# 定义提示模板
physics_template = """
你是一个非常聪明的物理学教授。
你擅长以简洁易懂的方式回答物理问题。
当你不知道答案时，你会承认不知道。

这是一个问题：
{query}"""

math_template = """
你是一个非常优秀的数学家。
你擅长回答数学问题。
你之所以优秀是因为你能够将难题分解为组成部分，
回答组成部分，然后把它们组合起来回答更广泛的问题。

这是一个问题：
{query}"""

# 将模板转换为嵌入向量
prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

# 定义路由函数，根据输入问题选择最合适的模板
def prompt_router(input):
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    print("使用数学" if most_similar == math_template else "使用物理")
    return PromptTemplate.from_template(most_similar)

# 创建链式操作，将输入通过多个步骤处理
chain = (
    {"query": RunnablePassthrough()}  # 将查询数据传递下去
    | RunnableLambda(prompt_router)  # 通过路由函数选择适当的提示模板
    | chat_ty  # 使用 Tongyi 聊天模型进行处理
)

# 测试输入
response = chain.invoke({"query": "什么是黑洞"})
print(response)
