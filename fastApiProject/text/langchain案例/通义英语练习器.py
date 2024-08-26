
#输入三个模型各自的key

import os
from langchain_community.llms import Tongyi
from langchain_community.llms import SparkLLM
from langchain_community.llms import QianfanLLMEndpoint

os.environ["DASHSCOPE_API_KEY"] = "sk-39a2d85e79a5493a8859b60b725e5e55"

os.environ["IFLYTEK_SPARK_APP_ID"] = "7fc4114e"
os.environ["IFLYTEK_SPARK_API_KEY"] = "0537b19ffd7b25bf4e3114325898e284"
os.environ["IFLYTEK_SPARK_API_SECRET"] = "OTkxMTNkMDI5ODkwYjk0YTQ3YmRkYTUx"

os.environ["QIANFAN_AK"] = "7aSK7bReecqYMjpHtHHowyRM"
os.environ["QIANFAN_SK"] = "8VCRFZhDfIWHa6UYHNTa7hWaYBk2DTsG"

# 参数 Temperature

llm_tongyi_di=Tongyi(temperature=0.1)
llm_tongyi_zh=Tongyi(temperature=0.5)
llm_tongyi_ga=Tongyi(temperature=1)


from langchain import PromptTemplate, FewShotPromptTemplate

# 首先，创建Few Shot示例列表
examples = [
    {"words": ["happiness", "family", "holiday"], "essay": "Happiness is often found in the simple moments we share with our family, especially during the holidays. These times bring us closer together and create lasting memories."},
    {"words": ["technology", "future", "education"], "essay": "Technology is rapidly transforming the future of education. With the advent of new tools and platforms, learning is becoming more accessible and engaging for students around the world."},
]

# 接下来，我们指定用于格式化示例的模板。
example_formatter_template = """Words: {words}
Essay: {essay}
"""

example_prompt = PromptTemplate(
    input_variables=["words", "essay"],
    template=example_formatter_template,
)

# 创建FewShotPromptTemplate对象
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Given the following words, write a short essay that incorporates all of them:\n",
    suffix="Words: {input}\nEssay: ",
    input_variables=["input"],
    example_separator="\n",
)

# 使用format方法生成一个提示
input_words = ["friendship", "trust", "adventure"]
formatted_prompt = few_shot_prompt.format(input=", ".join(input_words))

# 调用大模型生成作文
result = llm_tongyi_di(formatted_prompt)
print(result)
