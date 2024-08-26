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
    {"shanglian": "春眠不觉晓", "xialian": "处处闻啼鸟"},
    {"shanglian": "红雨随心翻作浪", "xialian": "青山着意化为桥"},
]

# 接下来，我们指定用于格式化示例的模板。
example_formatter_template = """上联: {shanglian}
下联: {xialian}
"""

example_prompt = PromptTemplate(
    input_variables=["shanglian", "xialian"],
    template=example_formatter_template,
)


# 创建FewShotPromptTemplate对象
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="给出上联，并生成对应的下联\n",
    suffix="上联: {input}\n下联: ",
    input_variables=["input"],
    example_separator="\n",
)

# 使用format方法生成一个提示
formatted_prompt = few_shot_prompt.format(input="鸟鸣山更幽")

# 调用大模型生成下联
result = llm_tongyi_di(formatted_prompt)
print(result)
