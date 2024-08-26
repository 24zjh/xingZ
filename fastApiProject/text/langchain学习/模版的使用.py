# ## 模版的使用
#
# ### 简单模版
#
# 语言模型以文本作为输入，这段文本通常被称为提示（Prompt）。通常情况下，这不仅仅是一个硬编码的字符串，而是模板、示例和用户输入的组合。LangChain提供了多个类和函数，以便轻松构建和处理提示。
#
# 提示模板是指一种可复制的生成提示的方式。它包含一个文本字符串（模板），可以从最终用户处接收一组参数并生成提示。提示模板可能包含以下内容：
#
# * 对语言模型的指令
# * 少量示例，以帮助语言模型生成更好的回复
# * 对语言模型的问题
#
# 原文链接： https://machinelearning.blog.csdn.net/article/details/131988052


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


from langchain import PromptTemplate

template = """
我希望你能充当新公司的命名顾问。
一个生产{product}的公司的好名字是什么？
一个生产{product1}的公司的好名字是什么？
"""

prompt = PromptTemplate(
    input_variables=["product","product1"],
    template=template,
)
prompt.format(product="彩色袜子",product1="袜子")


print(prompt)

# 多个输入变量的示例提示
multiple_input_prompt = PromptTemplate(
    input_variables=["adjective", "content"],
    template="给我讲一个{adjective}的关于{content}的笑话。"
)
multiple_input_prompt.format(adjective="有趣的", content="小鸡")

print(multiple_input_prompt)

# ###  包含例子的模版
#
# 向模板添加Few Shot示例
# Few Shot示例是一组可以帮助语言模型生成更好响应的示例。要使用Few Shot示例生成提示，可以使用FewShotPromptTemplate。此类接受PromptTemplate和Few Shot示例列表。然后，它使用Few Shot示例格式化提示模板。
#
# 在下面示例中，我们将创建一个生成单词反义词的提示

from langchain import PromptTemplate, FewShotPromptTemplate

# 首先，创建Few Shot示例列表
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

# 接下来，我们指定用于格式化示例的模板。
# 我们使用`PromptTemplate`类来实现这个目的。
example_formatter_template = """Word: {word}
Antonym: {antonym}
"""

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template,
)

# 最后，创建`FewShotPromptTemplate`对象。
few_shot_prompt = FewShotPromptTemplate(
    # 这些是我们要插入到提示中的示例。
    examples=examples,
    # 这是我们在将示例插入到提示中时要使用的格式。
    example_prompt=example_prompt,
    # 前缀是出现在提示中示例之前的一些文本。
    # 通常，这包括一些说明。
    prefix="Give the antonym of every input\n",
    # 后缀是出现在提示中示例之后的一些文本。
    # 通常，这是用户输入的地方。
    suffix="Word: {input}\nAntonym: ",
    # 输入变量是整个提示期望的变量。
    input_variables=["input"],
    # 示例分隔符是我们将前缀、示例和后缀连接在一起的字符串。
    example_separator="\n",
)

# 现在，我们可以使用`format`方法生成一个提示。
print(few_shot_prompt.format(input="big"))

print(llm_tongyi_di(few_shot_prompt.format(input="美丽")))

