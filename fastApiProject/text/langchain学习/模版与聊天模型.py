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

prompt0="请写一首有关大理的四言诗"

print(llm_tongyi_di.invoke(prompt0))


print(llm_tongyi_zh.invoke(prompt0))

print(llm_tongyi_ga.invoke(prompt0))

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

### 参数streaming——流式输出
llm_wx_liu = QianfanLLMEndpoint(streaming=True, callbacks=[StreamingStdOutCallbackHandler()],temperature=0.1)

resp = llm_wx_liu.invoke("帮我写一首有关西湖的歌")

# llm_tongyi_ga.get_num_tokens("fasfdsafreqwrwe") 不可用

### 函数generate————多个提示词的支持

res0=llm_tongyi_di.generate(["写一首四句的诗","写一幅对联"])

print(res0)

print(res0.generations[0][0].generation_info)












