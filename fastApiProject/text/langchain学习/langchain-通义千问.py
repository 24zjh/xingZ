# import os
# from langchain_community.llms import Tongyi
#
# os.environ["DASHSCOPE_API_KEY"] = "sk-39a2d85e79a5493a8859b60b725e5e55"
#
# try:
#     llm_tongyi = Tongyi()
#     response = llm_tongyi.invoke("帮我写个爬虫代码")
#     print(response)
# except Exception as e:
#     print(f"Error: {e}")
# pip install langchain-community --trusted-host mirrors.cloud.tencent.com
# pip install dashscope

import os
from langchain_community.llms import Tongyi

# 设置API密钥
os.environ["DASHSCOPE_API_KEY"] = "sk-39a2d85e79a5493a8859b60b725e5e55"

# 初始化Tongyi模型
llm_tongyi = Tongyi()

# 构建提示
prompt = """
请帮我介绍下神经网络的原理，相关公式请使用latex表达，并展示相关python程序，程序实现请使用pytorch。
"""

