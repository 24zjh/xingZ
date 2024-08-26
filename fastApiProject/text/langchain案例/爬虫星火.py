from langchain.prompts import PromptTemplate
from langchain.chains import LLMRequestsChain, LLMChain
import os
from langchain_community.llms import SparkLLM

# 设置API密钥
os.environ["IFLYTEK_SPARK_APP_ID"] = "7fc4114e"
os.environ["IFLYTEK_SPARK_API_KEY"] = "0537b19ffd7b25bf4e3114325898e284"
os.environ["IFLYTEK_SPARK_API_SECRET"] = "OTkxMTNkMDI5ODkwYjk0YTQ3YmRkYTUx"



#

# 定义Prompt模板
template = """
在 >>> 和 <<< 之间是网页的返回的HTML内容。
网页是新浪财经A股上市公司的公司简介。
请抽取参数请求的信息。

>>> {requests_result} <<<
请使用如下的JSON格式返回数据
{{
  "company_name":"a",
  "company_english_name":"b",
  "issue_price":"c",
  "date_of_establishment":"d",
  "registered_capital":"e",
  "office_address":"f",
  "Company_profile":"g"
}}
Extracted:
"""

prompt = PromptTemplate(
    input_variables=["requests_result"],
    template=template
)

# 初始化SparkLLM模型
llm_spark = SparkLLM()
chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm_spark, prompt=prompt))

# 定义输入
inputs = {
    "url": "https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/600519.phtml"
}

# 获取响应并打印结果
response = chain(inputs)
print(response['output'])
