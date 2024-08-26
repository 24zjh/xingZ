import os
import json
from langchain_community.llms import Tongyi, QianfanLLMEndpoint, SparkLLM
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser

# 设置 API 密钥
os.environ["DASHSCOPE_API_KEY"] = "sk-39a2d85e79a5493a8859b60b725e5e55"
os.environ["IFLYTEK_SPARK_APP_ID"] = "7fc4114e"
os.environ["IFLYTEK_SPARK_API_KEY"] = "0537b19ffd7b25bf4e3114325898e284"
os.environ["IFLYTEK_SPARK_API_SECRET"] = "OTkxMTNkMDI5ODkwYjk0YTQ3YmRkYTUx"
os.environ["QIANFAN_AK"] = "7aSK7bReecqYMjpHtHHowyRM"
os.environ["QIANFAN_SK"] = "8VCRFZhDfIWHa6UYHNTa7hWaYBk2DTsG"

# 原始数据
data = {
    "cn": {
        "message": {
            "旅游网站": "",
            "首页": "",
            "目的地": "",
            "旅游产品": "",
            "关于我们": "",
            "更多内容": "",
            "欢迎来到旅游主页": "",
            "我们提供全球各地最美的旅游景点，带你畅游世界的旅游美景": "",
            "查看更多": "",
            "预定旅行": "",
            "专业团队": "",
        }
    }
}

# 提取所有键
keys_set = set(data['cn']['message'].keys())

# 输出逗号分隔的键
formatted_keys = ', '.join(keys_set)

# 创建输出解析器
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

# 初始化语言模型
llm_spark = SparkLLM(temperature=0.1)

# 创建提示模板
prompt = PromptTemplate(
    input_variables=["words", "language"],
    template="用简洁的语言将以下内容翻译成{language}，输出时不包含任何引号或额外符号，并使用以下结构输出:{format_instructions}。需要翻译的内容如下：{words}",
    partial_variables={"format_instructions": format_instructions}
)

# 生成输入提示
_input = prompt.format(words=formatted_keys, language="意大利")

# 调用语言模型进行翻译
output = llm_spark.invoke(_input)

# 解析模型输出
parsed_output = output_parser.parse(output)

def clean_and_split_output(output_list):
    """
    清理输出结果，移除多余的引号和空格，替换全角逗号为半角逗号，并根据逗号分割为独立项。
    """
    cleaned_output = []
    for item in output_list:
        # 移除首尾的引号，并替换全角逗号为半角逗号
        cleaned_item = item.replace('\"', '').replace('，', ',').strip()
        cleaned_output.extend(cleaned_item.split(','))  # 根据逗号分割并添加到输出列表
    return [item.strip() for item in cleaned_output]  # 再次移除可能的多余空格

# 处理并清理输出
final_output = clean_and_split_output(parsed_output)

# 将键值对重新组合成字典
final_dict = {key: value for key, value in zip(keys_set, final_output)}

# 生成最终的 JSON 数据
final_json = {
    "cn": {
        "message": final_dict
    }
}

# 输出最终的 JSON 数据
print(json.dumps(final_json, ensure_ascii=False, indent=4))
