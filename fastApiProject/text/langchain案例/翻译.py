import json
import os
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi

# 设置API密钥
os.environ["DASHSCOPE_API_KEY"] = "sk-39a2d85e79a5493a8859b60b725e5e55"

# 中文JSON集合
chinese_json = {
    "姓名": "小明",
    "公司": "华为",
    "职位": "工程师"
}

# 翻译模板
translation_template = ChatPromptTemplate.from_template("将以下JSON数据从中文翻译成{language}：\n{json_data}")

# 翻译模型实例
model = ChatTongyi()

# 定义函数进行翻译
def translate_json(chinese_json, language):
    json_str = json.dumps(chinese_json, ensure_ascii=False)  # 将JSON转为字符串
    prompt = translation_template.format(language=language, json_data=json_str)
    translated_json_str = model(prompt)  # 进行翻译
    translated_json = json.loads(translated_json_str)  # 将翻译后的字符串转为JSON
    return translated_json

# 翻译为英文和日语
english_json = translate_json(chinese_json, "英文")
japanese_json = translate_json(chinese_json, "日语")

# 输出翻译后的JSON集合
print("英文 JSON:", json.dumps(english_json, ensure_ascii=False, indent=4))
print("日语 JSON:", json.dumps(japanese_json, ensure_ascii=False, indent=4))


from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.prompts import ChatPromptTemplate

# 初始化模型
model = ChatTongyi()

# 创建翻译模板
template = ChatPromptTemplate.from_template("将以下JSON数据从中文翻译成{language}：\n{json_data}")

# LCEL 表达式
{
    "中文 JSON": {"姓名": "小明", "公司": "华为", "职位": "工程师"},
    "英文 JSON": (template.format(language="英文", json_data=json.dumps({"姓名": "小明", "公司": "华为", "职位": "工程师"}, ensure_ascii=False)) | model | StrOutputParser() | json.loads),
    "日语 JSON": (template.format(language="日语", json_data=json.dumps({"姓名": "小明", "公司": "华为", "职位": "工程师"}, ensure_ascii=False)) | model | StrOutputParser() | json.loads)
}
