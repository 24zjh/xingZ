import os
import re
from bs4 import BeautifulSoup, Comment
import json

# 读取当前目录下的 Vue 文件内容
file_name = 'tourismProduct.vue'  # 替换为你的 Vue 文件名
file_path = os.path.join(os.getcwd(), file_name)

# 确保文件存在
if not os.path.exists(file_path):
    print(f"文件 {file_name} 不存在.")
else:
    with open(file_path, 'r', encoding='utf-8') as file:
        vue_content = file.read()

    # 使用BeautifulSoup解析Vue文件
    soup = BeautifulSoup(vue_content, 'html.parser')

    # 提取<template>标签内的内容
    template_content = soup.find('template')

    if template_content:
        # 删除所有注释内容
        for element in template_content(text=lambda text: isinstance(text, Comment)):
            element.extract()

        # 提取所有的文本内容
        text_elements = template_content.find_all(string=True)
        text_content = []
        for text in text_elements:
            # 去除多余空格和换行符
            cleaned_text = text.strip()
            # 去掉 \n 和多余空格等字符
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            # 过滤掉包含 {{ }} 的动态文本
            if cleaned_text and "{{" not in cleaned_text and "}}" not in cleaned_text:
                text_content.append(cleaned_text)

        # 创建JSON结构
        json_structure = {
            "cn": {
                "message": {}
            }
        }

        # 将提取出的文本内容保存为键值对
        for text in text_content:
            json_structure["cn"]["message"][text] = "Your translation here"  # 这里可以替换为实际的翻译

        # 将JSON结构转换为字符串并保存
        json_output = json.dumps(json_structure, ensure_ascii=False, indent=4)

        # 保存到文件
        output_file_name = f'{os.path.splitext(file_name)[0]}_translation.json'
        output_file_path = os.path.join(os.getcwd(), output_file_name)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(json_output)

        print(f"JSON 生成完毕并保存到: {output_file_path}\n", json_output)
    else:
        print("<template> 标签未找到.")
