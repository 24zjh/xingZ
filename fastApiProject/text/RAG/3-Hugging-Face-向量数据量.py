from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
import requests
import os

class DownloadProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_with_progress(url, filepath):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(filepath, 'wb') as file, DownloadProgressBar(
            desc=os.path.basename(filepath),
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download_model(model_name, cache_dir):
    model_url = f"https://huggingface.co/{model_name}/resolve/main/pytorch_model.bin"
    config_url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
    tokenizer_url = f"https://huggingface.co/{model_name}/resolve/main/tokenizer.json"

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    download_with_progress(model_url, os.path.join(cache_dir, "pytorch_model.bin"))
    download_with_progress(config_url, os.path.join(cache_dir, "config.json"))
    download_with_progress(tokenizer_url, os.path.join(cache_dir, "tokenizer.json"))

# 下载模型
model_name = "sentence-transformers/all-MiniLM-L6-v2"
cache_dir = "./cache/all-MiniLM-L6-v2"
download_model(model_name, cache_dir)

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(cache_dir)
model = AutoModel.from_pretrained(cache_dir)

# 嵌入查询
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

embeddings_hf = HuggingFaceEmbeddings(model_name=cache_dir)
query1 = "狗"
query2 = "猫"
query3 = "雨"

emb1 = embeddings_hf.embed_query(query1)
emb2 = embeddings_hf.embed_query(query2)
emb3 = embeddings_hf.embed_query(query3)

emb1 = np.array(emb1)
emb2 = np.array(emb2)
emb3 = np.array(emb3)

print(np.dot(emb1, emb2))
print(np.dot(emb3, emb2))
print(np.dot(emb1, emb3))
