import os
import qianfan
from models.text import TextResponse

# Set environment variables for Qianfan SDK
os.environ["QIANFAN_ACCESS_KEY"] = "ALTAKRiLe3OG7kyJbujNiYyHLS"
os.environ["QIANFAN_SECRET_KEY"] = "66f5477ab20e4aca8aeec4009f4ea84f"

def get_text_response(content: str) -> str:
    chat = qianfan.ChatCompletion(model="Yi-34B-Chat")
    resp = chat.do(messages=[{"role": "user", "content": content}],
                   top_p=0.8,
                   temperature=0.9,
                   penalty_score=1.0,
                   )
    print(resp["result"])
    return resp["result"]
