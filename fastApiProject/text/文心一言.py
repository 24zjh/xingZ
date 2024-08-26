import os
import qianfan

# 动态获取最新模型列表依赖 IAM Access Key 进行鉴权，使用应用 AK 鉴权时不支持该功能
os.environ["QIANFAN_ACCESS_KEY"] = "ALTAKRiLe3OG7kyJbujNiYyHLS"
os.environ["QIANFAN_SECRET_KEY"] = "66f5477ab20e4aca8aeec4009f4ea84f"


# 定义消息类
class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}


# 定义聊天服务
class ChatService:
    def __init__(self, model_name):
        self.chat_comp = qianfan.ChatCompletion(model=model_name)

    def get_response(self, messages, top_p=0.8, temperature=0.9, penalty_score=1.0):
        formatted_messages = [msg.to_dict() for msg in messages]
        resp = self.chat_comp.do(
            messages=formatted_messages,
            top_p=top_p,
            temperature=temperature,
            penalty_score=penalty_score
        )
        return resp["result"]


# 定义聊天控制器
class ChatController:
    def __init__(self, chat_service):
        self.chat_service = chat_service
        self.chat_history = []

    def handle_message(self, user_input):
        self.chat_history.append(Message("user", user_input))
        response = self.chat_service.get_response(self.chat_history)
        self.chat_history.append(Message("assistant", response))
        return response


# 主函数
def main():
    # 模型名称可以通过 qianfan.ChatCompletion.models() 获取
    # 也可以在命令行运行 qianfan chat --list-model 查看
    model_name = "Yi-34B-Chat"

    chat_service = ChatService(model_name)
    chat_controller = ChatController(chat_service)

    print("与AI聊天，输入 'exit' 退出程序。")

    while True:
        user_input = input("你: ")
        if user_input.lower() == "exit":
            break
        response = chat_controller.handle_message(user_input)
        print("AI:", response)


if __name__ == "__main__":
    main()
