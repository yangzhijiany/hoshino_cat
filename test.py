import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('openai_api_key')


def helper(user_input):
    """
    和 Hoshino 进行聊天的核心函数。
    """
    try:
        # 调用 OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # 替换为你的微调模型名称
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI that decides whether to call a knowledge base or conversation history. "
                        "Input: "
                        "1. User Input: Current query or statement. "
                        "2. Knowledge Base (optional): Background knowledge. "
                        "3. Conversation History (optional): Prior exchanges summary. "
                        "Output: "
                        "- Two integers: [Knowledge Base Call, History Call]. "
                        "- 1 = call needed, 0 = not needed. "
                        "Rules: "
                        "1. If the query needs background knowledge, Knowledge Base Call = 1, else 0. "
                        "2. If the query needs prior context, History Call = 1, else 0. "
                        "3. If neither is needed, output [0, 0]."
                    ),
                }, 
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            temperature=0,
            max_tokens=200,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )
        # 返回聊天响应
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        # 捕获错误并返回
        return f"出现错误喵: {str(e)}"
    

def main():
    print("欢迎来到 helper！随便聊点什么吧，输入 'exit' 退出喵~")
    while True:
        user_input = input("你: ")
        if user_input.lower() == "exit":
            print("Hoshino: 再见喵~希望你还会回来找我喵！")
            break
        response = helper(user_input)
        knowledge_base_call = response[1]
        history_call = response[4]
        print(knowledge_base_call)
        print(history_call)
        print(f"Hoshino: {response}")

if __name__ == "__main__":
    main()