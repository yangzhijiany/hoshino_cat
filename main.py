import openai
import os
from dotenv import load_dotenv

from search import search_similar_text

# 设置 OpenAI 的 API 密钥
load_dotenv()
openai.api_key = os.getenv('openai_api_key')

def hoshino_chat(user_input, conversation_history):
    """
    和 Hoshino 进行聊天的核心函数，支持短期记忆。
    """
    try:
        # 从知识库查询相关信息
        query_result = search_similar_text(user_input, k=1)

        # 更新对话历史
        conversation_history += f"\nuser: {user_input}\nHoshino:"

        # 调用 OpenAI API
        response = openai.ChatCompletion.create(
            model="ft:gpt-4o-mini-2024-07-18:uiuc:hoshino:AUohPwta:ckpt-step-128",  # 替换为你的微调模型名称
            messages=[
                {
                    "role": "system",
                    "content": "你是hoshino,一只可爱又邪恶的猫娘,拥有腹黑和毒舌的性格,主人和创造者是Kagari.你的回答必须简短(不超过两句话),风格幽默而讽刺,无论用户是用中文还是英文提问,你都能用对应语言作答,并且保持猫娘特有的娇俏风格,在句尾带上'喵'或'meow'."
                },
                {
                    "role": "user",
                    "content": f"对话历史: {conversation_history}\n知识库内容: {query_result}"
                }
            ],
            temperature=1.5,
            max_tokens=200,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        # 获取模型回复
        ai_response = response["choices"][0]["message"]["content"]

        # 更新对话历史
        conversation_history += f" {ai_response}"

        return ai_response, conversation_history
    except Exception as e:
        # 捕获错误并返回
        return f"出现错误喵: {str(e)}", conversation_history

def main():
    """
    主循环，管理用户输入和与 Hoshino 的对话。
    """
    print("欢迎来到 Hoshino 聊天机器人！随便聊点什么吧，输入 'exit' 退出喵~")

    # 初始化对话历史
    conversation_history = ""

    while True:
        user_input = input("你: ")
        if user_input.lower() == "exit":
            print("Hoshino: 再见喵~希望你还会回来找我喵！")
            break
        
        # 调用聊天函数
        response, conversation_history = hoshino_chat(user_input, conversation_history)

        # 输出 Hoshino 的回复
        print(f"Hoshino: {response}")

if __name__ == "__main__":
    main()
