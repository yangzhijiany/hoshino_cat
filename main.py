import openai
import os
from dotenv import load_dotenv
import datetime

from search import add_text_to_database_and_index
from search import search_similar_text
from test import helper

# 设置 OpenAI 的 API 密钥
load_dotenv()
openai.api_key = os.getenv('openai_api_key')

functions = [
    {
        "name": "get_time",
        "description": "获取当前时间。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

# 初始化全局 messages
messages = [
    {
        "role": "system",
        "content": (
            "你是hoshino,一只可爱又邪恶的猫娘,拥有腹黑和毒舌的性格,主人和创造者是Kagari."
            "你的回答必须简短(不超过两句话),风格幽默而讽刺,无论用户是用中文还是英文提问,"
            "你都能用对应语言作答,并且保持猫娘特有的娇俏风格,在句尾带上'喵'或'meow'."
        ),
    }
]

def get_time_tool():
    """工具：返回当前时间"""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def hoshino_chat(user_input):
    """
    和 Hoshino 进行聊天的核心函数，支持短期记忆。
    """
    try:
        query_result = ""
        query_result = search_similar_text(user_input, k=1)
        check_database = helper(user_input)[1]
        if check_database == "1":
            query_result = search_similar_text(user_input, k=1)
        else:
            query_result = ""

        messages.append({"role": "user", "content": f"User's_input: {user_input}\nDatabase_context: {query_result}"})

        response = openai.ChatCompletion.create(
            model="ft:gpt-4o-mini-2024-07-18:uiuc:hoshino:AUohPwta:ckpt-step-128",
            messages=messages,
            functions=functions,
            temperature=1.65,
            max_tokens=200,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )

        assistant_message = response["choices"][0]["message"]

        if "function_call" in assistant_message:
            function_call = assistant_message["function_call"]
            function_name = function_call["name"]
            if function_name == "get_time":
                tool_result = get_time_tool()

                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": tool_result
                })

                final_response = openai.ChatCompletion.create(
                    model="ft:gpt-4o-mini-2024-07-18:uiuc:hoshino:AUohPwta:ckpt-step-128",
                    messages=messages
                )
                assistant_reply = final_response["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": assistant_reply})
                return assistant_reply
        else:
            assistant_reply = assistant_message["content"]
            messages.append({"role": "assistant", "content": assistant_reply})
            return assistant_reply

    except Exception as e:
        # 捕获错误并返回
        return f"出现错误喵: {str(e)}"

# 主循环
def main():
    print("欢迎来到 Hoshino 聊天机器人！随便聊点什么吧，输入 'exit' 退出喵~")
    while True:
        user_input = input("你: ")
        if user_input.lower() == "exit":
            print("Hoshino: 再见喵~希望你还会回来找我喵！")
            break
        response = hoshino_chat(user_input)
        print(f"Hoshino: {response}")

if __name__ == "__main__":
    main()