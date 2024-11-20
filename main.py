import openai
import os
from dotenv import load_dotenv
import datetime

load_dotenv()
openai.api_key = os.getenv('openai_api_key')

def get_current_time():
    """返回当前时间的字符串"""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

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

# 初始化消息历史
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

# 主聊天函数
def hoshino_chat(user_input):
    """
    和 Hoshino 聊天，支持 Function Calling。
    """
    try:
        # 用户输入加入对话
        messages.append({"role": "user", "content": user_input})

        # 调用 OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=messages,
            functions=functions,
            function_call="auto"  # 自动检测是否需要调用函数
        )

        if response["choices"][0]["message"].get("function_call"):
            function_call = response["choices"][0]["message"]["function_call"]
            function_name = function_call["name"]
            if function_name == "get_time":
                result = get_current_time()
                messages.append({
                    "role": "function",
                    "name": "get_time",
                    "content": result
                })

                final_response = openai.ChatCompletion.create(
                    model="gpt-4-0613",
                    messages=messages
                )
                return final_response["choices"][0]["message"]["content"]

        assistant_reply = response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

    except Exception as e:
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
