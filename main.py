import openai
import os
from dotenv import load_dotenv
import datetime

from search import add_text_to_database_and_index
from search import search_similar_text
from summary import search_similar_text_history
from summary import add_text_to_database_and_index_history
from test import helper, helper2
from test import getSummary

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
            "用户的输入有三个内容:当前输入,从知识库获取的数据,从历史记录中获取的数据.你需要自行判断是否需要用到后面两个内容。"
        ),
    }
]

message_string = ""
remember_keywords = ["记住这个", "添加知识", "储存信息", "保存这个", "记下这点", "存一下", "把这个记下来", "记住这一点", "你需要记住这个"]

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
        history_result = ""

        check_use = helper2(user_input)
        probabilities = check_use.strip("[]").split(", ")
        prob_db = float(probabilities[0])
        prob_history = float(probabilities[1])

        global remember_keywords
        if any(keyword in user_input for keyword in remember_keywords):
            add_text_to_database_and_index(user_input)
            print("Knowledge Base Updated")

        print(check_use)
        if prob_db >= 0.5:
            query_result = search_similar_text(user_input, k=2)
        else:
            query_result = ""
        if prob_history >= 0.6:
            history_result = search_similar_text_history(user_input, k=2)
        else:
            history_result = ""

        messages.append({"role": "user", "content": f"User's_input: {user_input}\nDatabase_context: {query_result}\nDatabase_context: {history_result}"})
        global message_string
        message_string += f"User: {user_input}\n"
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
                message_string += f"Hoshino: {assistant_reply}\n"
                return assistant_reply
        else:
            assistant_reply = assistant_message["content"]
            messages.append({"role": "assistant", "content": assistant_reply})
            message_string += f"Hoshino: {assistant_reply}\n"
            return assistant_reply

    except Exception as e:
        # 捕获错误并返回
        return f"出现错误喵: {str(e)}"
    

def save_summary_to_file(timestamp, summary):
    """
    Save the conversation summary to a file in the /data directory.
    """
    os.makedirs("data", exist_ok=True)
    filename = f"data/{timestamp.replace(':', '-')}.txt"
    with open(filename, "w", encoding="utf-8") as file:
        his_string = (f"Timestamp: {timestamp}, {summary}\n")
        file.write(his_string)
    return his_string

# 主循环
def main():
    print("欢迎来到 Hoshino 聊天机器人！随便聊点什么吧，输入 'exit' 退出喵~")
    while True:
        user_input = input("你: ")
        if user_input.lower() == "exit":
            his_string = save_summary_to_file(get_time_tool(), getSummary(message_string))
            add_text_to_database_and_index_history(his_string)
            print("Hoshino: 再见喵~希望你还会回来找我喵！")
            break
        response = hoshino_chat(user_input)
        print(f"Hoshino: {response}")

if __name__ == "__main__":
    main()