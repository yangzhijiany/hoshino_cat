import openai
import os
from dotenv import load_dotenv

from search import add_text_to_database_and_index
from search import search_similar_text

# 设置 OpenAI 的 API 密钥
load_dotenv()
openai.api_key = os.getenv('openai_api_key')

# DB_NAME = "embeddings.db"
# FAISS_INDEX_FILE = "faiss_index.bin"
# EMBEDDING_DIM = 1536  # 假设使用 text-embedding-ada-002

def hoshino_chat(user_input):
    """
    和 Hoshino 进行聊天的核心函数。
    """
    try:
        query_result = search_similar_text(user_input, k=1)
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
                    "content": f"User's_input: {user_input}\nDatabase_context: {query_result}"
                }
            ],
            temperature=1.65,
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
