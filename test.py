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
                        "The user's response or query, if it contains names or real-life characters or terminologies,"
                        "it's likely that Knowledge Base is needed."
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
    
def getSummary(hist):
    """
    和 Hoshino 进行聊天的核心函数。
    """
    try:
        # 调用 OpenAI API
        response = openai.ChatCompletion.create(
            model="ft:gpt-4o-mini-2024-07-18:uiuc:hoshino:AUohPwta:ckpt-step-128",
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是Hoshino, 一只可爱又邪恶的猫娘, 以下是你最近的一个聊天记录, 总结这个聊天记录以便于你今后查看"
                        "请用一段完整的话总结，并且用第三人称总结"
                    ),
                }, 
                {
                    "role": "user",
                    "content": hist
                }
            ],
            temperature=1.4,
            max_tokens=500,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )
        # 返回聊天响应
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        # 捕获错误并返回
        return f"出现错误喵: {str(e)}"
    
def helper2(user_input):
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
                        "You are an AI that decides the probability of needing to call a knowledge base or conversation history. "
                        "Output:"
                        "Two probabilities: [Knowledge Base Call Probability, History Call Probability], each represented as a decimal between 0.00 and 1.00. The example output: [0.90, 0.25]"
                        "Rules:"
                        "1. If the query contains names, terms, or requires background knowledge, assign a higher Knowledge Base Call Probability (closer to 1.00); otherwise, assign a lower value (closer to 0.00).\n"
                        "Example input: 'What is the anime Mygo?' Output: [0.95, 0.5], because Mygo is a terminology/name."
                        "2. If the query involves facts about the user, yourself, or context from prior conversation, assign a higher History Call Probability; otherwise, assign a lower value."
                        "Example input: 'Do you know me?' Output: [0.1, 0.9], because it is asking your opinion."
                        "3. If the query contains names/terms and also asks about you, assign high probabilities to both."
                        "Example input: 'Do you like the anime Mygo?' Output: [0.8, 0.8], because Mygo is a name/term, and it is asking about you."
                        "4. If neither is needed, assign both probabilities values closer to 0.00."
                        "Ensure the probabilities reflect uncertainty, allowing overlap if both sources are relevant."
                    )
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
        hist = input("你: ")
        if hist.lower() == "exit":
            print("Hoshino: 再见喵~希望你还会回来找我喵！")
            break
        response = helper(hist)
        knowledge_base_call = response[1]
        history_call = response[4]
        print(knowledge_base_call)
        print(history_call)
        print(f"Hoshino: {response}")

if __name__ == "__main__":
    main()