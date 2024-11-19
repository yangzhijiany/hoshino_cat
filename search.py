import sqlite3
import faiss
import numpy as np
import openai
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
openai.api_key = os.getenv('openai_api_key')

# 数据库和索引文件名
DB_NAME = "embeddings.db"
FAISS_INDEX_FILE = "faiss_index.bin"
EMBEDDING_DIM = 1536  # 假设使用 text-embedding-ada-002

# 加载或初始化 FAISS 索引
if os.path.exists(FAISS_INDEX_FILE):
    index = faiss.read_index(FAISS_INDEX_FILE)
    print(f"已加载本地FAISS索引：{FAISS_INDEX_FILE}")
    print(f"Index type after loading: {type(index)}")
else:
    index_flat = faiss.IndexFlatL2(EMBEDDING_DIM)
    index = faiss.IndexIDMap2(index_flat)
    print(f"创建新的FAISS索引：{FAISS_INDEX_FILE}")

def get_embedding(text):
    """
    使用 OpenAI API 生成文本的嵌入向量。
    """
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def add_text_to_database_and_index(text):
    """
    添加文本到 SQLite 数据库和 FAISS 索引。
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 创建表如果不存在
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT UNIQUE
        )
    """)

    # 检查文本是否已存在
    cursor.execute("SELECT id FROM documents WHERE text = ?", (text,))
    result = cursor.fetchone()
    if result:
        doc_id = result[0]
        print(f"文本已存在数据库中，ID: {doc_id}")
    else:
        # 插入文本到数据库
        cursor.execute("INSERT INTO documents (text) VALUES (?)", (text,))
        doc_id = cursor.lastrowid
        conn.commit()
        print(f"新增文本到数据库，ID: {doc_id}")

    # 检查嵌入是否已存在于索引
    existing_ids = faiss.vector_to_array(index.id_map) if index.ntotal > 0 else np.array([])
    if doc_id not in existing_ids:
        # 生成嵌入向量并添加到索引
        embedding = get_embedding(text)
        embedding_np = np.array([embedding]).astype('float32')
        ids = np.array([doc_id]).astype('int64')  # FAISS 需要 int64 类型的 ID
        index.add_with_ids(embedding_np, ids)

        # 保存索引到文件
        faiss.write_index(index, FAISS_INDEX_FILE)
        print(f"新增嵌入到索引，ID: {doc_id}")
    else:
        print(f"嵌入已存在于索引中，ID: {doc_id}")

    conn.close()
    return doc_id

def query_text_by_id(doc_id):
    """
    根据文档 ID 查询文本内容。
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT text FROM documents WHERE id = ?", (doc_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return result[0]
    else:
        print(f"未找到ID为 {doc_id} 的文本。")
        return None

def search_similar_text(query, k=3):
    """
    根据查询文本，在索引中查找最相似的文本。
    """
    query_embedding = get_embedding(query)
    query_embedding_np = np.array([query_embedding]).astype('float32')

    # 搜索最相似的 k 个向量
    distances, indices = index.search(query_embedding_np, k)

    # 根据索引查询原始文本
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    results = []
    for idx in indices[0]:
        cursor.execute("SELECT text FROM documents WHERE id = ?", (int(idx),))
        result = cursor.fetchone()
        if result:
            results.append(result[0])
        else:
            print(f"未找到ID为 {int(idx)} 的文本。")
    conn.close()
    # print(f"搜索返回的距离: {distances}")
    # print(f"搜索返回的索引: {indices}")

    return results

if __name__ == "__main__":
    # 示例：添加文本
    texts = [
        "David是Kagari的朋友,他在加拿大的UBC上学,专业是计算机科学."
    ]
    for text in texts:
        add_text_to_database_and_index(text)

    # 示例：查询 ID 为 1 的文本
    print(query_text_by_id(1))

    # 示例：搜索最相似的文本
    query = "David"
    results = search_similar_text(query, k=2)
    print("最相似的文本：")
    for result in results:
        print(result)
