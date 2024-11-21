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
DB_NAME = "chat_history.db"
FAISS_INDEX_FILE = "faiss_history_index.bin"
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

def get_embedding_history(text):
    """
    使用 OpenAI API 生成文本的嵌入向量。
    """
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def add_text_to_database_and_index_history(text):
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
        embedding = get_embedding_history(text)
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

def query_text_by_id_history(doc_id):
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

def search_similar_text_history(query, k=3):
    """
    根据查询文本，在索引中查找最相似的文本。
    """
    query_embedding = get_embedding_history(query)
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

def sync_index_with_database_history():
    """
    同步 FAISS 索引与 SQLite 数据库的内容。
    删除索引中不存在于数据库的嵌入。
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 获取数据库中所有的有效 ID
    cursor.execute("SELECT id FROM documents")
    db_ids = set(row[0] for row in cursor.fetchall())
    conn.close()
    
    # 获取索引中的所有 ID
    index_ids = faiss.vector_to_array(index.id_map) if index.ntotal > 0 else np.array([])
    
    # 找出需要删除的 ID
    ids_to_remove = [int(id_) for id_ in index_ids if id_ not in db_ids]
    
    if ids_to_remove:
        print(f"从索引中移除以下ID：{ids_to_remove}")
        index.remove_ids(np.array(ids_to_remove).astype('int64'))
        # 保存更新后的索引
        faiss.write_index(index, FAISS_INDEX_FILE)
    else:
        print("索引和数据库已经同步，无需更新。")

def rebuild_index_from_database_history():
    """
    从 SQLite 数据库重新构建索引。
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 清空现有索引
    index.reset()
    
    # 遍历数据库内容并重新添加到索引
    cursor.execute("SELECT id, text FROM documents")
    for doc_id, text in cursor.fetchall():
        embedding = get_embedding_history(text)
        embedding_np = np.array([embedding]).astype('float32')
        ids = np.array([doc_id]).astype('int64')
        index.add_with_ids(embedding_np, ids)
    
    # 保存索引
    faiss.write_index(index, FAISS_INDEX_FILE)
    print("索引已重新构建。")
    conn.close()

if __name__ == "__main__":
    # 示例：添加文本
    texts = [
        "Timestamp: 2024-11-20 19:49:58, Hoshino 解释了千早爱音的身份,并补充了她是 mygo 乐队的吉他手以及羽丘女子学园一年级的优秀学姐。"
    ]
    for text in texts:
        add_text_to_database_and_index_history(text)

    # 示例：查询 ID 为 1 的文本
    print(query_text_by_id_history(1))

    # 示例：搜索最相似的文本
    query = "千早爱音"
    results = search_similar_text_history(query, k=1)
    print("最相似的文本：")
    for result in results:
        print(result)
