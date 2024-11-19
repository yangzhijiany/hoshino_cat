import sqlite3

def initialize_database(db_name="embeddings.db"):
    """
    初始化 SQLite 数据库，创建所需表。
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 创建 documents 表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL
    )
    ''')

    # 提交更改并关闭连接
    conn.commit()
    conn.close()
    print(f"数据库 {db_name} 初始化完成，并已创建表 documents。")

if __name__ == "__main__":
    initialize_database()
