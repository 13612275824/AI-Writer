import sqlite3

conn = sqlite3.connect("data/vectors/chroma.sqlite3")
cursor = conn.cursor()

# 列出所有表
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("=== 数据库中的表 ===")
for t in tables:
    print(f"  - {t[0]}")

print()

# 每张表显示前5行
for t in tables:
    table_name = t[0]
    print(f"--- {table_name} (前5行) ---")
    cursor.execute(f"SELECT * FROM [{table_name}] LIMIT 5")
    rows = cursor.fetchall()
    # 获取列名
    col_names = [desc[0] for desc in cursor.description]
    print(f"  列: {col_names}")
    for row in rows:
        print(f"  {row}")
    print()

conn.close()
