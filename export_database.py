import os
import sqlite3
import pandas as pd

db_path = 'facialPalsy.db'

output_folder = '/Users/cuijinglei/Documents/facialPalsy_old/analyze_results/HGFA/database'
os.makedirs(output_folder, exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql_query(query, conn)

for table_name in tables['name']:

    print(f"正在导出表: {table_name}")
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql_query(query, conn)

    output_excel_path = os.path.join(output_folder, f"{table_name}.xlsx")

    df.to_excel(output_excel_path, index=False, engine='openpyxl')
    print(f"数据表{table_name} 已保存到 {output_excel_path}")

conn.close()

print("所有表的数据已成功导出到单独的Excel文件中。")
