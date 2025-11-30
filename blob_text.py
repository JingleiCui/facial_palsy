import sqlite3, numpy as np

conn = sqlite3.connect("facialPalsy.db")
cur = conn.cursor()
cur.execute("""
SELECT video_id, motion_dim, motion_features
FROM video_features
WHERE motion_features IS NOT NULL
""")
rows = cur.fetchall()
conn.close()

for video_id, motion_dim, blob in rows:
    arr = np.frombuffer(blob, dtype=np.float32)
    print(video_id, "motion_dim=", motion_dim, "len=", len(arr), "values=", arr.tolist())
