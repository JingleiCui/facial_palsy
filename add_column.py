import sqlite3
from pathlib import Path

conn = sqlite3.connect('facialPalsy.db')
cur = conn.cursor()
cur.execute("ALTER TABLE video_features ADD COLUMN visual_global_features BLOB")

conn.commit()
conn.close()