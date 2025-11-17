import sqlite3
from pathlib import Path

conn = sqlite3.connect('facialPalsy.db')
cur = conn.cursor()
cur.execute("SELECT video_id, peak_frame_path FROM video_features")
rows = cur.fetchall()

def to_segmented_path(p):
    raw = Path(p)
    seg_dir = str(raw).replace("/keyframes/", "/keyframes_segmented/")
    seg = Path(seg_dir).with_name(raw.stem + "_segmented.png")
    return str(seg)

for video_id, peak_path in rows:
    seg_path = to_segmented_path(peak_path)
    cur.execute("""
        UPDATE video_features
        SET peak_frame_segmented_path = ?
        WHERE video_id = ?
    """, (seg_path, video_id))

conn.commit()
conn.close()
