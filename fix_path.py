# fix_peak_paths.py
import sqlite3
import sys
from pathlib import Path

OLD = "/Users/cuijinglei/Documents/facialPalsy/pipeline/"
NEW = "/Users/cuijinglei/Documents/facialPalsy/HGFA/"

db_path = 'facialPalsy.db'

def main():

    conn = sqlite3.connect(db_path, timeout=60)
    try:
        cur = conn.cursor()

        # 1) 统计受影响行
        cur.execute("""
            SELECT COUNT(*)
            FROM video_features
            WHERE (peak_frame_path LIKE ?)
               OR (peak_frame_segmented_path LIKE ?)
        """, (OLD + "%", OLD + "%"))
        affected = cur.fetchone()[0]
        print(f"[INFO] 将更新的行数: {affected}")

        # 2) 预览前 10 条
        cur.execute("""
            SELECT video_id, peak_frame_path, peak_frame_segmented_path
            FROM video_features
            WHERE (peak_frame_path LIKE ?)
               OR (peak_frame_segmented_path LIKE ?)
            LIMIT 10
        """, (OLD + "%", OLD + "%"))
        rows = cur.fetchall()
        if rows:
            print("\n[PREVIEW] 前 10 条(更新前)：")
            for vid, p1, p2 in rows:
                print(f"- video_id={vid}")
                print(f"  peak_frame_path:           {p1}")
                print(f"  peak_frame_segmented_path: {p2}")

        # 3) 执行更新：只替换“以 OLD 开头”的路径
        cur.execute("""
            UPDATE video_features
            SET peak_frame_path =
                CASE
                    WHEN peak_frame_path LIKE ? THEN REPLACE(peak_frame_path, ?, ?)
                    ELSE peak_frame_path
                END,
                peak_frame_segmented_path =
                CASE
                    WHEN peak_frame_segmented_path LIKE ? THEN REPLACE(peak_frame_segmented_path, ?, ?)
                    ELSE peak_frame_segmented_path
                END,
                updated_at = CURRENT_TIMESTAMP
            WHERE (peak_frame_path LIKE ?)
               OR (peak_frame_segmented_path LIKE ?)
        """, (
            OLD + "%", OLD, NEW,
            OLD + "%", OLD, NEW,
            OLD + "%", OLD + "%"
        ))

        print(f"\n[INFO] 实际更新行数(rowcount): {cur.rowcount}")
        conn.commit()

        # 4) 再预览更新后的前 10 条
        cur.execute("""
            SELECT video_id, peak_frame_path, peak_frame_segmented_path
            FROM video_features
            WHERE (peak_frame_path LIKE ?)
               OR (peak_frame_segmented_path LIKE ?)
            LIMIT 10
        """, (NEW + "%", NEW + "%"))
        rows2 = cur.fetchall()
        if rows2:
            print("\n[PREVIEW] 前 10 条(更新后)：")
            for vid, p1, p2 in rows2:
                print(f"- video_id={vid}")
                print(f"  peak_frame_path:           {p1}")
                print(f"  peak_frame_segmented_path: {p2}")

        print("\n[DONE] 路径替换完成。")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
