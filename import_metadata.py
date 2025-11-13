"""
导入检查元数据到数据库

目录结构:
videos/
├── XW000007/                          # 患者目录 (patient_id)
│   ├── XW000007_20230227_09-51-45/    # 检查目录 (examination_id)
│   │   ├── *.MP4
│   │   └── metadata.json
│   └── ...
└── XW000395/
    └── XW000395_20241230_09-23-39/

职责:
1. 遍历 videos_root/patient_id/examination_id/ 两层目录
2. 解析 metadata.json（动作名已标准化）
3. 建立 patients / examinations / video_files 记录
4. 不涉及标签（标签由 import_labels.py 导入）

特点:
- 两层目录结构：患者目录 -> 检查目录
- examination_id = 检查目录名（如 XW000007_20230227_09-51-45）
- patient_id = 患者目录名（如 XW000007）
- relative_path = patient_id/examination_id/filename.MP4
"""

import os
import json
import sqlite3
import cv2
from datetime import datetime
from pathlib import Path


def parse_examination_id(dir_name):
    """
    解析检查目录名，支持3种格式:
    1) XW000003_20230222_09-46-57
    2) XW000222_20240325_10-41
    3) XW000001_20250905-13-26-38

    返回:
        {
            'patient_id': 'XW000003',
            'capture_datetime': '2023-02-22 09:46:57'
        }
    """
    parts = dir_name.split('_')

    if len(parts) == 3:
        # 格式1, 2: XW000003_20230222_09-46-57
        patient_id = parts[0]
        date_str = parts[1]  # '20230222'
        time_str = parts[2]  # '09-46-57' or '10-41'

        # 解析日期
        year = date_str[0:4]
        month = date_str[4:6]
        day = date_str[6:8]

        # 解析时间
        time_parts = time_str.split('-')
        hour = time_parts[0]
        minute = time_parts[1]
        second = time_parts[2] if len(time_parts) >= 3 else '00'

        capture_datetime = f"{year}-{month}-{day} {hour}:{minute}:{second}"

    elif len(parts) == 2:
        # 格式3: XW000001_20250905-13-26-38
        patient_id = parts[0]
        rest = parts[1]  # '20250905-13-26-38'

        tokens = rest.split('-')
        date_str = tokens[0]  # '20250905'

        year = date_str[0:4]
        month = date_str[4:6]
        day = date_str[6:8]

        hour = tokens[1] if len(tokens) >= 2 else '00'
        minute = tokens[2] if len(tokens) >= 3 else '00'
        second = tokens[3] if len(tokens) >= 4 else '00'

        capture_datetime = f"{year}-{month}-{day} {hour}:{minute}:{second}"

    else:
        raise ValueError(f"无法解析检查ID: {dir_name}")

    return {
        'patient_id': patient_id,
        'capture_datetime': capture_datetime
    }

def get_video_fps(video_path, default=30.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return default
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return int(round(fps)) or default


def time_str_to_frame(time_str, fps):
    """
    时间字符串转帧号
    '00:00:07:000' (时:分:秒:毫秒) -> 帧号
    """
    if not time_str:
        return None
    try:
        parts = time_str.split(':')
        h, m, s, ms = map(int, parts)
        total_seconds = h * 3600 + m * 60 + s + ms / 1000
        return int(total_seconds * fps)
    except:
        return None


def import_metadata(
    db_path,
    videos_base_path,
    reset_tables=True
):
    """
    导入检查元数据（两层目录结构）

    参数:
        db_path: 数据库路径
        videos_base_path: 视频根目录（例如：/Users/.../videos）
        reset_tables: 是否清空重建相关表

    流程:
    1. (可选)清空相关表
    2. 遍历 videos_base_path/patient_id/examination_id/
    3. 解析 metadata.json
    4. 插入数据库记录
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys = ON;")

    if reset_tables:
        print("⚠️  清空相关表...")
        cursor.execute("DELETE FROM video_files;")
        cursor.execute("DELETE FROM frame_files;")
        cursor.execute("DELETE FROM feature_files;")
        cursor.execute("DELETE FROM examinations;")
        cursor.execute("DELETE FROM patients;")
        conn.commit()
        print("    ✓ 已清空")

    stats = {
        'patients': 0,
        'examinations': 0,
        'videos': 0,
        'errors': []
    }

    # 获取11个核心动作的映射 (action_name_en -> action_id)
    cursor.execute("SELECT action_id, action_name_en FROM action_types")
    action_map = {name_en: aid for aid, name_en in cursor.fetchall()}

    print(f"\n开始扫描视频目录: {videos_base_path}")
    print(f"支持的动作: {list(action_map.keys())}\n")

    # ========== 遍历患者目录（第一层） ==========
    patient_dirs = []
    for item in sorted(os.listdir(videos_base_path)):
        item_path = os.path.join(videos_base_path, item)

        # 只处理目录，且目录名以XW开头
        if os.path.isdir(item_path) and item.startswith('XW'):
            patient_dirs.append((item, item_path))

    print(f"发现 {len(patient_dirs)} 个患者目录\n")

    for patient_id, patient_dir in patient_dirs:
        # 插入患者记录
        cursor.execute(
            "INSERT OR IGNORE INTO patients (patient_id) VALUES (?)",
            (patient_id,)
        )
        if cursor.rowcount > 0:
            stats['patients'] += 1
            print(f"患者: {patient_id}")

        # ========== 遍历检查目录（第二层） ==========
        for exam_dir_name in sorted(os.listdir(patient_dir)):
            exam_dir = os.path.join(patient_dir, exam_dir_name)

            # 只处理目录
            if not os.path.isdir(exam_dir):
                continue

            # examination_id = 检查目录名
            examination_id = exam_dir_name

            print(f"  检查: {examination_id}")

            try:
                # 解析检查ID，提取时间等信息
                exam_info = parse_examination_id(examination_id)

                # 从目录名提取的patient_id应该与父目录一致
                extracted_patient_id = exam_info['patient_id']
                if extracted_patient_id != patient_id:
                    stats['errors'].append(
                        f"{examination_id}: 目录名中的patient_id({extracted_patient_id}) "
                        f"与父目录({patient_id})不一致"
                    )
                    continue

                capture_datetime = exam_info['capture_datetime']

                # 查找JSON文件
                json_files = [f for f in os.listdir(exam_dir) if f.endswith('.json')]

                json_path = None
                if json_files:
                    # 优先使用 metadata.json 或 afa-patient-basic-metadata.json
                    for preferred_name in ['metadata.json', 'afa-patient-basic-metadata.json']:
                        if preferred_name in json_files:
                            json_path = os.path.join(exam_dir, preferred_name)
                            break

                    # 如果没找到，就用第一个JSON
                    if json_path is None:
                        json_path = os.path.join(exam_dir, json_files[0])

                # 插入检查记录
                cursor.execute('''
                    INSERT OR REPLACE INTO examinations (
                        examination_id,
                        patient_id,
                        capture_datetime,
                        json_file_path,
                        video_root_dir,
                        has_videos,
                        import_version
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                ''', (
                    examination_id,
                    patient_id,
                    capture_datetime,
                    json_path,
                    exam_dir,
                    1 if json_path else 0
                ))
                stats['examinations'] += 1

                # 如果有JSON文件，解析并导入视频信息
                if json_path and os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8-sig') as f:
                        data = json.load(f)

                    video_meta = data.get('VideoMetaInfo', {})
                    video_file_list = video_meta.get('VideoFileList', [])
                    actions = video_meta.get('ActionList', [])

                    # 遍历每个动作
                    for action in actions:
                        action_name = action.get('Action', '')  # 已标准化的动作名

                        # 直接查询action_id（JSON中的名称已标准化）
                        action_id = action_map.get(action_name)
                        if not action_id:
                            # 不是11个核心动作，跳过
                            continue

                        # 获取视频文件路径
                        video_index = action.get('VideoFileIndex')
                        video_path = None
                        relative_path = None

                        if (video_index is not None and
                            isinstance(video_index, int) and
                            0 <= video_index < len(video_file_list)):

                            # 从JSON获取文件名
                            json_rel_path = video_file_list[video_index].get('Path', '')
                            video_filename = os.path.basename(json_rel_path.replace('\\', '/'))

                            # 构建实际绝对路径
                            video_path = os.path.join(exam_dir, video_filename)

                            # 构建相对路径：patient_id/examination_id/filename
                            relative_path = f"{patient_id}/{examination_id}/{video_filename}"

                        # 时间信息
                        start_time_str = (action.get('StartFrameLocation') or [''])[0]
                        end_time_str = (action.get('EndFrameLocation') or [''])[0]

                        fps = get_video_fps(video_path)
                        start_frame = time_str_to_frame(start_time_str, fps)
                        end_frame = time_str_to_frame(end_time_str, fps)

                        duration = None
                        if start_frame is not None and end_frame is not None:
                            duration = end_frame - start_frame

                        # 检查文件是否存在
                        file_exists = 0
                        file_size = None
                        if video_path and os.path.exists(video_path):
                            file_exists = 1
                            file_size = os.path.getsize(video_path)

                        # 插入视频记录
                        cursor.execute('''
                            INSERT INTO video_files (
                                examination_id,
                                action_id,
                                video_file_path,
                                relative_path,
                                video_file_index,
                                start_time_str,
                                end_time_str,
                                start_frame,
                                end_frame,
                                duration_frames,
                                fps,
                                file_exists,
                                file_size_bytes
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            examination_id,
                            action_id,
                            video_path,
                            relative_path,
                            video_index,
                            start_time_str,
                            end_time_str,
                            start_frame,
                            end_frame,
                            duration,
                            fps,
                            file_exists,
                            file_size
                        ))
                        stats['videos'] += 1

            except Exception as e:
                error_msg = f"{examination_id}: {str(e)}"
                stats['errors'].append(error_msg)
                print(f"      ⚠️  错误: {error_msg}")

    conn.commit()

    # 记录导入日志
    import json as json_module
    error_json = json_module.dumps(stats['errors'], ensure_ascii=False)

    cursor.execute('''
        INSERT INTO import_logs (
            import_type,
            import_source,
            records_processed,
            records_succeeded,
            records_failed,
            error_messages
        )
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        'metadata',
        videos_base_path,
        stats['examinations'],
        stats['examinations'] - len(stats['errors']),
        len(stats['errors']),
        error_json
    ))

    conn.commit()
    conn.close()

    # 打印汇总
    print("\n" + "="*60)
    print("✅ 元数据导入完成!")
    print("="*60)
    print(f"   患者数:   {stats['patients']}")
    print(f"   检查数:   {stats['examinations']}")
    print(f"   视频数:   {stats['videos']}")

    if stats['errors']:
        print(f"\n⚠️  错误数: {len(stats['errors'])}")
        print("   (前20条错误):")
        for err in stats['errors'][:20]:
            print(f"     - {err}")

    print("="*60 + "\n")


if __name__ == '__main__':
    import_metadata(
        db_path='facialPalsy.db',
        videos_base_path='/Users/cuijinglei/Documents/facialPalsy/videos',
        reset_tables=True
    )