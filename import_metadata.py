"""
导入检查元数据到数据库 - 智能增量更新版

新增功能:
1. 增量更新模式: 只更新changed的video_files记录
2. 保护video_features: 不删除已处理的特征数据
3. 灵活的更新策略:
   - update_mode='incremental': 只更新时间改变的记录
   - update_mode='full_reset': 完全清空重建(危险!)
   - update_mode='smart': 智能模式(推荐)

使用场景:
- JSON中某些动作的时间段调整了 -> 运行后自动更新
- 新增了examination -> 自动添加
- 不影响已提取的video_features
"""

import os
import json
import sqlite3
import cv2
from datetime import datetime
from pathlib import Path


def parse_examination_id(dir_name):
    """解析检查目录名"""
    parts = dir_name.split('_')

    if len(parts) == 3:
        patient_id = parts[0]
        date_str = parts[1]
        time_str = parts[2]

        year = date_str[0:4]
        month = date_str[4:6]
        day = date_str[6:8]

        time_parts = time_str.split('-')
        hour = time_parts[0]
        minute = time_parts[1]
        second = time_parts[2] if len(time_parts) >= 3 else '00'

        capture_datetime = f"{year}-{month}-{day} {hour}:{minute}:{second}"

    elif len(parts) == 2:
        patient_id = parts[0]
        rest = parts[1]

        tokens = rest.split('-')
        date_str = tokens[0]

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
    """获取视频帧率"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return default
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return int(round(fps)) or default


def time_str_to_frame(time_str, fps):
    """时间字符串转帧号"""
    if not time_str:
        return None
    try:
        parts = time_str.split(':')
        h, m, s, ms = map(int, parts)
        total_seconds = h * 3600 + m * 60 + s + ms / 1000
        return int(total_seconds * fps)
    except:
        return None


def check_video_changed(cursor, examination_id, action_id, new_start_frame, new_end_frame):
    """
    检查video_files记录是否需要更新

    Returns:
        'not_exist': 记录不存在,需要插入
        'changed': 时间段改变,需要更新
        'unchanged': 无变化,跳过
    """
    cursor.execute("""
        SELECT start_frame, end_frame
        FROM video_files
        WHERE examination_id = ? AND action_id = ?
    """, (examination_id, action_id))

    row = cursor.fetchone()

    if not row:
        return 'not_exist'

    old_start, old_end = row

    if old_start != new_start_frame or old_end != new_end_frame:
        return 'changed'

    return 'unchanged'


def import_metadata(
    db_path,
    videos_base_path,
    update_mode='smart'
):
    """
    导入检查元数据 - 智能增量更新版

    参数:
        db_path: 数据库路径
        videos_base_path: 视频根目录
        update_mode: 更新模式
            - 'incremental': 增量更新(推荐,保留video_features)
            - 'smart': 智能模式(检测变化后更新)
            - 'full_reset': 完全清空重建(危险!会丢失video_features)

    智能更新逻辑:
    1. patients/examinations: INSERT OR REPLACE (自动合并)
    2. video_files:
       - 检测时间段是否改变
       - 改变 -> UPDATE (同时删除对应的video_features)
       - 未改变 -> 跳过
       - 不存在 -> INSERT
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    stats = {
        'patients_new': 0,
        'examinations_new': 0,
        'videos_inserted': 0,
        'videos_updated': 0,
        'videos_unchanged': 0,
        'features_deleted': 0,  # 因更新而删除的features
        'errors': []
    }

    # ⚠️ 危险模式警告
    if update_mode == 'full_reset':
        print("="*60)
        print("⚠️⚠️⚠️  警告: 完全清空模式 ⚠️⚠️⚠️")
        print("这将删除所有数据,包括已处理的video_features!")
        print("="*60)

        response = input("确定要继续吗? (yes/no): ").strip().lower()
        if response != 'yes':
            print("已取消操作")
            return

        print("⚠️  清空所有表...")
        cursor.execute("DELETE FROM video_features;")  # 先删子表
        cursor.execute("DELETE FROM video_files;")
        cursor.execute("DELETE FROM examinations;")
        cursor.execute("DELETE FROM patients;")
        conn.commit()
        print("    ✓ 已清空所有表")

    elif update_mode in ['incremental', 'smart']:
        print("="*60)
        print(f"✅ {update_mode.upper()} 更新模式")
        print("只更新变化的记录,保护已处理的video_features")
        print("="*60)

    # 获取动作映射
    cursor.execute("SELECT action_id, action_name_en FROM action_types")
    action_map = {name_en: aid for aid, name_en in cursor.fetchall()}

    print(f"\n开始扫描视频目录: {videos_base_path}")
    print(f"支持的动作: {list(action_map.keys())}\n")

    # ========== 遍历患者目录 ==========
    patient_dirs = []
    for item in sorted(os.listdir(videos_base_path)):
        item_path = os.path.join(videos_base_path, item)
        if os.path.isdir(item_path) and item.startswith('XW'):
            patient_dirs.append((item, item_path))

    print(f"发现 {len(patient_dirs)} 个患者目录\n")

    for patient_id, patient_dir in patient_dirs:
        # 插入患者 (INSERT OR IGNORE自动去重)
        cursor.execute(
            "INSERT OR IGNORE INTO patients (patient_id) VALUES (?)",
            (patient_id,)
        )
        if cursor.rowcount > 0:
            stats['patients_new'] += 1
            print(f"患者: {patient_id} [新增]")
        else:
            print(f"患者: {patient_id}")

        # ========== 遍历检查目录 ==========
        for exam_dir_name in sorted(os.listdir(patient_dir)):
            exam_dir = os.path.join(patient_dir, exam_dir_name)

            if not os.path.isdir(exam_dir):
                continue

            examination_id = exam_dir_name
            print(f"  检查: {examination_id}", end="")

            try:
                exam_info = parse_examination_id(examination_id)
                extracted_patient_id = exam_info['patient_id']

                if extracted_patient_id != patient_id:
                    stats['errors'].append(
                        f"{examination_id}: patient_id不一致"
                    )
                    print(" [跳过:ID不一致]")
                    continue

                capture_datetime = exam_info['capture_datetime']

                # 查找JSON
                json_files = [f for f in os.listdir(exam_dir) if f.endswith('.json')]
                json_path = None

                if json_files:
                    for preferred_name in ['metadata.json', 'afa-patient-basic-metadata.json']:
                        if preferred_name in json_files:
                            json_path = os.path.join(exam_dir, preferred_name)
                            break
                    if json_path is None:
                        json_path = os.path.join(exam_dir, json_files[0])

                # 检查是否是新examination
                cursor.execute(
                    "SELECT 1 FROM examinations WHERE examination_id = ?",
                    (examination_id,)
                )
                is_new_exam = cursor.fetchone() is None

                # 插入/更新examination
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

                if is_new_exam:
                    stats['examinations_new'] += 1
                    print(" [新增]")
                else:
                    print(" [已存在]")

                # 解析JSON并更新video_files
                if json_path and os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8-sig') as f:
                        data = json.load(f)

                    video_meta = data.get('VideoMetaInfo', {})
                    video_file_list = video_meta.get('VideoFileList', [])
                    actions = video_meta.get('ActionList', [])

                    for action in actions:
                        action_name = action.get('Action', '')
                        action_id = action_map.get(action_name)

                        if not action_id:
                            continue

                        # 获取视频路径
                        video_index = action.get('VideoFileIndex')
                        video_path = None
                        relative_path = None

                        if (video_index is not None and
                            isinstance(video_index, int) and
                            0 <= video_index < len(video_file_list)):

                            json_rel_path = video_file_list[video_index].get('Path', '')
                            video_filename = os.path.basename(json_rel_path.replace('\\', '/'))
                            video_path = os.path.join(exam_dir, video_filename)
                            relative_path = f"{patient_id}/{examination_id}/{video_filename}"

                        # 时间信息
                        start_time_str = (action.get('StartFrameLocation') or [''])[0]
                        end_time_str = (action.get('EndFrameLocation') or [''])[0]

                        fps = get_video_fps(video_path) if video_path else 30.0
                        start_frame = time_str_to_frame(start_time_str, fps)
                        end_frame = time_str_to_frame(end_time_str, fps)

                        duration = None
                        if start_frame is not None and end_frame is not None:
                            duration = end_frame - start_frame

                        file_exists = 0
                        file_size = None
                        if video_path and os.path.exists(video_path):
                            file_exists = 1
                            file_size = os.path.getsize(video_path)

                        # 🔧 关键: 检查是否需要更新
                        change_status = check_video_changed(
                            cursor, examination_id, action_id,
                            start_frame, end_frame
                        )

                        if change_status == 'not_exist':
                            # 插入新记录
                            cursor.execute('''
                                INSERT INTO video_files (
                                    examination_id,
                                    action_id,
                                    file_path,
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
                                examination_id, action_id, video_path, relative_path,
                                video_index, start_time_str, end_time_str,
                                start_frame, end_frame, duration, fps,
                                file_exists, file_size
                            ))
                            stats['videos_inserted'] += 1
                            print(f"      {action_name}: 新增")

                        elif change_status == 'changed':
                            # ⚠️ 时间段改变,需要更新
                            # 1. 先删除对应的video_features (因为特征基于旧时间段)
                            cursor.execute("""
                                DELETE FROM video_features
                                WHERE video_id = (
                                    SELECT video_id FROM video_files
                                    WHERE examination_id = ? AND action_id = ?
                                )
                            """, (examination_id, action_id))

                            if cursor.rowcount > 0:
                                stats['features_deleted'] += cursor.rowcount
                                print(f"      {action_name}: 删除旧特征 (时间段已变)")

                            # 2. 更新video_files
                            cursor.execute("""
                                UPDATE video_files
                                SET file_path = ?,
                                    relative_path = ?,
                                    video_file_index = ?,
                                    start_time_str = ?,
                                    end_time_str = ?,
                                    start_frame = ?,
                                    end_frame = ?,
                                    duration_frames = ?,
                                    fps = ?,
                                    file_exists = ?,
                                    file_size_bytes = ?
                                WHERE examination_id = ? AND action_id = ?
                            """, (
                                video_path, relative_path, video_index,
                                start_time_str, end_time_str,
                                start_frame, end_frame, duration, fps,
                                file_exists, file_size,
                                examination_id, action_id
                            ))
                            stats['videos_updated'] += 1
                            print(f"      {action_name}: 更新 (时间: {start_frame}->{end_frame})")

                        else:  # unchanged
                            stats['videos_unchanged'] += 1
                            # 不打印,避免刷屏

            except Exception as e:
                error_msg = f"{examination_id}: {str(e)}"
                stats['errors'].append(error_msg)
                print(f" [错误: {str(e)}]")

    conn.commit()

    # 记录导入日志
    import json as json_module
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
        f'metadata_{update_mode}',
        videos_base_path,
        stats['examinations_new'],
        stats['videos_inserted'] + stats['videos_updated'],
        len(stats['errors']),
        json_module.dumps(stats['errors'], ensure_ascii=False)
    ))

    conn.commit()
    conn.close()

    # 打印汇总
    print("\n" + "="*60)
    print("✅ 元数据导入完成!")
    print("="*60)
    print(f"   新增患者:         {stats['patients_new']}")
    print(f"   新增检查:         {stats['examinations_new']}")
    print(f"   新增视频记录:     {stats['videos_inserted']}")
    print(f"   更新视频记录:     {stats['videos_updated']}")
    print(f"   未变化记录:       {stats['videos_unchanged']}")

    if stats['features_deleted'] > 0:
        print(f"\n⚠️  删除过时特征:   {stats['features_deleted']} 条")
        print("   (这些视频的时间段改变了,需要重新处理)")

    if stats['errors']:
        print(f"\n⚠️  错误数: {len(stats['errors'])}")
        for err in stats['errors'][:10]:
            print(f"     - {err}")

    print("="*60 + "\n")

    # 给出下一步建议
    if stats['videos_updated'] > 0 or stats['features_deleted'] > 0:
        print("💡 提示:")
        print("   部分视频的时间段已更新,建议运行 video_pipeline.py 重新提取特征")
        print()


if __name__ == '__main__':
    import_metadata(
        db_path='facialPalsy.db',
        videos_base_path='/Users/cuijinglei/Documents/facialPalsy/videos',
        update_mode='smart'
    )