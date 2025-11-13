"""
导入医生标注数据（从 labels.xlsx）

职责:
1. 读取 labels.xlsx 的 labels Sheet
2. 匹配Excel中的编号到数据库的 examination_id
3. 写入 examination_labels 和 action_labels
4. 自动补充健康人的标签

关键改进:
- 清晰的标签来源标记（doctor/assumed）
- 支持增量更新和全量刷新
- 记录Excel原始信息（行号、列名等）
- 直接使用标准化的动作名称
"""

import pandas as pd
import sqlite3
from typing import Optional
import json

# ============================================================
# Excel配置
# ============================================================

EXCEL_CONFIG = {
    # 全局标签列
    'session_id_col': '编号',
    'has_palsy_col': '面瘫',
    'palsy_side_col': '侧别',
    'hb_grade_col': 'HB分级',
    'sunnybrook_col': 'SB评分',

    # 11个动作的Excel列名 -> 标准化动作名
    'action_columns': {
        '静息': 'NeutralFace',
        '自然眨眼': 'SpontaneousEyeBlink',
        '自主眨眼': 'VoluntaryEyeBlink',
        '轻轻闭眼': 'CloseEyeSoftly',
        '用力闭眼': 'CloseEyeHardly',
        '抬眉': 'RaiseEyebrow',
        '微笑': 'Smile',
        '耸鼻': 'ShrugNose',
        '呲牙': 'ShowTeeth',
        '鼓腮': 'BlowCheek',
        '撅嘴': 'LipPucker'
    }
}


# ============================================================
# 辅助函数
# ============================================================

def is_empty(val):
    """判断单元格是否为空"""
    if val is None or pd.isna(val):
        return True
    s = str(val).strip()
    return s == '' or s == '-' or s.lower() in ['nan', 'none', 'null']


def normalize_has_palsy(raw_val):
    """
    标准化面瘫标识
    1 = 面瘫
    0 = 健康
    """
    if is_empty(raw_val):
        return 0

    s = str(raw_val).strip().lower()
    if s in ['1', 'true', '是', '有', '面瘫', 'yes', 'y']:
        return 1
    if s in ['0', 'false', '否', '无', '健康', 'no', 'n', 'normal']:
        return 0

    try:
        return 1 if int(raw_val) != 0 else 0
    except:
        return 0


def normalize_palsy_side(raw_val, has_palsy):
    """
    标准化患侧
    0 = 无/健康
    1 = 患者左侧（屏幕右侧下垂）
    2 = 患者右侧（屏幕左侧下垂）
    """
    if has_palsy == 0:
        return 0

    if is_empty(raw_val):
        return None

    s = str(raw_val).strip().lower()

    # 左侧
    if s in ['1', '左', '左侧', '患者左侧', 'left', 'screen_right']:
        return 1

    # 右侧
    if s in ['2', '右', '右侧', '患者右侧', 'right', 'screen_left']:
        return 2

    try:
        v = int(raw_val)
        if v in [1, 2]:
            return v
    except:
        pass

    return None


def match_examination_id(cursor, excel_session_id):
    """
    将Excel中的session_id匹配到数据库的examination_id

    策略:
    1. 精确匹配
    2. 前缀模糊匹配（处理有无秒的差异）

    返回:
        examination_id 或 None
    """
    raw = str(excel_session_id).strip()

    # 1. 精确匹配
    cursor.execute(
        "SELECT examination_id FROM examinations WHERE examination_id = ?",
        (raw,)
    )
    row = cursor.fetchone()
    if row:
        return row[0]

    # 2. 前缀模糊匹配
    parts = raw.split('_')

    if len(parts) >= 3:
        # XW000010_20231113_10-47 -> XW000010_20231113_10-47%
        prefix = '_'.join(parts[:3])
        cursor.execute(
            "SELECT examination_id FROM examinations WHERE examination_id LIKE ? LIMIT 1",
            (prefix + '%',)
        )
        row = cursor.fetchone()
        if row:
            return row[0]

    elif len(parts) == 2:
        # XW000001_20250905-13-26 -> XW000001_20250905-13-26%
        cursor.execute(
            "SELECT examination_id FROM examinations WHERE examination_id LIKE ? LIMIT 1",
            (raw + '%',)
        )
        row = cursor.fetchone()
        if row:
            return row[0]

    return None


# ============================================================
# 主导入函数
# ============================================================

def import_labels(
        db_path,
        excel_path,
        sheet_name='labels',
        reset_tables=True
):
    """
    从Excel导入医生标注

    参数:
        db_path: 数据库路径
        excel_path: Excel文件路径
        sheet_name: Sheet名称（默认'labels'）
        reset_tables: 是否清空标签表（True=全量刷新，False=增量更新）
    """
    # 读取Excel
    print(f"读取Excel: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"  共 {len(df)} 行数据\n")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys = ON;")

    if reset_tables:
        print("⚠️  清空标签表...")
        cursor.execute("DELETE FROM action_labels;")
        cursor.execute("DELETE FROM examination_labels;")
        cursor.execute("UPDATE examinations SET has_labels = 0")
        conn.commit()
        print("    ✓ 已清空\n")

    # 获取action映射（action_name_en -> action_id）
    cursor.execute("SELECT action_id, action_name_en FROM action_types")
    action_map = {name_en: aid for aid, name_en in cursor.fetchall()}

    stats = {
        'exam_labels': 0,
        'action_labels': 0,
        'warnings': []
    }

    cfg = EXCEL_CONFIG

    print("开始导入标签...\n")

    # 遍历Excel每一行
    for row_idx, row in df.iterrows():
        excel_row_num = row_idx + 2  # Excel行号（从1开始，表头占1行）

        # ===== 1. 获取并匹配 examination_id =====
        if cfg['session_id_col'] not in row or is_empty(row[cfg['session_id_col']]):
            stats['warnings'].append(f"第{excel_row_num}行: 缺少 {cfg['session_id_col']}")
            continue

        excel_sid = str(row[cfg['session_id_col']]).strip()

        examination_id = match_examination_id(cursor, excel_sid)
        if not examination_id:
            stats['warnings'].append(f"无法匹配到数据库: {excel_sid}")
            continue

        # ===== 2. 解析全局标签 =====
        has_palsy = normalize_has_palsy(row.get(cfg['has_palsy_col']))

        palsy_side = normalize_palsy_side(
            row.get(cfg['palsy_side_col']) if cfg['palsy_side_col'] in row else None,
            has_palsy
        )

        hb_grade = None
        if cfg['hb_grade_col'] in row and not is_empty(row[cfg['hb_grade_col']]):
            try:
                hb_grade = int(row[cfg['hb_grade_col']])
            except:
                pass

        sunnybrook = None
        if cfg['sunnybrook_col'] in row and not is_empty(row[cfg['sunnybrook_col']]):
            try:
                sunnybrook = float(row[cfg['sunnybrook_col']])
            except:
                pass

        # ===== 3. 健康人自动补充 =====
        label_source = 'doctor'
        if has_palsy == 0:
            if hb_grade is None:
                hb_grade = 1
            if sunnybrook is None:
                sunnybrook = 100.0
            if palsy_side is None:
                palsy_side = 0

        # ===== 4. 插入检查级标签 =====
        cursor.execute('''
            INSERT OR REPLACE INTO examination_labels (
                examination_id,
                has_palsy,
                palsy_side,
                hb_grade,
                sunnybrook_score,
                label_source,
                excel_row_number,
                excel_session_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            examination_id,
            has_palsy,
            palsy_side,
            hb_grade,
            sunnybrook,
            label_source,
            excel_row_num,
            excel_sid
        ))

        # 更新 examinations 表的 has_labels 标识
        cursor.execute(
            "UPDATE examinations SET has_labels = 1 WHERE examination_id = ?",
            (examination_id,)
        )

        stats['exam_labels'] += 1

        # ===== 5. 插入动作级标签 =====
        for cn_col, en_name in cfg['action_columns'].items():
            if cn_col not in df.columns:
                continue

            raw_score = row.get(cn_col)

            if is_empty(raw_score):
                # 医生没有标注该动作
                if has_palsy == 0:
                    # 健康人：自动补1分
                    severity_score = 1
                    action_source = 'assumed'
                else:
                    # 面瘫患者：没标注就跳过
                    continue
            else:
                # 医生有标注
                try:
                    severity_score = int(raw_score)
                    action_source = 'doctor'
                except:
                    stats['warnings'].append(
                        f"非法评分: Excel行{excel_row_num}, {cn_col}={raw_score}"
                    )
                    continue

            # 获取 action_id
            action_id = action_map.get(en_name)
            if not action_id:
                stats['warnings'].append(f"未知动作: {en_name}")
                continue

            # 插入动作标签
            cursor.execute('''
                INSERT OR REPLACE INTO action_labels (
                    examination_id,
                    action_id,
                    severity_score,
                    label_source,
                    excel_column_name
                )
                VALUES (?, ?, ?, ?, ?)
            ''', (
                examination_id,
                action_id,
                severity_score,
                action_source,
                cn_col
            ))

            stats['action_labels'] += 1

    conn.commit()

    # ===== 6. 记录导入日志 =====
    error_json = json.dumps(stats['warnings'], ensure_ascii=False)

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
        'labels',
        excel_path,
        len(df),
        stats['exam_labels'],
        len(stats['warnings']),
        error_json
    ))

    conn.commit()
    conn.close()

    # ===== 7. 打印汇总 =====
    print("\n" + "=" * 60)
    print("✅ 标签导入完成!")
    print("=" * 60)
    print(f"   检查级标签: {stats['exam_labels']}")
    print(f"   动作级标签: {stats['action_labels']}")

    if stats['warnings']:
        print(f"\n⚠️  警告数: {len(stats['warnings'])}")
        print("   (前20条警告):")
        for w in stats['warnings'][:20]:
            print(f"     - {w}")

    print("=" * 60 + "\n")


if __name__ == '__main__':
    import_labels(
        db_path='facialPalsy.db',
        excel_path='/Users/cuijinglei/Documents/facialPalsy/notes/HospitalLabels/labels.xlsx',
        sheet_name='labels',
        reset_tables=True
    )