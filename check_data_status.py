"""
数据完整性检查脚本
==================

检查所有特征提取阶段是否完成,为特征融合做准备
"""

import sqlite3
import sys
from pathlib import Path


def check_database_status(db_path='facialPalsy.db'):
    """检查数据库完整性"""

    if not Path(db_path).exists():
        print(f"❌ 数据库文件不存在: {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n" + "=" * 80)
    print("H-GFA Net 数据完整性检查")
    print("=" * 80)

    # 1. 基础数据统计
    print("\n【1】基础数据统计")
    print("-" * 80)

    cursor.execute("SELECT COUNT(*) FROM patients")
    n_patients = cursor.fetchone()[0]
    print(f"患者数量: {n_patients}")

    cursor.execute("SELECT COUNT(*) FROM examinations")
    n_exams = cursor.fetchone()[0]
    print(f"检查数量: {n_exams}")

    cursor.execute("SELECT COUNT(*) FROM video_files WHERE file_exists = 1")
    n_videos = cursor.fetchone()[0]
    print(f"视频文件: {n_videos} (file_exists=1)")

    # 2. 标注数据统计
    print("\n【2】标注数据统计")
    print("-" * 80)

    cursor.execute("SELECT COUNT(*) FROM examination_labels")
    n_exam_labels = cursor.fetchone()[0]
    print(f"检查级标注: {n_exam_labels}")

    cursor.execute("SELECT COUNT(*) FROM action_labels")
    n_action_labels = cursor.fetchone()[0]
    print(f"动作级标注: {n_action_labels}")

    # 3. 特征提取完成情况
    print("\n【3】特征提取完成情况")
    print("-" * 80)

    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN static_features IS NOT NULL THEN 1 ELSE 0 END) as has_static,
            SUM(CASE WHEN dynamic_features IS NOT NULL THEN 1 ELSE 0 END) as has_dynamic,
            SUM(CASE WHEN visual_features IS NOT NULL THEN 1 ELSE 0 END) as has_visual,
            SUM(CASE WHEN wrinkle_features IS NOT NULL THEN 1 ELSE 0 END) as has_wrinkle,
            SUM(CASE WHEN motion_features IS NOT NULL THEN 1 ELSE 0 END) as has_motion
        FROM video_features
    """)

    row = cursor.fetchone()
    total = row[0]

    if total == 0:
        print("❌ video_features表为空,请先运行video_pipeline.py")
        conn.close()
        return False

    features = {
        'static_features': row[1],
        'dynamic_features': row[2],
        'visual_features': row[3],
        'wrinkle_features': row[4],
        'motion_features': row[5]
    }

    all_complete = True
    for feat_name, count in features.items():
        percentage = (count / total) * 100
        status = "✅" if count == total else "⚠️"
        print(f"{status} {feat_name:<20} {count:>4}/{total:<4} ({percentage:>5.1f}%)")
        if count < total:
            all_complete = False

    # 4. 检查缺失的特征
    if not all_complete:
        print("\n【4】缺失特征详情")
        print("-" * 80)

        cursor.execute("""
            SELECT 
                vf.video_id,
                at.action_name_en,
                CASE WHEN feat.static_features IS NULL THEN 'Y' ELSE '' END as miss_static,
                CASE WHEN feat.dynamic_features IS NULL THEN 'Y' ELSE '' END as miss_dynamic,
                CASE WHEN feat.visual_features IS NULL THEN 'Y' ELSE '' END as miss_visual,
                CASE WHEN feat.wrinkle_features IS NULL THEN 'Y' ELSE '' END as miss_wrinkle,
                CASE WHEN feat.motion_features IS NULL THEN 'Y' ELSE '' END as miss_motion
            FROM video_files vf
            INNER JOIN action_types at ON vf.action_id = at.action_id
            LEFT JOIN video_features feat ON vf.video_id = feat.video_id
            WHERE vf.file_exists = 1
              AND (feat.static_features IS NULL 
                OR feat.dynamic_features IS NULL 
                OR feat.visual_features IS NULL 
                OR feat.wrinkle_features IS NULL 
                OR feat.motion_features IS NULL)
            ORDER BY vf.video_id
            LIMIT 20
        """)

        rows = cursor.fetchall()
        if rows:
            print(f"\n前20条缺失记录:")
            print(
                f"{'video_id':<10} {'action':<20} {'static':<8} {'dynamic':<8} {'visual':<8} {'wrinkle':<8} {'motion':<8}")
            print("-" * 80)
            for r in rows:
                print(f"{r[0]:<10} {r[1]:<20} {r[2]:<8} {r[3]:<8} {r[4]:<8} {r[5]:<8} {r[6]:<8}")

            if len(rows) == 20:
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM video_files vf
                    LEFT JOIN video_features feat ON vf.video_id = feat.video_id
                    WHERE vf.file_exists = 1
                      AND (feat.static_features IS NULL 
                        OR feat.dynamic_features IS NULL 
                        OR feat.visual_features IS NULL 
                        OR feat.wrinkle_features IS NULL 
                        OR feat.motion_features IS NULL)
                """)
                total_missing = cursor.fetchone()[0]
                print(f"\n... 共有 {total_missing} 条记录缺失特征")

    # 5. 融合特征状态
    print("\n【5】融合特征状态")
    print("-" * 80)

    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN geo_refined_features IS NOT NULL THEN 1 ELSE 0 END) as has_geo,
            SUM(CASE WHEN visual_guided_features IS NOT NULL THEN 1 ELSE 0 END) as has_vg,
            SUM(CASE WHEN fused_action_features IS NOT NULL THEN 1 ELSE 0 END) as has_fused
        FROM video_features
    """)

    row = cursor.fetchone()
    fusion_features = {
        'geo_refined_features': row[1],
        'visual_guided_features': row[2],
        'fused_action_features': row[3]
    }

    for feat_name, count in fusion_features.items():
        percentage = (count / total) * 100
        status = "✅" if count == total else "⬜"
        print(f"{status} {feat_name:<25} {count:>4}/{total:<4} ({percentage:>5.1f}%)")

    # 6. 维度验证
    print("\n【6】特征维度验证")
    print("-" * 80)

    cursor.execute("""
        SELECT 
            static_dim,
            dynamic_dim,
            visual_dim,
            wrinkle_dim,
            motion_dim
        FROM video_features
        WHERE static_features IS NOT NULL
        LIMIT 1
    """)

    dims = cursor.fetchone()
    if dims:
        print(f"static_dim:  {dims[0]} (动作相关: 5-11)")
        print(f"dynamic_dim: {dims[1]} (动作相关: 0-8)")
        print(f"visual_dim:  {dims[2]} (预期: 1280)")
        print(f"wrinkle_dim: {dims[3]} (预期: 10)")
        print(f"motion_dim:  {dims[4]} (预期: 12)")

        # 检查visual维度
        if dims[2] != 1280:
            print(f"⚠️ visual_dim异常,预期1280,实际{dims[2]}")

    # 7. 检查峰值帧图像
    print("\n【7】峰值帧图像检查")
    print("-" * 80)

    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN peak_frame_path IS NOT NULL AND peak_frame_path != '' THEN 1 ELSE 0 END) as has_path
        FROM video_features
    """)

    row = cursor.fetchone()
    total_vf = row[0]
    has_path = row[1]

    if has_path > 0:
        # 检查文件是否存在
        cursor.execute("""
            SELECT peak_frame_path
            FROM video_features
            WHERE peak_frame_path IS NOT NULL AND peak_frame_path != ''
            LIMIT 1
        """)
        sample_path = cursor.fetchone()[0]

        if Path(sample_path).exists():
            print(f"✅ 峰值帧路径: {has_path}/{total_vf}")
            print(f"   示例路径: {sample_path}")
        else:
            print(f"⚠️ 峰值帧路径存在但文件不存在")
            print(f"   示例路径: {sample_path}")
    else:
        print(f"⬜ 峰值帧未保存")

    # 8. 总结与建议
    print("\n【8】总结与建议")
    print("=" * 80)

    if all_complete:
        print("✅ 所有原始特征提取已完成!")

        if fusion_features['fused_action_features'] == total:
            print("✅ 特征融合已完成,可以开始训练!")
            print("\n建议执行:")
            print("  1. python train_model.py config.yaml")
        elif fusion_features['geo_refined_features'] == total:
            print("⬜ 几何特征融合已完成,继续执行:")
            print("  1. python stage2_gqca.py facialPalsy.db")
            print("  2. python stage3_mfa.py facialPalsy.db")
        else:
            print("⬜ 特征融合未开始,按顺序执行:")
            print("  1. python stage1_cdcaf.py facialPalsy.db")
            print("  2. python stage2_gqca.py facialPalsy.db")
            print("  3. python stage3_mfa.py facialPalsy.db")
    else:
        print("⚠️ 部分特征提取未完成,需要补充:")

        if features['static_features'] < total or features['dynamic_features'] < total:
            print("  1. python video_pipeline.py  # 补充几何特征")

        if features['visual_features'] < total:
            print("  2. python visual_feature_extractor.py facialPalsy.db  # 补充视觉特征")

        if features['wrinkle_features'] < total:
            print("  3. python wrinkle_feature.py facialPalsy.db  # 补充皱纹特征")

        if features['motion_features'] < total:
            print("  4. python motion_feature.py facialPalsy.db  # 补充运动特征")

        print("\n补充完成后再运行特征融合")

    conn.close()

    print("\n" + "=" * 80)

    return all_complete


if __name__ == '__main__':
    db_path = 'facialPalsy.db'
    check_database_status(db_path)