"""
Stage 5: 推理并写回数据库
--------------------------------

功能概览:
1. 加载 Stage4 训练得到的:
    - models/action_severity_mlp.pth
    - models/exam_multitask_mlp.pth
    - models/feature_scaler.npz

2. 从 SQLite 中读取:
    - video_features.fused_action_features (512维)
    - video_files(examination_id, action_id, video_id)
    - examination_labels (如果有，用来对比/评估；此处主要是生成预测)

3. 生成两个级别的预测，并写入新表:
    - 动作级预测: action_predictions
        * video_id
        * examination_id
        * action_id
        * severity_pred

    - 检查级预测: examination_predictions
        * examination_id
        * has_palsy_prob
        * palsy_side_pred
        * hb_grade_pred
        * sunnybrook_pred

使用:
    python stage5_inference_to_db.py [facialPalsy.db]

注意:
    - 默认假设 fused_action_features 维度为 512
    - 如果还没建预测表，会自动 CREATE TABLE IF NOT EXISTS
"""

import os
import sys
import sqlite3
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


# =============================
# 配置
# =============================

DB_PATH_DEFAULT = "facialPalsy.db"
MODEL_DIR = "models"

ACTION_MODEL_PATH = os.path.join(MODEL_DIR, "action_severity_mlp.pth")
EXAM_MODEL_PATH = os.path.join(MODEL_DIR, "exam_multitask_mlp.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.npz")

BATCH_SIZE = 128
RANDOM_SEED = 42


# =============================
# 通用工具
# =============================

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# =============================
# 与 Stage4 保持一致的模型定义
# =============================

class ActionSeverityMLP(nn.Module):
    """
    动作严重程度回归模型
    输入:  fused_action_features (512 维)
    输出:  severity_score (实数)
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class ExamMultiTaskMLP(nn.Module):
    """
    检查级多任务模型:
    - has_palsy      : 是否面瘫 (0/1, 输出 logits)
    - palsy_side     : 患侧 (0=无, 1=左, 2=右, 输出 logits)
    - hb_grade       : HB 分级 (0~5 对应 1~6, 输出 logits)
    - sunnybrook     : Sunnybrook (0~1 归一化, 输出实数)
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.head_has_palsy = nn.Linear(hidden_dim, 1)
        self.head_palsy_side = nn.Linear(hidden_dim, 3)
        self.head_hb_grade = nn.Linear(hidden_dim, 6)
        self.head_sunny = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.backbone(x)
        out_hp = self.head_has_palsy(h)
        out_side = self.head_palsy_side(h)
        out_hb = self.head_hb_grade(h)
        out_sunny = self.head_sunny(h)
        return out_hp, out_side, out_hb, out_sunny


# =============================
# DB 初始化预测表
# =============================

def init_prediction_tables(conn: sqlite3.Connection):
    cur = conn.cursor()

    # 动作级预测表
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS action_predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            examination_id INTEGER NOT NULL,
            action_id INTEGER NOT NULL,
            video_id INTEGER NOT NULL UNIQUE,
            severity_pred REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES video_files(video_id)
        )
        """
    )

    # 检查级预测表
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS examination_predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            examination_id INTEGER NOT NULL UNIQUE,
            has_palsy_prob REAL,        -- AI 预测为“有面瘫”的概率
            palsy_side_pred INTEGER,    -- 0=无, 1=左, 2=右
            hb_grade_pred INTEGER,      -- 1~6
            sunnybrook_pred REAL,       -- 0~100
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (examination_id) REFERENCES examinations(examination_id)
        )
        """
    )

    conn.commit()
    print("[DB] action_predictions / examination_predictions 表检查完成（若不存在则已创建）")


# =============================
# 读取 scaler 和模型
# =============================

def load_scaler() -> Dict[str, np.ndarray]:
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"未找到特征标准化文件: {SCALER_PATH}")
    data = np.load(SCALER_PATH)
    mean = data["mean"]  # (1, D)
    std = data["std"]    # (1, D)
    print(f"[Scaler] 已加载 mean/std, 维度 = {mean.shape[1]}")
    return {"mean": mean, "std": std}


def load_models(device: str):
    # 动作级
    if not os.path.exists(ACTION_MODEL_PATH):
        raise FileNotFoundError(f"未找到动作级模型权重: {ACTION_MODEL_PATH}")
    action_state = torch.load(ACTION_MODEL_PATH, map_location=device)
    input_dim = list(action_state.values())[0].shape[1]  # 粗略推测输入维度
    action_model = ActionSeverityMLP(input_dim=input_dim)
    action_model.load_state_dict(action_state)
    action_model.to(device)
    action_model.eval()
    print(f"[Model] 已加载 ActionSeverityMLP, 输入维度 = {input_dim}")

    # 检查级
    if not os.path.exists(EXAM_MODEL_PATH):
        raise FileNotFoundError(f"未找到检查级模型权重: {EXAM_MODEL_PATH}")
    exam_state = torch.load(EXAM_MODEL_PATH, map_location=device)
    input_dim_exam = list(exam_state.values())[0].shape[1]
    exam_model = ExamMultiTaskMLP(input_dim=input_dim_exam)
    exam_model.load_state_dict(exam_state)
    exam_model.to(device)
    exam_model.eval()
    print(f"[Model] 已加载 ExamMultiTaskMLP, 输入维度 = {input_dim_exam}")

    return action_model, exam_model


# =============================
# 动作级推理并写入 action_predictions
# =============================

def infer_actions_and_write(conn: sqlite3.Connection,
                            action_model: nn.Module,
                            scaler: Dict[str, np.ndarray],
                            device: str):
    cur = conn.cursor()

    # 只选取尚未有预测结果的视频
    cur.execute(
        """
        SELECT
            vf.fused_action_features,
            v.video_id,
            v.examination_id,
            v.action_id
        FROM video_features vf
        JOIN video_files v
          ON v.video_id = vf.video_id
        LEFT JOIN action_predictions ap
               ON ap.action_id = v.action_id
        WHERE vf.fused_action_features IS NOT NULL
          AND ap.action_id IS NULL
        """
    )
    rows = cur.fetchall()

    if not rows:
        print("[ActionInference] 没有需要推理的动作样本 (可能都已经有预测记录了)")
        return

    print(f"[ActionInference] 需要推理的动作样本数 = {len(rows)}")

    mean = scaler["mean"]
    std = scaler["std"]

    # 分批推理
    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start : start + BATCH_SIZE]

        feats = []
        video_ids = []
        exam_ids = []
        action_ids = []

        for fused_blob, vid, exam_id, action_id in batch_rows:
            feat = np.frombuffer(fused_blob, dtype=np.float32)
            feats.append(feat)
            video_ids.append(vid)
            exam_ids.append(exam_id)
            action_ids.append(action_id)

        X = np.vstack(feats)  # (B, D)
        X_norm = (X - mean) / std

        xb = torch.from_numpy(X_norm.astype(np.float32)).to(device)

        with torch.no_grad():
            pred = action_model(xb)  # (B, 1)
            pred_np = pred.cpu().numpy().reshape(-1)

        # 写入 action_predictions
        for vid, eid, aid, sp in zip(video_ids, exam_ids, action_ids, pred_np):
            cur.execute(
                """
                INSERT INTO action_predictions (
                    examination_id,
                    action_id,
                    pred_severity_score,
                    model_name,
                    model_version
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (eid, aid, float(sp), 'test', '0.1'),
            )

        conn.commit()
        print(f"[ActionInference] 已推理并写入 {start + len(batch_rows)}/{len(rows)} 条")

    print("[ActionInference] 全部完成 ✅")


# =============================
# 检查级推理并写入 examination_predictions
# =============================

def infer_exams_and_write(conn: sqlite3.Connection,
                          exam_model: nn.Module,
                          scaler: Dict[str, np.ndarray],
                          device: str):
    cur = conn.cursor()

    # 先取出所有 examination_id 对应的 fused_action_features，用于求平均
    cur.execute(
        """
        SELECT
            v.examination_id,
            vf.fused_action_features
        FROM video_features vf
        JOIN video_files v
          ON v.video_id = vf.video_id
        WHERE vf.fused_action_features IS NOT NULL
        """
    )
    rows = cur.fetchall()

    if not rows:
        print("[ExamInference] 没有任何 fused_action_features 可用于检查级推理")
        return

    # exam_id -> list of features
    exam_feat_dict: Dict[int, List[np.ndarray]] = {}
    for exam_id, fused_blob in rows:
        feat = np.frombuffer(fused_blob, dtype=np.float32)
        exam_feat_dict.setdefault(exam_id, []).append(feat)

    # 只对还没有预测记录的 examination 做推理
    cur.execute("SELECT examination_id FROM examination_predictions")
    existed = {row[0] for row in cur.fetchall()}

    exam_ids = []
    feat_avg_list = []

    for exam_id, feats in exam_feat_dict.items():
        if exam_id in existed:
            continue
        feats_arr = np.vstack(feats)    # (n_actions, D)
        feat_mean = feats_arr.mean(axis=0)
        exam_ids.append(exam_id)
        feat_avg_list.append(feat_mean)

    if not exam_ids:
        print("[ExamInference] 所有 examination 都已有预测记录, 无需重复推理")
        return

    X = np.vstack(feat_avg_list)            # (N, D)
    mean = scaler["mean"]
    std = scaler["std"]
    X_norm = (X - mean) / std

    xb_all = torch.from_numpy(X_norm.astype(np.float32)).to(device)

    exam_model.eval()
    results = []

    with torch.no_grad():
        # 也可以分 batch 推理, 这里 N 一般不大, 直接一次性也可以
        out_hp, out_side, out_hb, out_sunny = exam_model(xb_all)

        hp_logit = out_hp.squeeze(1)            # (N,)
        side_logits = out_side                  # (N, 3)
        hb_logits = out_hb                      # (N, 6)
        sunny_norm = out_sunny.squeeze(1)       # (N,)

        hp_prob = torch.sigmoid(hp_logit)       # (N,)
        side_pred = torch.argmax(side_logits, dim=1)   # (N,)
        hb_pred = torch.argmax(hb_logits, dim=1)       # (N,)
        sunny_pred = sunny_norm * 100.0         # 还原到 0~100

        hp_prob_np = hp_prob.cpu().numpy()
        side_pred_np = side_pred.cpu().numpy()
        hb_pred_np = hb_pred.cpu().numpy()
        sunny_pred_np = sunny_pred.cpu().numpy()

        for eid, p_hp, ps, hb, sb in zip(
            exam_ids, hp_prob_np, side_pred_np, hb_pred_np, sunny_pred_np
        ):
            # hb_pred 存到表里时恢复到 1~6
            hb_grade_pred = int(hb) + 1
            results.append((eid, float(p_hp), int(ps), hb_grade_pred, float(sb)))

    # 写入 examination_predictions
    for eid, hp_p, ps, hb_g, sb in results:
        cur.execute(
            """
            INSERT INTO examination_predictions (
                examination_id,
                has_palsy_prob,
                palsy_side_pred,
                hb_grade_pred,
                sunnybrook_pred
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (eid, hp_p, ps, hb_g, sb),
        )

    conn.commit()
    print(f"[ExamInference] 已为 {len(results)} 个 examination 写入预测结果 ✅")


# =============================
# main
# =============================

def main():
    set_seed(RANDOM_SEED)

    db_path = sys.argv[1] if len(sys.argv) > 1 else DB_PATH_DEFAULT
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")

    device = choose_device()
    print(f"[Device] 使用设备: {device}")

    # 连接数据库 & 初始化预测表
    conn = sqlite3.connect(db_path)
    init_prediction_tables(conn)

    # 加载 scaler + 模型
    scaler = load_scaler()
    action_model, exam_model = load_models(device)

    # 动作级推理
    print("\n====== 动作级推理 (Action-Level) ======")
    infer_actions_and_write(conn, action_model, scaler, device)

    # 检查级推理
    print("\n====== 检查级推理 (Exam-Level) ======")
    infer_exams_and_write(conn, exam_model, scaler, device)

    conn.close()
    print("\n✅ Stage 5 推理与写库全部完成。")


if __name__ == "__main__":
    main()
