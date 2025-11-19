"""
Stage 4: 模型训练脚本
--------------------

功能概览:
1. 从 facialPalsy.db 中读取:
   - video_features.fused_action_features (动作级多模态特征, 512维)
   - action_labels.severity_score         (动作严重程度)
   - examination_labels.(has_palsy, palsy_side, hb_grade, sunnybrook_score)

2. 构建两个训练集:
   - Action-level: 每个 video_id 一条样本
   - Exam-level:   每个 examination_id 一条样本 (对该检查下所有动作特征做平均)

3. 对特征做标准化 (mean/std), 并保存 scaler

4. 用 PyTorch 训练两个模型:
   - ActionSeverityMLP:   输入 512 → 输出 1 (严重程度回归)
   - ExamMultiTaskMLP:    输入 512 → 输出:
       * has_palsy      (二分类, logits)
       * palsy_side     (三分类 0/1/2, logits)
       * hb_grade       (6分类 1~6 → 0~5, logits)
       * sunnybrook     (回归)

5. 将训练好的模型与 scaler 保存到 models/ 目录

使用方式:
    python stage4_train_models.py [facialPalsy.db]

"""

import os
import sys
import sqlite3
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =============================
# 配置
# =============================

DB_PATH_DEFAULT = "facialPalsy.db"
MODEL_DIR = "models"

ACTION_MODEL_PATH = os.path.join(MODEL_DIR, "action_severity_mlp.pth")
EXAM_MODEL_PATH = os.path.join(MODEL_DIR, "exam_multitask_mlp.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.npz")

BATCH_SIZE = 64
EPOCHS_ACTION = 80
EPOCHS_EXAM = 120
LR_ACTION = 1e-3
LR_EXAM = 1e-3
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.2
RANDOM_SEED = 42


# =============================
# 通用工具
# =============================

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)


def split_train_val(X: np.ndarray, *ys, val_ratio: float = 0.2):
    """简单的打乱 + 划分训练/验证集"""
    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)

    val_size = max(1, int(N * val_ratio))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    def _split(arr):
        return arr[train_idx], arr[val_idx]

    X_train, X_val = _split(X)
    ys_train_val = []
    for y in ys:
        y_train, y_val = _split(y)
        ys_train_val.append((y_train, y_val))

    return X_train, X_val, ys_train_val


# =============================
# 数据集定义
# =============================

class ActionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ExamDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y_has_palsy: np.ndarray,
        y_palsy_side: np.ndarray,
        y_hb_grade: np.ndarray,
        y_sunny: np.ndarray,
    ):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y_has_palsy = torch.from_numpy(y_has_palsy.astype(np.float32)).view(-1, 1)
        self.y_palsy_side = torch.from_numpy(y_palsy_side.astype(np.int64))
        self.y_hb_grade = torch.from_numpy(y_hb_grade.astype(np.int64))
        self.y_sunny = torch.from_numpy(y_sunny.astype(np.float32)).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y_has_palsy[idx],
            self.y_palsy_side[idx],
            self.y_hb_grade[idx],
            self.y_sunny[idx],
        )


# =============================
# 模型定义
# =============================

class ActionSeverityMLP(nn.Module):
    """
    动作严重程度回归
    输入:  fused_action_features (512 维)
    输出:  severity_score (一个实数, 与医生标注回归拟合)
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
    检查级多任务预测:
    - has_palsy      : 是否面瘫 (0/1, 使用 BCEWithLogitsLoss)
    - palsy_side     : 患侧 (0=无, 1=左, 2=右, 使用 CrossEntropyLoss)
    - hb_grade       : HB 分级 (1~6 → 0~5, 使用 CrossEntropyLoss)
    - sunnybrook     : Sunnybrook 总分 (0~100, 回归, MSELoss)
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

        self.head_has_palsy = nn.Linear(hidden_dim, 1)   # logits
        self.head_palsy_side = nn.Linear(hidden_dim, 3)  # 0/1/2
        self.head_hb_grade = nn.Linear(hidden_dim, 6)    # 1~6 → 0~5
        self.head_sunny = nn.Linear(hidden_dim, 1)       # regression

    def forward(self, x):
        h = self.backbone(x)
        out_has_palsy = self.head_has_palsy(h)
        out_side = self.head_palsy_side(h)
        out_hb = self.head_hb_grade(h)
        out_sunny = self.head_sunny(h)
        return out_has_palsy, out_side, out_hb, out_sunny


# =============================
# DB 读取: 动作级数据集
# =============================

def load_action_dataset_from_db(db_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 DB 读取:
    - video_features.fused_action_features
    - action_labels.severity_score

    通过 video_files(examination_id, action_id, video_id) 关联
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    query = """
        SELECT
            vf.fused_action_features,
            al.severity_score
        FROM video_features vf
        JOIN video_files v
          ON v.video_id = vf.video_id
        JOIN action_labels al
          ON al.examination_id = v.examination_id
         AND al.action_id = v.action_id
        WHERE vf.fused_action_features IS NOT NULL
          AND al.severity_score IS NOT NULL
    """
    cur.execute(query)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("Action-level 数据集为空，请检查 video_features / action_labels 是否已导入。")

    X_list = []
    y_list = []

    for fused_blob, severity in rows:
        feat = np.frombuffer(fused_blob, dtype=np.float32)
        X_list.append(feat)
        y_list.append(float(severity))

    X = np.vstack(X_list)   # (N, D)
    y = np.array(y_list, dtype=np.float32)  # (N,)

    print(f"[ActionDataset] 样本数 = {X.shape[0]}, 特征维度 = {X.shape[1]}")
    return X, y


# =============================
# DB 读取: 检查级数据集
# =============================

def load_exam_dataset_from_db(db_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    从 DB 读取:
    1. 每个 examination_id 下面所有 video 的 fused_action_features，做平均得到 exam-level 特征
    2. 对应 examination_labels 的多任务标签
       - has_palsy (0/1)
       - palsy_side (0/1/2)
       - hb_grade (1~6)
       - sunnybrook_score

    只保留四个标签都不为空的检查。
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1) 读取所有已算好 fused_action_features 的 video，并按 examination_id 归类
    cur.execute("""
        SELECT
            v.examination_id,
            vf.fused_action_features
        FROM video_features vf
        JOIN video_files v
          ON v.video_id = vf.video_id
        WHERE vf.fused_action_features IS NOT NULL
    """)
    rows = cur.fetchall()

    exam_feat_dict = {}  # exam_id -> list of feature vectors
    for exam_id, fused_blob in rows:
        feat = np.frombuffer(fused_blob, dtype=np.float32)
        exam_feat_dict.setdefault(exam_id, []).append(feat)

    # 对每个 exam 求平均特征
    exam_feat_avg = {}
    for exam_id, feats in exam_feat_dict.items():
        feats_arr = np.vstack(feats)   # (n_actions, D)
        exam_feat_avg[exam_id] = feats_arr.mean(axis=0)  # (D,)

    if not exam_feat_avg:
        conn.close()
        raise RuntimeError("没有任何 examination 的 fused_action_features，可用，请确认 Stage1-3 已运行。")

    # 2) 读取 examination_labels
    cur.execute("""
        SELECT
            examination_id,
            has_palsy,
            palsy_side,
            hb_grade,
            sunnybrook_score
        FROM examination_labels
        WHERE has_palsy IS NOT NULL
          AND palsy_side IS NOT NULL
          AND hb_grade IS NOT NULL
          AND sunnybrook_score IS NOT NULL
    """)
    label_rows = cur.fetchall()
    conn.close()

    X_list = []
    y_has_palsy = []
    y_palsy_side = []
    y_hb_grade = []
    y_sunny = []

    for exam_id, has_palsy, palsy_side, hb_grade, sunnybrook in label_rows:
        if exam_id not in exam_feat_avg:
            # 该检查还没有成功提取所有动作特征，跳过
            continue

        feat = exam_feat_avg[exam_id]
        X_list.append(feat)

        y_has_palsy.append(int(has_palsy))                # 0/1
        y_palsy_side.append(int(palsy_side))              # 0/1/2
        y_hb_grade.append(int(hb_grade) - 1)              # 映射到 0~5
        y_sunny.append(float(sunnybrook))                 # 0~100

    if not X_list:
        raise RuntimeError("Exam-level 数据集为空，请检查 examination_labels / fused_action_features 是否匹配。")

    X = np.vstack(X_list)
    labels = {
        "has_palsy": np.array(y_has_palsy, dtype=np.float32),
        "palsy_side": np.array(y_palsy_side, dtype=np.int64),
        "hb_grade": np.array(y_hb_grade, dtype=np.int64),
        "sunnybrook": np.array(y_sunny, dtype=np.float32),
    }

    print(f"[ExamDataset] 样本数 = {X.shape[0]}, 特征维度 = {X.shape[1]}")
    return X, labels


# =============================
# 训练函数: 动作级
# =============================

def train_action_model(X: np.ndarray, y: np.ndarray, device: str) -> Tuple[ActionSeverityMLP, Dict]:
    X_train, X_val, (y_train_val,) = split_train_val(X, y, val_ratio=VAL_RATIO)
    y_train, y_val = y_train_val

    train_ds = ActionDataset(X_train, y_train)
    val_ds = ActionDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X.shape[1]
    model = ActionSeverityMLP(input_dim=input_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_ACTION, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS_ACTION + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Action] Epoch {epoch:03d}/{EPOCHS_ACTION} "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

    # 恢复最优
    model.load_state_dict(best_state)
    print(f"[Action] 训练完成, 最佳验证 MSE = {best_val_loss:.4f}")
    return model, history


# =============================
# 训练函数: 检查级多任务
# =============================

def train_exam_model(X: np.ndarray, labels: Dict[str, np.ndarray], device: str) -> Tuple[ExamMultiTaskMLP, Dict]:
    y_has_palsy = labels["has_palsy"]
    y_palsy_side = labels["palsy_side"]
    y_hb_grade = labels["hb_grade"]
    y_sunny = labels["sunnybrook"] / 100.0  # Sunnybrook 缩放到 0~1 之间, 有利于收敛

    X_train, X_val, ys_train_val = split_train_val(
        X,
        y_has_palsy,
        y_palsy_side,
        y_hb_grade,
        y_sunny,
        val_ratio=VAL_RATIO,
    )
    (hp_train, hp_val), (side_train, side_val), (hb_train, hb_val), (sunny_train, sunny_val) = ys_train_val

    train_ds = ExamDataset(X_train, hp_train, side_train, hb_train, sunny_train)
    val_ds = ExamDataset(X_val, hp_val, side_val, hb_val, sunny_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X.shape[1]
    model = ExamMultiTaskMLP(input_dim=input_dim).to(device)

    # 损失函数
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_EXAM, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS_EXAM + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            xb, y_hp, y_side, y_hb, y_sunny = batch
            xb = xb.to(device)
            y_hp = y_hp.to(device)
            y_side = y_side.to(device)
            y_hb = y_hb.to(device)
            y_sunny = y_sunny.to(device)

            out_hp, out_side, out_hb, out_sunny = model(xb)

            loss_hp = bce(out_hp, y_hp)
            loss_side = ce(out_side, y_side)
            loss_hb = ce(out_hb, y_hb)
            loss_sunny = mse(out_sunny, y_sunny)

            # 简单加权和, 可后续调参
            loss = 0.5 * loss_hp + 0.5 * loss_side + 1.0 * loss_hb + 1.0 * loss_sunny

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                xb, y_hp, y_side, y_hb, y_sunny = batch
                xb = xb.to(device)
                y_hp = y_hp.to(device)
                y_side = y_side.to(device)
                y_hb = y_hb.to(device)
                y_sunny = y_sunny.to(device)

                out_hp, out_side, out_hb, out_sunny = model(xb)

                loss_hp = bce(out_hp, y_hp)
                loss_side = ce(out_side, y_side)
                loss_hb = ce(out_hb, y_hb)
                loss_sunny = mse(out_sunny, y_sunny)

                loss = 0.5 * loss_hp + 0.5 * loss_side + 1.0 * loss_hb + 1.0 * loss_sunny
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Exam] Epoch {epoch:03d}/{EPOCHS_EXAM} "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    print(f"[Exam] 训练完成, 最佳验证综合损失 = {best_val_loss:.4f}")
    return model, history


# =============================
# 主流程
# =============================

def main():
    set_seed(RANDOM_SEED)
    ensure_model_dir()

    db_path = DB_PATH_DEFAULT

    # 1. 读取数据集
    print("\n====== 读取动作级数据集 ======")
    X_action, y_severity = load_action_dataset_from_db(db_path)

    print("\n====== 读取检查级数据集 ======")
    X_exam, exam_labels = load_exam_dataset_from_db(db_path)

    # 2. 计算特征标准化 (用所有 fused_action_features)
    all_features = np.vstack([X_action, X_exam])
    feat_mean = all_features.mean(axis=0, keepdims=True)
    feat_std = all_features.std(axis=0, keepdims=True) + 1e-6

    X_action_norm = (X_action - feat_mean) / feat_std
    X_exam_norm = (X_exam - feat_mean) / feat_std

    np.savez(
        SCALER_PATH,
        mean=feat_mean,
        std=feat_std,
    )
    print(f"\n[Scaler] 已保存特征标准化参数到: {SCALER_PATH}")

    # 3. 选择设备
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[Device] 使用设备: {device}")

    # 4. 训练动作级模型
    print("\n====== 训练 ActionSeverityMLP ======")
    action_model, action_hist = train_action_model(X_action_norm, y_severity, device=device)
    torch.save(action_model.state_dict(), ACTION_MODEL_PATH)
    print(f"[Action] 模型已保存到: {ACTION_MODEL_PATH}")

    # 5. 训练检查级多任务模型
    print("\n====== 训练 ExamMultiTaskMLP ======")
    exam_model, exam_hist = train_exam_model(X_exam_norm, exam_labels, device=device)
    torch.save(exam_model.state_dict(), EXAM_MODEL_PATH)
    print(f"[Exam] 模型已保存到: {EXAM_MODEL_PATH}")

    print("\n✅ Stage 4 模型训练完成。后续可以写 Stage 5: 使用这两个模型回写预测结果到 DB / 提供 API。")


if __name__ == "__main__":
    main()
