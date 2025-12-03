#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H-GFA Net训练脚本
多任务学习 + 不确定性加权损失

功能：
1. 从数据库加载训练和验证数据（通过dataset_members表）
2. 训练H-GFA Net模型（action-level severity分类）
3. 使用不确定性加权的多任务学习
4. 保存最佳模型和训练曲线

直接在PyCharm中点击运行即可！
修改下面的FOLD变量来训练不同的fold (0, 1, 2)
"""

import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
# 训练哪一个fold (0, 1, 或 2)
FOLD = 2  # <--- 修改这里来训练不同的fold

# 数据库路径
DB_PATH = "facialPalsy.db"

# 输出路径
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / f"fold_{FOLD}"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# 设备配置
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 训练超参数
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 25
GRAD_CLIP_NORM = 1.0

# 模型参数
INPUT_DIM = 512
HIDDEN_DIM = 256
NUM_CLASSES = 5  # severity grades: 1-5
NUM_HEADS = 8
DROPOUT = 0.1

# Focal Loss参数
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1

# 数据集版本
SPLIT_VERSION = "v1.0"

# ===== 统一的特征解码函数（float32 BLOB → np.ndarray） =====
def decode_feature(blob, dim):
    """
    将数据库中的 float32 BLOB 解码为 numpy 一维数组
    """
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32, count=dim)

# ==================== 数据集类 ====================
class FacialPalsyDataset(Dataset):
    """面部麻痹数据集"""

    def __init__(self, db_path, fold, split_type='train', split_version='v1.0'):
        self.db_path = db_path
        self.fold = fold
        self.split_type = split_type
        self.split_version = split_version
        self.samples = self.load_data()

    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 通过dataset_members表获取该fold的患者
        cursor.execute("""
            SELECT 
                vf.feature_id,
                vf.fused_action_features,
                al.severity_score,
                at.action_name_en
            FROM video_features vf
            JOIN video_files v ON vf.video_id = v.video_id
            JOIN examinations e ON v.examination_id = e.examination_id
            JOIN action_labels al ON e.examination_id = al.examination_id 
                                  AND v.action_id = al.action_id
            JOIN action_types at ON v.action_id = at.action_id
            JOIN dataset_members dm ON e.patient_id = dm.patient_id
            JOIN dataset_splits ds ON dm.split_id = ds.split_id
            WHERE vf.fused_action_features IS NOT NULL
              AND al.severity_score IS NOT NULL
              AND ds.split_type = ?
              AND dm.fold_number = ?
              AND ds.split_version = ?
            ORDER BY vf.feature_id
        """, (self.split_type, self.fold, self.split_version))

        samples = []
        for row in cursor.fetchall():
            feature_id, fused_blob, severity, action_name = row
            fused_features = decode_feature(fused_blob, INPUT_DIM)

            samples.append({
                'feature_id': feature_id,
                'features': fused_features,
                'severity': severity - 1,  # 转换为0-4索引
                'action_name': action_name
            })

        conn.close()
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.FloatTensor(sample['features'])
        severity = torch.LongTensor([sample['severity']])[0]
        return features, severity


# ==================== 模型定义 ====================
class HGFANet(nn.Module):
    """H-GFA Net主模型"""

    def __init__(self, input_dim=512, hidden_dim=256, num_classes=5,
                 num_heads=8, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Severity classification head
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        severity_logits = self.severity_head(x)
        return severity_logits


# ==================== 损失函数 ====================
class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""

    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class MultiTaskLoss(nn.Module):
    """不确定性加权的多任务损失"""

    def __init__(self, num_tasks=1):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        weighted_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss += precision * loss + self.log_vars[i]
        return weighted_loss


# ==================== 训练函数 ====================
def train_epoch(model, dataloader, optimizer, focal_loss, multi_task_loss, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for features, severity in tqdm(dataloader, desc="Training", leave=False):
        features = features.to(device)
        severity = severity.to(device)

        severity_logits = model(features)
        severity_loss = focal_loss(severity_logits, severity)
        loss = multi_task_loss([severity_loss])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()

        preds = severity_logits.argmax(dim=1).cpu().numpy()
        labels = severity.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def validate(model, dataloader, focal_loss, multi_task_loss, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, severity in tqdm(dataloader, desc="Validating", leave=False):
            features = features.to(device)
            severity = severity.to(device)

            severity_logits = model(features)
            severity_loss = focal_loss(severity_logits, severity)
            loss = multi_task_loss([severity_loss])

            total_loss += loss.item()

            preds = severity_logits.argmax(dim=1).cpu().numpy()
            labels = severity.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    return avg_loss, accuracy, precision, recall, f1


def plot_training_curves(history, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Acc', marker='o', markersize=3)
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==================== 主训练流程 ====================
def train_model():
    """主训练流程"""
    print("=" * 60)
    print(f"H-GFA Net训练 - Fold {FOLD}")
    print("=" * 60)
    print(f"数据库路径: {DB_PATH}")
    print(f"检查点目录: {CHECKPOINT_DIR}")
    print(f"设备: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print()

    # 1. 加载数据
    print("正在加载数据...")
    train_dataset = FacialPalsyDataset(DB_PATH, FOLD, 'train', SPLIT_VERSION)
    val_dataset = FacialPalsyDataset(DB_PATH, FOLD, 'val', SPLIT_VERSION)

    print(f"✓ 训练集: {len(train_dataset)} 样本")
    print(f"✓ 验证集: {len(val_dataset)} 样本")
    print()

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("❌ 错误: 训练集或验证集为空！")
        print("请先运行 dataset_split.py")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. 创建模型
    print("创建模型...")
    model = HGFANet(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ 模型已创建")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print()

    # 3. 创建损失函数和优化器
    focal_loss = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    multi_task_loss = MultiTaskLoss(num_tasks=1).to(DEVICE)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(multi_task_loss.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # 4. 训练循环
    print("开始训练...")
    print()

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, focal_loss, multi_task_loss, DEVICE
        )

        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(
            model, val_loader, focal_loss, multi_task_loss, DEVICE
        )

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, CHECKPOINT_DIR / "best.pth")

            print(f"  ✓ 保存最佳模型 (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered (patience={EARLY_STOPPING_PATIENCE})")
            break

        print()

    # 5. 保存最终模型和训练曲线
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss
    }, CHECKPOINT_DIR / "final.pth")

    plot_training_curves(history, CHECKPOINT_DIR / "training_curves.png")

    print("=" * 60)
    print(f"✓ 训练完成！Fold {FOLD}")
    print(f"✓ 最佳验证准确率: {best_val_acc:.4f}")
    print(f"✓ 模型保存在: {CHECKPOINT_DIR}")
    print("=" * 60)
    print()
    print("下一步: 运行 evaluate_model.py 评估模型")


if __name__ == "__main__":
    train_model()