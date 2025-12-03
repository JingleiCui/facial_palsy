#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H-GFA Net模型评估脚本

功能：
1. 加载训练好的模型
2. 在验证集上评估
3. 计算准确率、精确率、召回率、F1分数
4. 生成混淆矩阵可视化
5. 生成详细的分类报告

直接在PyCharm中点击运行即可！
修改下面的FOLD和CHECKPOINT_TYPE来评估不同的模型
"""

import sqlite3
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
# 评估哪一个fold (0, 1, 或 2)
FOLD = 2  # <--- 修改这里来评估不同的fold

# 使用哪个检查点 ("best" 或 "final")
CHECKPOINT_TYPE = "best"  # <--- 修改这里来评估不同的检查点

# 数据库路径
DB_PATH = "facialPalsy.db"

# 检查点路径
CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / f"fold_{FOLD}" / f"{CHECKPOINT_TYPE}.pth"

# 输出路径
OUTPUT_DIR = Path(__file__).parent / "evaluation_results" / f"fold_{FOLD}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设备配置
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 模型参数（必须与训练时一致）
INPUT_DIM = 512
HIDDEN_DIM = 256
NUM_CLASSES = 5
NUM_HEADS = 8
DROPOUT = 0.1

# 批量大小
BATCH_SIZE = 32

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

    def __init__(self, db_path, fold, split_type='val', split_version='v1.0'):
        self.db_path = db_path
        self.fold = fold
        self.split_type = split_type
        self.split_version = split_version
        self.samples = self.load_data()

    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

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


# ==================== 评估函数 ====================
def evaluate_model(model, dataloader, device):
    """评估模型并返回预测和标签"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, severity in tqdm(dataloader, desc="Evaluating"):
            features = features.to(device)
            severity = severity.to(device)

            severity_logits = model(features)
            probs = torch.softmax(severity_logits, dim=1)

            preds = severity_logits.argmax(dim=1).cpu().numpy()
            labels = severity.cpu().numpy()
            probs_np = probs.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs_np)

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5'],
        yticklabels=['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5']
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Severity Classification')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_classification_report(y_true, y_pred, save_path):
    """生成并保存分类报告"""
    target_names = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Severity Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n")

    return report


# ==================== 主评估流程 ====================
def run_evaluation():
    """主评估流程"""
    print("=" * 60)
    print(f"H-GFA Net模型评估 - Fold {FOLD}")
    print("=" * 60)
    print(f"数据库路径: {DB_PATH}")
    print(f"检查点路径: {CHECKPOINT_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"设备: {DEVICE}")
    print()

    # 1. 检查检查点是否存在
    if not CHECKPOINT_PATH.exists():
        print(f"❌ 错误: 检查点文件不存在: {CHECKPOINT_PATH}")
        print("请先运行 train_hgfa_net.py 训练模型")
        return

    # 2. 加载数据
    print("正在加载验证数据...")
    val_dataset = FacialPalsyDataset(DB_PATH, FOLD, 'val', SPLIT_VERSION)
    print(f"✓ 验证集: {len(val_dataset)} 样本")
    print()

    if len(val_dataset) == 0:
        print("❌ 错误: 验证集为空！")
        return

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 创建模型并加载权重
    print("正在加载模型...")
    model = HGFANet(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    ).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✓ 模型已加载")
    print(f"  训练轮次: {checkpoint['epoch'] + 1}")
    print(f"  验证准确率: {checkpoint['val_acc']:.4f}")
    print(f"  验证损失: {checkpoint['val_loss']:.4f}")
    print()

    # 4. 评估模型
    print("开始评估...")
    predictions, labels, probabilities = evaluate_model(model, val_loader, DEVICE)
    print("✓ 评估完成")
    print()

    # 5. 计算指标
    print("=" * 60)
    print("评估结果:")
    print("=" * 60)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    print(f"整体指标:")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    print(f"  F1分数 (F1-Score): {f1:.4f}")
    print()

    # 6. 生成混淆矩阵
    print("生成混淆矩阵...")
    cm_path = OUTPUT_DIR / "severity_confusion_matrix.png"
    plot_confusion_matrix(labels, predictions, cm_path)
    print(f"✓ 混淆矩阵已保存: {cm_path}")
    print()

    # 7. 生成分类报告
    print("生成分类报告...")
    report_path = OUTPUT_DIR / "severity_classification_report.txt"
    report = generate_classification_report(labels, predictions, report_path)
    print(report)
    print(f"✓ 分类报告已保存: {report_path}")
    print()

    # 8. 每个类别的统计
    print("=" * 60)
    print("各严重程度等级的详细统计:")
    print("=" * 60)
    for i in range(NUM_CLASSES):
        mask = labels == i
        if mask.sum() > 0:
            acc = (predictions[mask] == i).mean()
            print(f"Grade {i + 1}: {mask.sum()} 样本, 准确率: {acc:.4f}")
    print()

    print("=" * 60)
    print("✓ 评估完成！")
    print(f"✓ 结果保存在: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()