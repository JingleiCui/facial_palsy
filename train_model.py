"""
H-GFA Net 训练代码 - 最小改动版
保留visual_features的1280维度，使用正确的数据库结构
充分利用MPS加速
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入Stage模块
from stage1_cdcaf import CDCAF
from stage2_gqca import GQCA
from stage3_mfa import MFA


# =========================
# 配置参数
# =========================
DB_PATH = 'facialPalsy.db'
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# 使用MPS加速（MacBook Pro）
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 使用 Apple Silicon MPS 加速")
    # MPS优化设置
    torch.mps.set_per_process_memory_fraction(0.0)  # 自动管理内存
else:
    DEVICE = torch.device("cpu")

# 动作维度配置（从数据统计得出）
ACTION_DIMS = {
    'BlowCheek': (5, 2),
    'CloseEyeHardly': (10, 8),
    'CloseEyeSoftly': (7, 4),
    'LipPucker': (5, 2),
    'NeutralFace': (8, 0),
    'RaiseEyebrow': (10, 3),
    'ShowTeeth': (7, 3),
    'ShrugNose': (5, 2),
    'Smile': (11, 4),
    'SpontaneousEyeBlink': (5, 7),
    'VoluntaryEyeBlink': (5, 4),
}


# =========================
# 1. 数据集类
# =========================
class FacialPalsyDataset(Dataset):
    """面瘫评估数据集 - 支持动作分组和可变维度"""

    def __init__(self, video_ids, data_dict, labels, action_names):
        self.video_ids = video_ids
        self.data_dict = data_dict
        self.labels = labels
        self.action_names = action_names

        # 按动作分组，便于批处理
        self.action_groups = self._group_by_action()

    def _group_by_action(self):
        """按动作类型分组"""
        groups = defaultdict(list)
        for idx, vid in enumerate(self.video_ids):
            action = self.action_names[vid]
            groups[action].append(idx)
        return groups

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]

        # 获取特征（注意处理可能的None值）
        static_feat = self.data_dict['static_features'][idx]
        dynamic_feat = self.data_dict['dynamic_features'][idx]
        visual_feat = self.data_dict['visual_features'][idx]

        # 转换为tensor，确保在MPS设备上的数据类型正确
        static_tensor = torch.tensor(static_feat, dtype=torch.float32).contiguous()
        dynamic_tensor = torch.tensor(dynamic_feat, dtype=torch.float32).contiguous() if dynamic_feat is not None and len(dynamic_feat) > 0 else None
        visual_tensor = torch.tensor(visual_feat, dtype=torch.float32).contiguous()

        return {
            'video_id': vid,
            'action': self.action_names[vid],
            'static': static_tensor,
            'dynamic': dynamic_tensor,
            'visual': visual_tensor,
            'label': self.labels[vid] - 1  # 转为0-4（原始是1-5）
        }


# =========================
# 2. 自定义collate函数（关键！）
# =========================
def collate_fn_padding(batch):
    """
    使用 padding 处理不同维度的特征
    按动作分组；对每个动作内部的样本分别做 padding
    """
    from collections import defaultdict
    import torch

    # 1. 按动作分组
    action_groups = defaultdict(list)
    for item in batch:
        action_groups[item['action']].append(item)

    batch_dict = {}

    for action, items in action_groups.items():
        if not items:
            continue

        # ---------- 静态特征维度 ----------
        max_static_dim = max(item['static'].shape[0] for item in items)

        # ---------- 动态特征维度（可能全是 None） ----------
        dyn_dims = [
            item['dynamic'].shape[0]
            for item in items
            if (item['dynamic'] is not None and item['dynamic'].numel() > 0)
        ]
        if len(dyn_dims) > 0:
            max_dynamic_dim = max(dyn_dims)
        else:
            # 这一组动作（比如 NeutralFace）完全没有动态特征
            max_dynamic_dim = 0

        padded_static = []
        padded_dynamic = []
        visual_list = []
        label_list = []

        for item in items:
            # ---------- 静态特征 padding ----------
            static = item['static']
            if static.shape[0] < max_static_dim:
                pad = torch.zeros(max_static_dim - static.shape[0], dtype=static.dtype)
                static = torch.cat([static, pad])
            padded_static.append(static)

            # ---------- 动态特征 padding ----------
            if max_dynamic_dim > 0:
                dyn = item['dynamic']
                if dyn is None or dyn.numel() == 0:
                    # 没有动态特征，用全 0 填充
                    dyn = torch.zeros(max_dynamic_dim, dtype=static.dtype)
                elif dyn.shape[0] < max_dynamic_dim:
                    pad = torch.zeros(max_dynamic_dim - dyn.shape[0], dtype=dyn.dtype)
                    dyn = torch.cat([dyn, pad])
                padded_dynamic.append(dyn)

            # ---------- 视觉特征 & 标签 ----------
            visual_list.append(item['visual'])
            label_list.append(item['label'])

        batch_dict[action] = {
            'static': torch.stack(padded_static),
            'dynamic': torch.stack(padded_dynamic) if max_dynamic_dim > 0 else None,
            'visual': torch.stack(visual_list),
            'labels': torch.tensor(label_list, dtype=torch.long),
            'static_dim': max_static_dim,
            'dynamic_dim': max_dynamic_dim,
        }

    return batch_dict



# =========================
# 3. H-GFA Net模型（最小改动版）
# =========================
class HGFANet(nn.Module):
    """
    Hierarchical Geometry-Visual Fusion Attention Network
    最小改动：为每个动作创建独立的Stage1编码器
    保留visual_features的1280维
    """

    def __init__(self, action_dims=ACTION_DIMS, num_classes=5, device='cpu'):
        super().__init__()
        self.action_dims = action_dims
        self.num_classes = num_classes  # 5个严重程度等级
        self.device = device

        # 为每个动作创建独立的Stage1编码器（处理可变维度）
        self.stage1_modules = nn.ModuleDict()

        # 找出最大维度（用于创建统一的编码器）
        max_static_dim = max(d[0] for d in action_dims.values())
        max_dynamic_dim = max(d[1] for d in action_dims.values())

        for action, (s_dim, d_dim) in action_dims.items():
            # 每个动作用自己真实的静态/动态特征维度
            self.stage1_modules[action] = CDCAF(
                static_dim=s_dim,
                dynamic_dim=d_dim,
                clinical_dim=0,
                d_model=128,
                num_layers=2,
                num_heads=4,
                output_dim=256
            )

        # Stage2: 几何引导视觉注意力（共享）
        self.stage2 = GQCA(
            geo_dim=256,
            visual_dim=1280,  # 保持原始维度
            d_model=256,
            num_heads=8,
            num_layers=2,
            num_tokens=49,
            out_dim=256
        )

        # Stage3: 多模态融合（共享）
        self.stage3 = MFA(
            geo_dim=256,
            visual_guided_dim=256,
            visual_global_dim=1280,  # 保持原始维度
            feature_dim=256,
            num_heads=4,
            num_layers=2,
            output_dim=512
        )

        # 分类头（5个严重程度等级）
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # 将模型移到设备
        self.to(device)

    def forward(self, batch_dict):
        """
        前向传播
        batch_dict: 按动作分组的批次数据
        """
        all_logits = []
        all_labels = []

        for action, data in batch_dict.items():
            # 移动到设备（MPS优化）
            static = data['static'].to(self.device, non_blocking=True)
            dynamic = data['dynamic'].to(self.device, non_blocking=True) if data['dynamic'] is not None else None
            visual = data['visual'].to(self.device, non_blocking=True)
            labels = data['labels'].to(self.device, non_blocking=True)

            # Stage1: 几何特征融合（动作特异性）
            stage1_module = self.stage1_modules[action]
            geo_refined = stage1_module(static, dynamic)  # (B, 256)

            # Stage2: 几何引导视觉注意力
            visual_guided = self.stage2(geo_refined, visual)  # (B, 256)

            # Stage3: 多模态融合
            fused = self.stage3(geo_refined, visual_guided, visual)  # (B, 512)

            # 分类
            logits = self.classifier(fused)  # (B, 5)

            all_logits.append(logits)
            all_labels.append(labels)

        # 合并所有动作的结果
        if all_logits:
            combined_logits = torch.cat(all_logits, dim=0)
            combined_labels = torch.cat(all_labels, dim=0)
        else:
            # 处理空批次
            combined_logits = torch.empty(0, self.num_classes, device=self.device)
            combined_labels = torch.empty(0, dtype=torch.long, device=self.device)

        return combined_logits, combined_labels


# =========================
# 4. 数据加载函数（使用正确的数据库表）
# =========================
def load_data_from_db(db_path):
    """从数据库加载训练数据"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 正确的查询：从video_features和action_labels表获取数据
    query = """
    SELECT 
        vf.video_id,
        at.action_name_en,
        vfeat.static_features,
        vfeat.dynamic_features,
        vfeat.visual_features,
        vfeat.static_dim,
        vfeat.dynamic_dim,
        al.severity_score
    FROM video_files vf
    INNER JOIN video_features vfeat ON vf.video_id = vfeat.video_id
    INNER JOIN action_types at ON vf.action_id = at.action_id
    INNER JOIN action_labels al ON vf.examination_id = al.examination_id 
        AND vf.action_id = al.action_id
    WHERE vfeat.static_features IS NOT NULL
      AND vfeat.visual_features IS NOT NULL
      AND al.severity_score BETWEEN 1 AND 5
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    print(f"\n[DataLoader] 从数据库加载 {len(rows)} 个样本")

    # 解析数据
    video_ids = []
    action_names = {}
    static_features = []
    dynamic_features = []
    visual_features = []
    labels = {}

    # 动作分布统计
    action_counts = defaultdict(int)
    dim_stats = defaultdict(list)

    for row in rows:
        vid, action, static_blob, dynamic_blob, visual_blob, s_dim, d_dim, severity = row

        # 跳过无效数据
        if not static_blob or not visual_blob:
            continue

        # 解析特征（从BLOB）
        static_feat = np.frombuffer(static_blob, dtype=np.float32)
        dynamic_feat = np.frombuffer(dynamic_blob, dtype=np.float32) if dynamic_blob else np.zeros(0, dtype=np.float32)
        visual_feat = np.frombuffer(visual_blob, dtype=np.float32)

        # 验证维度
        if visual_feat.shape[0] != 1280:
            print(f"警告：视觉特征维度不正确 {vid}: {visual_feat.shape}")
            continue

        video_ids.append(vid)
        action_names[vid] = action
        static_features.append(static_feat)
        dynamic_features.append(dynamic_feat if len(dynamic_feat) > 0 else None)
        visual_features.append(visual_feat)
        labels[vid] = severity

        action_counts[action] += 1
        dim_stats[action].append((s_dim, d_dim))

    print(f"[DataLoader] 有效样本数: {len(video_ids)}")
    print("[DataLoader] 动作分布:")
    for action in sorted(action_counts.keys()):
        dims = dim_stats[action]
        if dims:
            s_dims = [d[0] for d in dims]
            d_dims = [d[1] for d in dims]
            print(f"  {action:25} {action_counts[action]:3} 个样本, "
                  f"静态维度={min(s_dims)}-{max(s_dims)}, "
                  f"动态维度={min(d_dims)}-{max(d_dims)}")

    # 构建数据字典
    data_dict = {
        'static_features': static_features,
        'dynamic_features': dynamic_features,
        'visual_features': visual_features
    }

    return data_dict, labels, action_names


# =========================
# 5. 训练器类（MPS优化）
# =========================
class Trainer:
    """H-GFA Net训练器 - MPS优化版"""

    def __init__(self, model, train_loader, val_loader, device,
                 learning_rate=1e-4, weight_decay=1e-4, save_dir='checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 优化器（MPS优化）
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=True if device.type == 'cuda' else False  # CUDA支持fused优化
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 保存路径
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_samples = 0
        correct = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, batch_dict in enumerate(pbar):
            if not batch_dict:
                continue

            # 清零梯度
            self.optimizer.zero_grad(set_to_none=True)  # MPS优化：set_to_none更高效

            # 前向传播
            logits, labels = self.model(batch_dict)

            if logits.numel() == 0:
                continue

            # 计算损失
            loss = self.criterion(logits, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()

            # 统计
            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            num_samples += batch_size

            # 计算准确率
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()

            # 更新进度条
            if num_samples > 0:
                avg_loss = epoch_loss / num_samples
                acc = correct / num_samples
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{acc:.4f}',
                    'batch': f'{batch_idx+1}/{len(self.train_loader)}'
                })

            # MPS内存管理
            if self.device.type == 'mps' and batch_idx % 10 == 0:
                torch.mps.synchronize()  # 定期同步，防止内存累积

        return epoch_loss / num_samples if num_samples > 0 else 0.0

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        val_loss = 0.0
        num_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

            for batch_dict in pbar:
                if not batch_dict:
                    continue

                # 前向传播
                logits, labels = self.model(batch_dict)

                if logits.numel() == 0:
                    continue

                # 计算损失
                loss = self.criterion(logits, labels)

                # 统计
                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size
                num_samples += batch_size

                # 预测
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 更新进度条
                if num_samples > 0:
                    pbar.set_postfix({'loss': f'{val_loss/num_samples:.4f}'})

        if num_samples > 0:
            avg_loss = val_loss / num_samples
            accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

            self.val_losses.append(avg_loss)
            self.val_accs.append(accuracy)

            # 更新最佳模型
            if avg_loss < self.best_val_loss:
                self.best_val_loss = avg_loss
                self.best_val_acc = accuracy
                self.save_checkpoint(epoch, 'best_model.pth')

            return avg_loss, accuracy, all_preds, all_labels
        else:
            return 0.0, 0.0, [], []

    def save_checkpoint(self, epoch, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'action_dims': self.model.action_dims,
            'device': str(self.device)
        }

        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"\n[Checkpoint] 模型已保存: {save_path}")

    def plot_curves(self):
        """ 画出训练 / 验证曲线并保存为 PNG"""
        if len(self.train_losses) == 0 or len(self.val_losses) == 0:
            print("⚠ 没有可用的训练记录，无法画曲线")
            return

        epochs = range(1, len(self.train_losses) + 1)

        # 1) 损失曲线
        plt.figure()
        plt.plot(epochs, self.train_losses, marker='o', label='Train Loss')
        plt.plot(epochs, self.val_losses, marker='o', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train & Val Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        loss_path = self.save_dir / 'loss_curve.png'
        plt.savefig(loss_path, dpi=300)
        plt.close()
        print(f"📉 损失曲线已保存: {loss_path}")

        # 2) 验证准确率曲线
        if len(self.val_accs) > 0:
            plt.figure()
            plt.plot(epochs, self.val_accs, marker='o', label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Val Accuracy')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            acc_path = self.save_dir / 'val_acc_curve.png'
            plt.savefig(acc_path, dpi=300)
            plt.close()
            print(f" 验证准确率曲线已保存: {acc_path}")

    def train(self, num_epochs):
        """完整训练循环"""
        print("\n" + "="*60)
        print(" 开始训练 H-GFA Net")
        print(f" 设备: {self.device}")
        print("="*60)

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")

            # 训练
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # 验证
            val_loss, val_acc, preds, labels = self.validate(epoch)

            # 更新学习率
            self.scheduler.step(val_loss)

            # 打印结果
            print(f"\n[Epoch {epoch}] 结果:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  验证准确率: {val_acc:.4f}")

            # 每5个epoch打印详细报告
            if epoch % 5 == 0 and len(labels) > 0:
                print("\n分类报告（严重程度1-5级）:")
                print(classification_report(
                    labels, preds,
                    target_names=[f"Grade {i+1}" for i in range(5)],
                    zero_division=0
                ))

            # 每10个epoch保存检查点
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')

            # MPS内存清理
            if self.device.type == 'mps':
                torch.mps.empty_cache()

        print("\n" + "="*60)
        print("✅ 训练完成!")
        print(f"📊 最佳验证损失: {self.best_val_loss:.4f}")
        print(f"🎯 最佳验证准确率: {self.best_val_acc:.4f}")
        print("="*60)
        self.plot_curves()


# =========================
# 6. 主函数
# =========================
def main():
    """主训练流程"""
    print("="*60)
    print("H-GFA Net 训练 - MacBook Pro MPS优化版")
    print("="*60)
    print(f"📱 设备: {DEVICE}")
    print(f"📦 批大小: {BATCH_SIZE}")
    print(f"🔄 训练轮数: {NUM_EPOCHS}")
    print(f"📈 学习率: {LEARNING_RATE}")
    print(f"🎯 任务: 面瘫严重程度分级（1-5级）")

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    data_dict, labels, action_names = load_data_from_db(DB_PATH)

    if not labels:
        print("❌ 错误：没有找到有效的训练数据")
        print("请检查数据库是否包含必要的数据")
        return

    # 2. 划分数据集
    print("\n[2/5] 划分数据集...")
    all_video_ids = list(labels.keys())

    # 按严重程度分层抽样
    train_ids, val_ids = train_test_split(
        all_video_ids,
        test_size=0.2,
        random_state=42,
        stratify=[labels[vid] for vid in all_video_ids]
    )

    print(f"  训练集: {len(train_ids)} 个样本")
    print(f"  验证集: {len(val_ids)} 个样本")

    # 统计严重程度分布
    train_severity = [labels[vid] for vid in train_ids]
    val_severity = [labels[vid] for vid in val_ids]
    print("\n严重程度分布:")
    for i in range(1, 6):
        train_count = train_severity.count(i)
        val_count = val_severity.count(i)
        print(f"  Grade {i}: 训练={train_count}, 验证={val_count}")

    # 3. 创建数据集
    print("\n[3/5] 创建数据加载器...")

    # 分割数据
    def get_indices(video_ids, all_ids):
        return [all_ids.index(vid) for vid in video_ids]

    train_indices = get_indices(train_ids, all_video_ids)
    val_indices = get_indices(val_ids, all_video_ids)

    train_data_dict = {
        'static_features': [data_dict['static_features'][i] for i in train_indices],
        'dynamic_features': [data_dict['dynamic_features'][i] for i in train_indices],
        'visual_features': [data_dict['visual_features'][i] for i in train_indices]
    }

    val_data_dict = {
        'static_features': [data_dict['static_features'][i] for i in val_indices],
        'dynamic_features': [data_dict['dynamic_features'][i] for i in val_indices],
        'visual_features': [data_dict['visual_features'][i] for i in val_indices]
    }

    train_dataset = FacialPalsyDataset(train_ids, train_data_dict, labels, action_names)
    val_dataset = FacialPalsyDataset(val_ids, val_data_dict, labels, action_names)

    # 创建数据加载器（MPS优化：pin_memory对MPS无效，使用non_blocking）
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_padding,
        num_workers=0,  # MPS不支持多进程
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_padding,
        num_workers=0,
        persistent_workers=False
    )

    # 4. 统计动作维度
    print("\n[4/5] 统计动作维度...")
    print(f"共 {len(ACTION_DIMS)} 个动作:")
    for action, (s_dim, d_dim) in ACTION_DIMS.items():
        print(f"  {action:25} 静态={s_dim:<3} 动态={d_dim:<3}")

    # 5. 创建模型
    print("\n[5/5] 创建模型...")
    model = HGFANet(
        action_dims=ACTION_DIMS,
        num_classes=5,  # 5个严重程度等级
        device=DEVICE
    )

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 总参数量: {total_params:,}")
    print(f"🎯 可训练参数: {trainable_params:,}")
    print(f"💾 模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    # 6. 创建训练器并开始训练
    print("\n开始训练...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        save_dir='checkpoints'
    )

    # 开始训练
    trainer.train(num_epochs=NUM_EPOCHS)

    print("\n✅ 训练完成！")
    print(f"📁 模型已保存到 checkpoints/ 目录")
    print(f"🎯 最佳模型: checkpoints/best_model.pth")

if __name__ == "__main__":
    main()