"""
H-GFA Net 层级多任务训练脚本
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from hgfa_net import HGFANet
from multi_task_loss import HierarchicalMultiTaskLoss
from dataset_palsy import HierarchicalPalsyDataset, collate_hierarchical


def train_epoch(model, dataloader, optimizer, loss_fn, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        # 移动到设备
        batch = move_to_device(batch, device)

        # 前向传播
        outputs = model(batch)

        # 计算损失
        loss, loss_info = loss_fn(outputs, batch['targets'])

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    # 收集预测和标签
    all_preds = {
        'has_palsy': [], 'palsy_side': [], 'hb_grade': [], 'sunnybrook': []
    }
    all_labels = {
        'has_palsy': [], 'palsy_side': [], 'hb_grade': [], 'sunnybrook': []
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch = move_to_device(batch, device)
            outputs = model(batch)
            loss, _ = loss_fn(outputs, batch['targets'])

            total_loss += loss.item()
            num_batches += 1

            # 收集预测
            session = outputs['session_outputs']
            for task in ['has_palsy', 'palsy_side', 'hb_grade']:
                preds = session[task].argmax(dim=1).cpu().numpy()
                labels = batch['targets'][task].cpu().numpy()
                all_preds[task].extend(preds)
                all_labels[task].extend(labels)

            all_preds['sunnybrook'].extend(session['sunnybrook'].cpu().numpy() * 100)
            all_labels['sunnybrook'].extend(batch['targets']['sunnybrook'].cpu().numpy())

    # 计算指标
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / num_batches

    return metrics


def compute_metrics(preds, labels):
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

    metrics = {}

    for task in ['has_palsy', 'palsy_side', 'hb_grade']:
        metrics[f'{task}_acc'] = accuracy_score(labels[task], preds[task])
        metrics[f'{task}_f1'] = f1_score(labels[task], preds[task], average='weighted')

    metrics['sunnybrook_mae'] = mean_absolute_error(labels['sunnybrook'], preds['sunnybrook'])

    return metrics


def move_to_device(batch, device):
    """递归移动batch到设备"""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    return batch


def main(args):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建数据集
    train_dataset = HierarchicalPalsyDataset(
        args.db_path, args.fold, 'train', args.split_version,
        use_augmentation=True, use_wrinkle_heatmap=args.use_wrinkle_heatmap
    )
    val_dataset = HierarchicalPalsyDataset(
        args.db_path, args.fold, 'val', args.split_version,
        use_augmentation=False, use_wrinkle_heatmap=args.use_wrinkle_heatmap
    )

    print(f"Train: {len(train_dataset)} examinations")
    print(f"Val: {len(val_dataset)} examinations")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_hierarchical, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_hierarchical, num_workers=0
    )

    # 创建模型
    model = HGFANet().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    loss_fn = HierarchicalMultiTaskLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    best_metric = 0
    checkpoint_dir = Path(args.checkpoint_dir) / f"fold_{args.fold}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"  has_palsy acc: {val_metrics['has_palsy_acc']:.4f}")
        print(f"  palsy_side acc: {val_metrics['palsy_side_acc']:.4f}")
        print(f"  hb_grade acc: {val_metrics['hb_grade_acc']:.4f}")
        print(f"  sunnybrook MAE: {val_metrics['sunnybrook_mae']:.2f}")

        # 保存最佳模型 (以HB准确率为主要指标)
        current_metric = val_metrics['hb_grade_acc']
        if current_metric > best_metric:
            best_metric = current_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
            }, checkpoint_dir / 'best.pth')
            print(f"  ✓ Saved best model (HB acc: {best_metric:.4f})")

    print(f"\nTraining complete! Best HB acc: {best_metric:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default='facialPalsy.db')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--split_version', type=str, default='v1.0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--use_wrinkle_heatmap', action='store_true')

    args = parser.parse_args()
    main(args)