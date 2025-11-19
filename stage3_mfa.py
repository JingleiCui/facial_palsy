"""
Stage 3: MFA (Multi-modal Fusion Attention)
多模态融合注意力模块 + SQLite 写入逻辑

功能:
1. 融合三种模态:
   - geo_refined_features   (B, 256)
   - visual_guided_features (B, 256)
   - visual_features        (B, 1280) 作为视觉全局输入
2. 通过 Transformer + token gating 得到动作级 512 维特征
3. 写入:
   - video_features.visual_global_features    (256维)
   - video_features.fused_action_features     (512维)

使用:
    python stage3_mfa.py [facialPalsy.db]
"""

import sys
import sqlite3
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn


# =========================
# 1. MFA 模块定义
# =========================

class MFA(nn.Module):
    """
    Multi-modal Fusion Attention

    输入:
        geo_refined    : (B, geo_dim=256)
        visual_guided  : (B, vg_dim=256)
        visual_global  : (B, vg_input_dim=1280)

    输出:
        fused_action   : (B, output_dim=512)
        (可选)details  : {
            'tokens_encoded': (B, 3, F),
            'modality_weights': (B, 3),
            'geo_token': (B, F),
            'vg_token': (B, F),
            'vglobal_token': (B, F),
        }
    """

    def __init__(
        self,
        geo_dim: int = 256,
        visual_guided_dim: int = 256,
        visual_global_dim: int = 1280,
        feature_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.geo_dim = geo_dim
        self.visual_guided_dim = visual_guided_dim
        self.visual_global_dim = visual_global_dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        # 1) 各模态 → 统一特征空间
        self.geo_proj = nn.Sequential(
            nn.LayerNorm(geo_dim),
            nn.Linear(geo_dim, feature_dim),
            nn.GELU(),
        )
        self.vg_proj = nn.Sequential(
            nn.LayerNorm(visual_guided_dim),
            nn.Linear(visual_guided_dim, feature_dim),
            nn.GELU(),
        )
        self.vglobal_proj = nn.Sequential(
            nn.LayerNorm(visual_global_dim),
            nn.Linear(visual_global_dim, feature_dim),
            nn.GELU(),
        )

        # 2) 三token TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3) token gating: 每个 token 输出 1 个 logit
        self.token_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 1),
        )

        # 4) 输出: 融合 token + 几何 token 残差 → 512 维
        self.out_proj = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, output_dim),
        )

    def forward(
        self,
        geo_refined: torch.Tensor,
        visual_guided: torch.Tensor,
        visual_global: torch.Tensor,
        return_details: bool = False,
    ):
        # 各模态投影
        geo_tok = self.geo_proj(geo_refined)           # (B, F)
        vg_tok = self.vg_proj(visual_guided)           # (B, F)
        vglob_tok = self.vglobal_proj(visual_global)   # (B, F)

        # 拼成 token 序列: [geo, vg, vglob]
        tokens = torch.stack([geo_tok, vg_tok, vglob_tok], dim=1)   # (B, 3, F)

        # Transformer 编码
        tokens_enc = self.encoder(tokens)                          # (B, 3, F)

        # token gating
        gate_logits = self.token_gate(tokens_enc).squeeze(-1)      # (B, 3)
        modality_weights = torch.softmax(gate_logits, dim=-1)      # (B, 3)

        weights = modality_weights.unsqueeze(-1)                    # (B, 3, 1)
        fused_token = (tokens_enc * weights).sum(dim=1)            # (B, F)

        # 几何残差拼接
        out_input = torch.cat([fused_token, geo_tok], dim=-1)      # (B, 2F)
        fused_action = self.out_proj(out_input)                    # (B, output_dim)

        if not return_details:
            return fused_action

        details = {
            "tokens_encoded": tokens_enc,
            "modality_weights": modality_weights,
            "geo_token": geo_tok,
            "vg_token": vg_tok,
            "vglobal_token": vglob_tok,
        }
        return fused_action, details


# =========================
# 2. SQLite 批处理 Runner
# =========================

class Stage3MFARunner:
    def __init__(
        self,
        db_path: str,
        batch_size: int = 128,
        device: Optional[str] = None,
        print_weight_stats: bool = True,
    ):
        self.db_path = db_path
        self.batch_size = batch_size
        self.print_weight_stats = print_weight_stats

        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        print(f"[Stage3] 使用设备: {self.device}")

        self.model = MFA(
            geo_dim=256,
            visual_guided_dim=256,
            visual_global_dim=1280,
            feature_dim=256,
            num_heads=4,
            num_layers=2,
            output_dim=512,
        ).to(self.device)
        self.model.eval()

    def _fetch_pending_rows(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute(
            """
            SELECT video_id,
                   geo_refined_features,
                   visual_guided_features,
                   visual_features
            FROM video_features
            WHERE geo_refined_features IS NOT NULL
              AND visual_guided_features IS NOT NULL
              AND visual_features IS NOT NULL
              AND fused_action_features IS NULL
            """
        )
        return cur.fetchall()

    def _run_batch(
        self,
        conn: sqlite3.Connection,
        batch: List[Tuple[int, bytes, bytes, bytes]],
    ):
        cur = conn.cursor()

        video_ids = []
        geo_list = []
        vg_list = []
        vglob_input_list = []

        for vid, g_blob, vg_blob, v_blob in batch:
            video_ids.append(vid)

            g = np.frombuffer(g_blob or b"", dtype=np.float32)
            vg = np.frombuffer(vg_blob or b"", dtype=np.float32)
            v = np.frombuffer(v_blob or b"", dtype=np.float32)

            geo_dim = 256
            vg_dim = 256
            vglob_dim = 1280

            g_fixed = np.zeros(geo_dim, dtype=np.float32)
            if g.size > 0:
                g_fixed[: min(geo_dim, g.size)] = g[:geo_dim]

            vg_fixed = np.zeros(vg_dim, dtype=np.float32)
            if vg.size > 0:
                vg_fixed[: min(vg_dim, vg.size)] = vg[:vg_dim]

            vglob_fixed = np.zeros(vglob_dim, dtype=np.float32)
            if v.size > 0:
                vglob_fixed[: min(vglob_dim, v.size)] = v[:vglob_dim]

            geo_list.append(g_fixed)
            vg_list.append(vg_fixed)
            vglob_input_list.append(vglob_fixed)

        geo_batch = torch.from_numpy(np.stack(geo_list, axis=0)).to(self.device)
        vg_batch = torch.from_numpy(np.stack(vg_list, axis=0)).to(self.device)
        vglob_batch = torch.from_numpy(np.stack(vglob_input_list, axis=0)).to(self.device)

        with torch.no_grad():
            fused_action, details = self.model(
                geo_batch, vg_batch, vglob_batch, return_details=True
            )

        fused_np = fused_action.detach().cpu().numpy().astype(np.float32)
        vglob_token_np = details["vglobal_token"].detach().cpu().numpy().astype(np.float32)
        modality_weights_np = details["modality_weights"].detach().cpu().numpy()

        # 可选: 打印一次模态权重统计
        if self.print_weight_stats:
            mean_weights = modality_weights_np.mean(axis=0)
            print(
                "[Stage3] 模态权重均值 (geo, vg, vglob) = "
                f"{mean_weights[0]:.3f}, {mean_weights[1]:.3f}, {mean_weights[2]:.3f}"
            )
            self.print_weight_stats = False

        # 写回 DB
        for vid, fa, vglo in zip(video_ids, fused_np, vglob_token_np):
            cur.execute(
                """
                UPDATE video_features
                SET fused_action_features = ?,
                    visual_global_features = ?,
                    processed_at = CURRENT_TIMESTAMP
                WHERE video_id = ?
                """,
                (fa.tobytes(), vglo.tobytes(), vid),
            )

        conn.commit()

    def run(self):
        conn = sqlite3.connect(self.db_path)
        rows = self._fetch_pending_rows(conn)

        if not rows:
            print("[Stage3] 没有需要处理的记录 (fused_action_features 已全部存在)")
            conn.close()
            return

        print(f"[Stage3] 共有 {len(rows)} 条记录待处理")

        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            self._run_batch(conn, batch)
            print(
                f"[Stage3] 已处理 {min(start + self.batch_size, len(rows))}/{len(rows)} 条"
            )

        conn.close()
        print("[Stage3] 全部完成 ✅")


# =========================
# 3. main
# =========================

def main():
    db_path = "facialPalsy.db"
    runner = Stage3MFARunner(db_path=db_path, batch_size=128, device=None)
    runner.run()


if __name__ == "__main__":
    main()
