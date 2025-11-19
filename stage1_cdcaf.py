"""
Stage 1: CDCAF (Clinical-Driven Cross-Attention Fusion)
临床驱动的交叉注意力融合模块 + SQLite 写入逻辑

功能:
1. 融合可变维度的静态和动态几何特征
2. 输出统一的增强几何特征 geo_refined (256维)
3. 直接写入 video_features.geo_refined_features 字段

使用:
    python stage1_cdcaf.py [facialPalsy.db]

注意:
- 支持不同动作几何维度不同(static_dim/dynamic_dim)
- NeutralFace 动作 dynamic_dim 可以为 0, 自动处理
"""

import sys
import sqlite3
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


# =========================
# 1. CDCAF 模块定义
# =========================

class CDCAF(nn.Module):
    """
    Clinical-Driven Cross-Attention Fusion

    输入:
        static_geo     : (B, static_dim)
        dynamic_geo    : (B, dynamic_dim) 或 None
        clinical_prior : (B, clinical_dim) 或 None (目前 Runner 未使用, 预留扩展)

    输出:
        geo_refined    : (B, output_dim=256)
        (可选)details  : {
            's_enc': (B, d_model),
            'd_enc': (B, d_model) 或 None,
            'g_enc': (B, d_model),
            'gate_weights': (B, K),
            'x_encoded': (B, L, d_model)
        }
    """

    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        clinical_dim: int = 0,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert static_dim > 0, "static_dim 必须 > 0"

        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.clinical_dim = clinical_dim
        self.d_model = d_model
        self.output_dim = output_dim

        # 1) 各模态编码器: 映射到统一 d_model
        self.static_encoder = nn.Sequential(
            nn.LayerNorm(static_dim),
            nn.Linear(static_dim, d_model),
            nn.GELU(),
        )

        if dynamic_dim > 0:
            self.dynamic_encoder = nn.Sequential(
                nn.LayerNorm(dynamic_dim),
                nn.Linear(dynamic_dim, d_model),
                nn.GELU(),
            )
        else:
            self.dynamic_encoder = None

        if clinical_dim > 0:
            self.clinical_encoder = nn.Sequential(
                nn.LayerNorm(clinical_dim),
                nn.Linear(clinical_dim, d_model),
                nn.GELU(),
            )
        else:
            self.clinical_encoder = None

        # 全局 token: static(+dynamic) 原始拼接后投影
        self.global_proj = nn.Sequential(
            nn.Linear(static_dim + max(dynamic_dim, 0), d_model),
            nn.GELU(),
        )

        # 2) 多层 Transformer Encoder (token 间交互)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # 输入 (B, L, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3) 模态级门控: 静态 / 动态 / 全局 自适应加权
        gate_in_dim = d_model * (2 + (1 if dynamic_dim > 0 else 0))
        gate_out_dim = 2 + (1 if dynamic_dim > 0 else 0)

        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, gate_out_dim),
        )

        # 4) 输出投影到 256 维几何特征
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, output_dim),
        )

    def forward(
        self,
        static_geo: torch.Tensor,
        dynamic_geo: Optional[torch.Tensor] = None,
        clinical_prior: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        B = static_geo.size(0)
        device = static_geo.device
        dtype = static_geo.dtype

        tokens = []

        # 静态 token
        s_token = self.static_encoder(static_geo)            # (B, D)
        tokens.append(s_token.unsqueeze(1))                  # (B, 1, D)

        # 动态 token (可选)
        if self.dynamic_dim > 0:
            if dynamic_geo is None:
                dynamic_geo = torch.zeros(B, self.dynamic_dim, device=device, dtype=dtype)
            d_token = self.dynamic_encoder(dynamic_geo)     # (B, D)
            tokens.append(d_token.unsqueeze(1))
        else:
            d_token = None

        # 临床先验 token (目前未用, 预留)
        if self.clinical_dim > 0:
            if clinical_prior is None:
                clinical_prior = torch.zeros(B, self.clinical_dim, device=device, dtype=dtype)
            c_token = self.clinical_encoder(clinical_prior)
            tokens.append(c_token.unsqueeze(1))
        else:
            c_token = None

        # 全局 token
        if self.dynamic_dim > 0:
            if dynamic_geo is None:
                dynamic_geo = torch.zeros(B, self.dynamic_dim, device=device, dtype=dtype)
            raw_concat = torch.cat([static_geo, dynamic_geo], dim=-1)
        else:
            raw_concat = static_geo
        g_token = self.global_proj(raw_concat)              # (B, D)
        tokens.append(g_token.unsqueeze(1))

        # 拼接 token 序列 (B, L, D)
        x = torch.cat(tokens, dim=1)

        # 记录索引
        idx = 0
        idx_static = idx
        idx += 1

        idx_dynamic = idx if self.dynamic_dim > 0 else None
        if self.dynamic_dim > 0:
            idx += 1

        idx_clinical = idx if self.clinical_dim > 0 else None
        if self.clinical_dim > 0:
            idx += 1

        idx_global = idx

        # Transformer 编码
        x_enc = self.encoder(x)                             # (B, L, D)

        s_enc = x_enc[:, idx_static, :]
        g_enc = x_enc[:, idx_global, :]
        if idx_dynamic is not None:
            d_enc = x_enc[:, idx_dynamic, :]
        else:
            d_enc = torch.zeros_like(s_enc)

        # 模态 Gate
        if idx_dynamic is not None:
            gate_in = torch.cat([s_enc, d_enc, g_enc], dim=-1)   # (B, 3D)
        else:
            gate_in = torch.cat([s_enc, g_enc], dim=-1)          # (B, 2D)

        gate_logits = self.gate_mlp(gate_in)                     # (B, K)
        gate_weights = torch.softmax(gate_logits, dim=-1)        # (B, K)

        if idx_dynamic is not None:
            w_s = gate_weights[:, 0:1]
            w_d = gate_weights[:, 1:2]
            w_g = gate_weights[:, 2:3]
        else:
            w_s = gate_weights[:, 0:1]
            w_g = gate_weights[:, 1:2]
            w_d = torch.zeros_like(w_s)

        fused_token = w_s * s_enc + w_d * d_enc + w_g * g_enc    # (B, D)
        geo_refined = self.output_proj(fused_token)              # (B, output_dim)

        if not return_details:
            return geo_refined

        details = {
            "s_enc": s_enc,
            "d_enc": d_enc if idx_dynamic is not None else None,
            "g_enc": g_enc,
            "gate_weights": gate_weights,
            "x_encoded": x_enc,
        }
        return geo_refined, details


# =========================
# 2. SQLite 批处理 Runner
# =========================

class Stage1CDCAFRunner:
    def __init__(
        self,
        db_path: str,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.db_path = db_path
        self.batch_size = batch_size

        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        print(f"[Stage1] 使用设备: {self.device}")

    def _fetch_pending_rows(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute(
            """
            SELECT video_id,
                   static_features,
                   dynamic_features,
                   static_dim,
                   dynamic_dim
            FROM video_features
            WHERE geo_refined_features IS NULL
              AND static_features IS NOT NULL
            """
        )
        return cur.fetchall()

    def _group_by_dims(
        self,
        rows,
    ) -> Dict[Tuple[int, int], List[Tuple[int, bytes, bytes]]]:
        groups: Dict[Tuple[int, int], List[Tuple[int, bytes, bytes]]] = {}
        for video_id, s_blob, d_blob, s_dim, d_dim in rows:
            # 维度容错: 如果为空, 尝试从BLOB长度推断
            if s_dim is None and s_blob is not None:
                s_dim = len(s_blob) // 4
            if d_dim is None and d_blob is not None:
                d_dim = len(d_blob) // 4

            s_dim = s_dim or 0
            d_dim = d_dim or 0

            key = (s_dim, d_dim)
            groups.setdefault(key, []).append((video_id, s_blob, d_blob))
        return groups

    def _run_group(
        self,
        conn: sqlite3.Connection,
        key: Tuple[int, int],
        items: List[Tuple[int, bytes, bytes]],
    ):
        s_dim, d_dim = key
        if s_dim <= 0:
            print(f"[Stage1] 跳过 static_dim <= 0 的组: {key}")
            return

        print(f"[Stage1] 处理维度组 static_dim={s_dim}, dynamic_dim={d_dim}, 样本数={len(items)}")

        model = CDCAF(
            static_dim=s_dim,
            dynamic_dim=d_dim,
            clinical_dim=0,
            d_model=128,
            num_layers=2,
            num_heads=4,
            output_dim=256,
        ).to(self.device)
        model.eval()

        cur = conn.cursor()

        for start in range(0, len(items), self.batch_size):
            batch_items = items[start : start + self.batch_size]

            video_ids = []
            static_list = []
            dynamic_list = []

            for vid, s_blob, d_blob in batch_items:
                video_ids.append(vid)

                # 反序列化为 float32
                s = np.frombuffer(s_blob or b"", dtype=np.float32)
                d = np.frombuffer(d_blob or b"", dtype=np.float32)

                # 保证长度与 static_dim / dynamic_dim 一致 (超长截断, 不足补零)
                s_fixed = np.zeros(s_dim, dtype=np.float32)
                if s.size > 0:
                    s_fixed[: min(s_dim, s.size)] = s[:s_dim]

                if d_dim > 0:
                    d_fixed = np.zeros(d_dim, dtype=np.float32)
                    if d.size > 0:
                        d_fixed[: min(d_dim, d.size)] = d[:d_dim]
                else:
                    d_fixed = np.zeros(0, dtype=np.float32)

                static_list.append(s_fixed)
                dynamic_list.append(d_fixed)

            static_batch = torch.from_numpy(np.stack(static_list, axis=0)).to(self.device)

            if d_dim > 0:
                dynamic_batch = torch.from_numpy(np.stack(dynamic_list, axis=0)).to(self.device)
            else:
                dynamic_batch = None

            with torch.no_grad():
                geo_refined = model(static_batch, dynamic_batch, return_details=False)  # (B, 256)

            geo_np = geo_refined.detach().cpu().numpy().astype(np.float32)

            for vid, feat in zip(video_ids, geo_np):
                cur.execute(
                    """
                    UPDATE video_features
                    SET geo_refined_features = ?,
                        processed_at = CURRENT_TIMESTAMP
                    WHERE video_id = ?
                    """,
                    (feat.tobytes(), vid),
                )

            conn.commit()
            print(
                f"[Stage1] 维度组 ({s_dim}, {d_dim}) 已处理 "
                f"{min(start + self.batch_size, len(items))}/{len(items)}"
            )

    def run(self):
        conn = sqlite3.connect(self.db_path)
        rows = self._fetch_pending_rows(conn)

        if not rows:
            print("[Stage1] 没有需要处理的记录 (geo_refined_features 已全部存在)")
            conn.close()
            return

        groups = self._group_by_dims(rows)
        print(f"[Stage1] 共 {len(rows)} 条记录, 分为 {len(groups)} 个维度组")

        for key, items in groups.items():
            self._run_group(conn, key, items)

        conn.close()
        print("[Stage1] 全部完成 ✅")


# =========================
# 3. main
# =========================

def main():
    db_path = "facialPalsy.db"
    runner = Stage1CDCAFRunner(db_path=db_path, batch_size=64, device=None)
    runner.run()


if __name__ == "__main__":
    main()
