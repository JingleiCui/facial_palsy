"""
Stage 2: GQCA (Geometry-guided Query Cross-Attention)
几何引导的查询交叉注意力模块 + SQLite 写入逻辑

功能:
1. 使用 Stage1 输出的 geo_refined 作为 Query
2. 使用 visual_features (1280维) 生成 7×7=49 个伪空间 token
3. 通过 FiLM 调制 + Multi-Head Cross-Attention 得到几何引导的视觉特征
4. 写入 video_features.visual_guided_features 字段

使用:
    python stage2_gqca.py [facialPalsy.db]
"""

import sys
import sqlite3
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn


# =========================
# 1. GQCA 模块定义
# =========================

class GQCA(nn.Module):
    """
    几何引导的 Cross-Attention 模块 (1D 视觉向量 → 伪 7×7 空间 token)

    输入:
        geo_refined    : (B, geo_dim)
        visual_vec     : (B, visual_dim=1280)

    输出:
        visual_guided  : (B, out_dim=256)
        (可选)details  : {
            'attention_map': (B, 7, 7),
            'q_final': (B, d_model),
            'x_pooled': (B, d_model),
        }
    """

    def __init__(
        self,
        geo_dim: int = 256,
        visual_dim: int = 1280,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        num_tokens: int = 49,  # 7x7
        out_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.geo_dim = geo_dim
        self.visual_dim = visual_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.out_dim = out_dim

        # 几何 → Query token
        self.geo_proj = nn.Sequential(
            nn.LayerNorm(geo_dim),
            nn.Linear(geo_dim, d_model),
            nn.GELU(),
        )

        # 视觉全局向量 → 基础 token 表示
        self.visual_base_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, d_model),
            nn.GELU(),
        )

        # 位置编码: 49 个伪空间位置
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_tokens, d_model) * 0.02
        )

        # FiLM 调制: 用几何特征生成 γ / β
        self.film_gamma = nn.Linear(d_model, d_model)
        self.film_beta = nn.Linear(d_model, d_model)

        # 多层 Cross-Attention
        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_q_layers = nn.ModuleList()
        self.norm_kv_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                )
            )
            self.norm_q_layers.append(nn.LayerNorm(d_model))
            self.norm_kv_layers.append(nn.LayerNorm(d_model))

        # 输出聚合: [q_final, x_pooled] → out_dim
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, out_dim),
        )

    def forward(
        self,
        geo_refined: torch.Tensor,
        visual_vec: torch.Tensor,
        return_details: bool = False,
    ):
        """
        geo_refined : (B, geo_dim)
        visual_vec  : (B, visual_dim)
        """
        B = geo_refined.size(0)

        # 1) 编码几何 query
        q = self.geo_proj(geo_refined)           # (B, D)
        q = q.unsqueeze(1)                       # (B, 1, D)

        # 2) 视觉向量 → 基础 token
        base = self.visual_base_proj(visual_vec) # (B, D)
        x = base.unsqueeze(1).expand(-1, self.num_tokens, -1)  # (B, L, D)
        x = x + self.pos_embed                   # 加上伪空间的位置编码

        # 3) FiLM 调制
        geo_for_film = q.squeeze(1)              # (B, D)
        gamma = self.film_gamma(geo_for_film).unsqueeze(1)  # (B, 1, D)
        beta = self.film_beta(geo_for_film).unsqueeze(1)    # (B, 1, D)
        x = (1.0 + gamma) * x + beta

        attn_map = None

        # 4) 多层 Cross-Attention: Q=几何, K/V=视觉 token
        for layer_idx in range(self.num_layers):
            attn = self.attn_layers[layer_idx]
            ffn = self.ffn_layers[layer_idx]
            norm_q = self.norm_q_layers[layer_idx]
            norm_kv = self.norm_kv_layers[layer_idx]

            q_norm = norm_q(q)
            x_norm = norm_kv(x)

            attn_out, attn_weights = attn(
                q_norm,  # (B, 1, D)
                x_norm,  # (B, L, D)
                x_norm,
                need_weights=True,
                average_attn_weights=True,
            )
            # attn_out: (B, 1, D), attn_weights: (B, 1, L)

            q = q + attn_out
            q = q + ffn(q)

            attn_map = attn_weights  # 只保留最后一层

        q_final = q.squeeze(1)           # (B, D)
        x_pooled = x.mean(dim=1)         # (B, D)

        out_concat = torch.cat([q_final, x_pooled], dim=-1)
        visual_guided = self.out_proj(out_concat)   # (B, out_dim)

        if not return_details:
            return visual_guided

        if attn_map is not None:
            attn_flat = attn_map.squeeze(1)         # (B, L)
            if self.num_tokens == 49:
                attn_2d = attn_flat.view(B, 7, 7)
            else:
                attn_2d = attn_flat
        else:
            attn_2d = None

        details = {
            "attention_map": attn_2d,
            "q_final": q_final,
            "x_pooled": x_pooled,
        }
        return visual_guided, details


# =========================
# 2. SQLite 批处理 Runner
# =========================

class Stage2GQCARunner:
    def __init__(
        self,
        db_path: str,
        batch_size: int = 128,
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

        print(f"[Stage2] 使用设备: {self.device}")

        self.model = GQCA(
            geo_dim=256,
            visual_dim=1280,
            d_model=256,
            num_heads=8,
            num_layers=2,
            num_tokens=49,
            out_dim=256,
        ).to(self.device)
        self.model.eval()

    def _fetch_pending_rows(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute(
            """
            SELECT video_id,
                   geo_refined_features,
                   visual_features
            FROM video_features
            WHERE geo_refined_features IS NOT NULL
              AND visual_features IS NOT NULL
              AND visual_guided_features IS NULL
            """
        )
        return cur.fetchall()

    def _run_batch(
        self,
        conn: sqlite3.Connection,
        batch: List[Tuple[int, bytes, bytes]],
    ):
        cur = conn.cursor()

        video_ids = []
        geo_list = []
        vis_list = []

        for vid, g_blob, v_blob in batch:
            video_ids.append(vid)

            g = np.frombuffer(g_blob or b"", dtype=np.float32)
            v = np.frombuffer(v_blob or b"", dtype=np.float32)

            # 维度容错: 目标 geo=256, visual=1280
            geo_dim = 256
            vis_dim = 1280

            g_fixed = np.zeros(geo_dim, dtype=np.float32)
            if g.size > 0:
                g_fixed[: min(geo_dim, g.size)] = g[:geo_dim]

            v_fixed = np.zeros(vis_dim, dtype=np.float32)
            if v.size > 0:
                v_fixed[: min(vis_dim, v.size)] = v[:vis_dim]

            geo_list.append(g_fixed)
            vis_list.append(v_fixed)

        geo_batch = torch.from_numpy(np.stack(geo_list, axis=0)).to(self.device)
        vis_batch = torch.from_numpy(np.stack(vis_list, axis=0)).to(self.device)

        with torch.no_grad():
            visual_guided = self.model(geo_batch, vis_batch, return_details=False)

        vg_np = visual_guided.detach().cpu().numpy().astype(np.float32)

        for vid, feat in zip(video_ids, vg_np):
            cur.execute(
                """
                UPDATE video_features
                SET visual_guided_features = ?,
                    processed_at = CURRENT_TIMESTAMP
                WHERE video_id = ?
                """,
                (feat.tobytes(), vid),
            )

        conn.commit()

    def run(self):
        conn = sqlite3.connect(self.db_path)
        rows = self._fetch_pending_rows(conn)

        if not rows:
            print("[Stage2] 没有需要处理的记录 (visual_guided_features 已全部存在)")
            conn.close()
            return

        print(f"[Stage2] 共有 {len(rows)} 条记录待处理")

        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            self._run_batch(conn, batch)
            print(
                f"[Stage2] 已处理 {min(start + self.batch_size, len(rows))}/{len(rows)} 条"
            )

        conn.close()
        print("[Stage2] 全部完成 ✅")


# =========================
# 3. main
# =========================

def main():
    db_path = "facialPalsy.db"
    runner = Stage2GQCARunner(db_path=db_path, batch_size=128, device=None)
    runner.run()


if __name__ == "__main__":
    main()
