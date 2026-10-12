"""CRVAD model: temporal adapter + text projection + normality head over slot tuples.

Everything upstream (video encoder, text encoder) is frozen and precomputed; only
the small heads below are trained, which is what makes 2x A40 more than enough.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import slots as S


class CRVADModel(nn.Module):
    def __init__(self, d_vid: int, d_txt: int, d: int = 256, n_layers: int = 2, n_heads: int = 4,
                 max_T: int = 64, dropout: float = 0.1, tau: float = 0.07):
        super().__init__()
        self.in_proj = nn.Sequential(nn.LayerNorm(d_vid), nn.Linear(d_vid, d))
        self.pos = nn.Parameter(torch.zeros(1, max_T, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.v_out = nn.Linear(d, d)
        self.t_proj = nn.Sequential(nn.LayerNorm(d_txt), nn.Linear(d_txt, d), nn.GELU(), nn.Linear(d, d))
        # normality head: judges the *described* event (slot tuple), not the pixels
        self.slot_emb = nn.ModuleList([nn.Embedding(r, d) for r in S.RADIX])
        self.norm_head = nn.Sequential(nn.Linear(len(S.RADIX) * d, d), nn.GELU(), nn.Linear(d, 1))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / tau)))
        self.register_buffer("bank_slots", torch.tensor(S.all_events(), dtype=torch.long), persistent=False)

    # ---- encoders -----------------------------------------------------------------
    @property
    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def encode_video(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (B,T,Dv) -> normalised per-segment embeddings (B,T,d)."""
        T = feats.shape[1]
        x = self.in_proj(feats) + self.pos[:, :T]
        return F.normalize(self.v_out(self.temporal(x)), dim=-1)

    def encode_text(self, txt: torch.Tensor) -> torch.Tensor:
        """txt (N,Dt) -> normalised hypothesis embeddings (N,d)."""
        return F.normalize(self.t_proj(txt), dim=-1)

    def abnormal_logits(self) -> torch.Tensor:
        """Abnormality logit for every hypothesis in the bank, shape (N_EVENTS,)."""
        emb = torch.cat([self.slot_emb[i](self.bank_slots[:, i]) for i in range(len(S.RADIX))], dim=-1)
        return self.norm_head(emb).squeeze(-1)

    # ---- scoring ------------------------------------------------------------------
    def posterior_scores(self, seg: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Per-segment anomaly score = E_{h ~ p(h | segment)} [ p_abnormal(h) ].  (B,T) in [0,1]."""
        sim = self.scale * torch.einsum("btd,nd->btn", seg, text_emb)
        post = sim.float().softmax(-1)
        abn = torch.sigmoid(self.abnormal_logits().float())
        return post @ abn

    @staticmethod
    def video_score(seg_scores: torch.Tensor, k_frac: float = 0.125) -> torch.Tensor:
        k = max(1, int(round(seg_scores.shape[1] * k_frac)))
        return seg_scores.topk(k, dim=1).values.mean(1)
