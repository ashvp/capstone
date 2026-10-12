"""Losses and negative samplers.

Negative modes (the central ablation):
  random : fresh uniformly random hypotheses              (LAVIDA-style decoys)
  multi  : 2-4 slots changed at once                      (Alert-CLIP-style semantic hard negatives, approximated)
  single : exactly one slot changed (minimal intervention) (proposed)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import slots as S

_MULT = None


def flat_index(slot_tuples: torch.Tensor) -> torch.Tensor:
    """(...,4) slot indices -> (...) flat hypothesis index (matches slots.encode)."""
    mult = torch.tensor([S.RADIX[1] * S.RADIX[2] * S.RADIX[3], S.RADIX[2] * S.RADIX[3], S.RADIX[3], 1],
                        device=slot_tuples.device)
    return (slot_tuples * mult).sum(-1)


def sample_negatives(pos_slots: torch.Tensor, K: int, mode: str) -> torch.Tensor:
    """pos_slots (B,4) -> negative slot tuples (B,K,4)."""
    B, dev = pos_slots.shape[0], pos_slots.device
    radix = torch.tensor(S.RADIX, device=dev)
    out = []
    for _ in range(K):
        if mode == "single":
            j = torch.randint(0, len(S.RADIX), (B,), device=dev)
            n = radix[j]
            old = pos_slots.gather(1, j[:, None]).squeeze(1)
            shift = 1 + (torch.rand(B, device=dev) * (n - 1)).long()
            neg = pos_slots.clone()
            neg.scatter_(1, j[:, None], ((old + shift) % n)[:, None])
        elif mode == "multi":
            m = torch.randint(2, len(S.RADIX) + 1, (B,), device=dev)
            rank = torch.rand(B, len(S.RADIX), device=dev).argsort(1).argsort(1)
            change = rank < m[:, None]
            shift = 1 + (torch.rand(B, len(S.RADIX), device=dev) * (radix - 1)).long()
            neg = torch.where(change, (pos_slots + shift) % radix, pos_slots)
        elif mode == "random":
            neg = (torch.rand(B, len(S.RADIX), device=dev) * radix).long()
        else:
            raise ValueError(f"unknown negative mode {mode}")
        out.append(neg)
    return torch.stack(out, dim=1)


def candidate_nce(seg: torch.Tensor, text_emb: torch.Tensor, pos_idx: torch.Tensor, neg_idx: torch.Tensor,
                  scale: torch.Tensor, margin: float = 0.2, weights: torch.Tensor | None = None):
    """InfoNCE over {positive hypothesis} U {sampled negatives}, plus a hinge margin vs the hardest negative.

    Similarity of a clip to a hypothesis = max over segments of cosine (the event may occupy only part of the clip).
    Returns (nce, margin_loss, top1_acc).
    """
    cand = torch.cat([pos_idx[:, None], neg_idx], dim=1)               # (B,1+K)
    t = text_emb[cand]                                                 # (B,1+K,d)
    sims = torch.einsum("btd,bkd->btk", seg, t).max(1).values          # (B,1+K)
    dup = cand[:, 1:] == pos_idx[:, None]                              # a 'negative' that equals the positive
    logits = scale * sims
    logits = torch.cat([logits[:, :1], logits[:, 1:].masked_fill(dup, -1e4)], dim=1)
    ce = F.cross_entropy(logits.float(), torch.zeros(len(cand), dtype=torch.long, device=cand.device),
                         reduction="none")
    hardest = sims[:, 1:].masked_fill(dup, -2.0).max(1).values
    marg = F.relu(margin - (sims[:, 0] - hardest))
    if weights is None:
        weights = torch.ones_like(ce)
    w = weights / weights.sum().clamp(min=1e-6)
    acc = (logits.argmax(1) == 0).float().mean()
    return (ce * w).sum(), (marg * w).sum(), acc


def invariance_loss(seg_a: torch.Tensor, seg_b: torch.Tensor) -> torch.Tensor:
    """Per-segment embeddings of two nuisance views of the same clip should coincide."""
    return (1.0 - (seg_a * seg_b).sum(-1)).mean()


def mil_loss(seg_scores: torch.Tensor, y: torch.Tensor, k_frac: float = 0.125):
    """Weak-label MIL: top-k mean of segment scores vs the video-level label."""
    k = max(1, int(round(seg_scores.shape[1] * k_frac)))
    vs = seg_scores.topk(k, dim=1).values.mean(1).clamp(1e-5, 1 - 1e-5)
    return F.binary_cross_entropy(vs, y.float()), vs


def rule_labels(device) -> torch.Tensor:
    return torch.tensor([float(S.is_abnormal(e)) for e in S.all_events()], device=device)


def rule_loss(abn_logits: torch.Tensor) -> torch.Tensor:
    """Optional weak prior: BCE of the normality head against ABNORMAL_RULES (class-balanced)."""
    y = rule_labels(abn_logits.device)
    pos_w = (1 - y.mean()) / y.mean().clamp(min=1e-6)
    return F.binary_cross_entropy_with_logits(abn_logits, y, pos_weight=pos_w)
