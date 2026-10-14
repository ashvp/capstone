"""Standardised ridge probes for minimal pairs, temporal-order aware.

Fixes versus the first linear_probe.py
  * features are z-scored with TRAINING-fold statistics (CLIP clip features have per-dim std
    ~0.009 and 0.955 mean pairwise cosine, the worst case for an unstandardised optimiser);
  * closed-form ridge with the penalty chosen by exact leave-one-out (no epochs/lr to tune);
  * the question is binary per antonym pair (is it A or B?), scored by AUC, not 26-way template
    accuracy (which is dominated by object/scene cues, not the relation);
  * features keep temporal order: 'mean' is order-blind BY CONSTRUCTION, so directional pairs
    (opening/closing, up/down, into/out of, covering/uncovering, folding/unfolding) cannot be
    solved from it. 'halves' and 'delta' expose order. A model that scores the same on
    'mean' and 'halves' for directional pairs is using appearance, not the relation.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from .metrics import macro, pair_auc


def variants(F: torch.Tensor) -> Dict[str, torch.Tensor]:
    """F: (N,T,D) per-frame/segment features -> {name: (N,d)} probe inputs."""
    out = {"mean": F.mean(1)}
    if F.shape[1] >= 2:
        h = F.shape[1] // 2
        a, b = F[:, :h].mean(1), F[:, h:].mean(1)
        out["halves"] = torch.cat([a, b], dim=1)
        out["delta"] = torch.cat([F.mean(1), b - a], dim=1)
    return out


def ridge_fit_predict(Xtr: torch.Tensor, ytr: torch.Tensor, Xte: torch.Tensor,
                      lam_scales: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0)) -> torch.Tensor:
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6)
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    yc = ytr - ym
    K = A @ A.T
    n = K.shape[0]
    eye = torch.eye(n, dtype=K.dtype)
    base = K.diag().mean()
    best, lam = None, lam_scales[0] * base
    for s in lam_scales:                                    # exact LOO for ridge: e_i / (1 - h_ii)
        H = K @ torch.linalg.inv(K + s * base * eye)
        err = (((yc - H @ yc) / (1 - H.diag()).clamp(min=1e-6)) ** 2).mean().item()
        if best is None or err < best:
            best, lam = err, s * base
    alpha = torch.linalg.solve(K + lam * eye, yc)
    return B @ (A.T @ alpha) + ym


def cv_pair_auc(XA: torch.Tensor, XB: torch.Tensor, folds: int = 5, repeats: int = 3, seed: int = 0,
                shuffle: bool = False) -> float:
    """Repeated stratified CV AUC of a ridge probe separating template A from B."""
    X = torch.cat([XA, XB]).double()
    y = torch.cat([torch.ones(len(XA)), -torch.ones(len(XB))]).double()
    g = torch.Generator().manual_seed(seed)
    if shuffle:
        y = y[torch.randperm(len(y), generator=g)]
    k = min(folds, int((y > 0).sum()), int((y < 0).sum()))
    if k < 2:
        return float("nan")
    aucs = []
    for _ in range(repeats):
        fold = torch.zeros(len(y), dtype=torch.long)
        for cls in (1.0, -1.0):
            idx = (y == cls).nonzero().flatten()
            idx = idx[torch.randperm(len(idx), generator=g)]
            fold[idx] = torch.arange(len(idx)) % k
        oof = torch.zeros(len(y), dtype=torch.double)
        for f in range(k):
            te = fold == f
            oof[te] = ridge_fit_predict(X[~te], y[~te], X[te])
        aucs.append(pair_auc(oof[y > 0], oof[y < 0]))
    return macro(aucs)


def probe_macro(pair_feats: List[Dict[str, torch.Tensor]], n_perm: int = 200, seed: int = 0) -> Dict[str, float]:
    """pair_feats: [{'A': (nA,d), 'B': (nB,d)}, ...]. Returns macro AUC, permutation mean and one-sided p.
    Pairs with fewer than 3 clips in either class are skipped (reported via n_pairs)."""
    use = [p for p in pair_feats if len(p["A"]) >= 3 and len(p["B"]) >= 3]
    if not use:
        return {"n_pairs": 0}
    obs = macro([cv_pair_auc(p["A"], p["B"], seed=seed) for p in use])
    null = []
    for i in range(n_perm):
        null.append(macro([cv_pair_auc(p["A"], p["B"], repeats=1, seed=seed + 1000 + i, shuffle=True) for p in use]))
    nt = torch.tensor(null)
    return {"n_pairs": len(use), "macro_auc": obs, "null_mean": nt.mean().item(),
            "null_p95": nt.quantile(0.95).item(), "p_value": (1 + int((nt >= obs).sum())) / (1 + n_perm)}
