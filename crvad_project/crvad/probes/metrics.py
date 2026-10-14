"""Bias-invariant metrics for minimal-pair probes (pure torch; no model dependencies).

Why this file exists
--------------------
The original pooled `edit_acc` pools the clips of template A (should prefer caption A) and the
clips of template B (should prefer caption B) and reports one accuracy. That is NOT immune to
class imbalance: a classifier that ignores the video and always prefers caption A scores
n_A / (n_A + n_B). In the first Gate-2 run, 12/22 (CLIP) and 13/22 (X-CLIP) pairs had a number
correct exactly equal to one template's clip count, which is that signature.

For a pair (A, B) define, per clip,   s = sim(clip, A) - sim(clip, B).
A-clips should have large s, B-clips small s. Then
  * AUC          = P(s_A > s_B)           invariant to any constant caption preference
  * balanced_acc = mean(0.5*(s_A>0) + 0.5*(s_B<0))   what the zero-shot decision rule really achieves
  * margin       = (mean s_A - mean s_B)/2   bias-cancelled separation
A video-blind classifier gets AUC 0.5 and balanced_acc 0.5 whatever the class sizes.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


def pair_auc(sA: torch.Tensor, sB: torch.Tensor) -> float:
    d = sA.reshape(-1, 1) - sB.reshape(1, -1)
    return ((d > 0).float().mean() + 0.5 * (d == 0).float().mean()).item()


def pair_report(sA: torch.Tensor, sB: torch.Tensor) -> Dict[str, float]:
    na, nb = len(sA), len(sB)
    raw_correct = int((sA > 0).sum()) + int((sB < 0).sum())
    return {
        "n_a": na, "n_b": nb,
        "auc": pair_auc(sA, sB),
        "acc_raw": raw_correct / (na + nb),
        "acc_balanced": 0.5 * ((sA > 0).float().mean().item() + (sB < 0).float().mean().item()),
        "majority_share": max(na, nb) / (na + nb),
        "pref_A": ((sA > 0).sum().item() + (sB > 0).sum().item()) / (na + nb),
        "margin": 0.5 * (sA.mean().item() - sB.mean().item()),
        # "video-blind" signature: the number correct equals one class size
        "constant_preference": raw_correct in (na, nb),
    }


def macro(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def macro_auc(pairs: List[dict]) -> float:
    return macro([pair_auc(p["sA"], p["sB"]) for p in pairs])


def bootstrap_macro_auc(pairs: List[dict], n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """Clip-level bootstrap. Clips are resampled WITHIN each template and the same resample is
    applied to every pair that uses that template (templates are reused across pairs), so shared
    clips are never treated as independent. Each pair dict needs: ta, tb, sA, sB (sA ordered by
    the clips of template ta, sB by the clips of tb, consistently across pairs)."""
    g = torch.Generator().manual_seed(seed)
    sizes: Dict[int, int] = {}
    for p in pairs:
        sizes[p["ta"]] = len(p["sA"]); sizes[p["tb"]] = len(p["sB"])
    vals = []
    for _ in range(n_boot):
        pos = {t: torch.randint(0, n, (n,), generator=g) for t, n in sizes.items()}
        vals.append(macro([pair_auc(p["sA"][pos[p["ta"]]], p["sB"][pos[p["tb"]]]) for p in pairs]))
    v = torch.tensor(vals)
    return v.quantile(0.025).item(), v.quantile(0.975).item()


def permutation_p(pairs: List[dict], n_perm: int = 2000, seed: int = 0) -> float:
    """One-sided P(macro AUC under within-pair label shuffling >= observed)."""
    g = torch.Generator().manual_seed(seed)
    obs = macro_auc(pairs)
    ge = 0
    for _ in range(n_perm):
        vals = []
        for p in pairs:
            s = torch.cat([p["sA"], p["sB"]])
            pm = torch.randperm(len(s), generator=g)
            vals.append(pair_auc(s[pm[: len(p["sA"])]], s[pm[len(p["sA"]):]]))
        ge += macro(vals) >= obs - 1e-12
    return (1 + ge) / (1 + n_perm)
