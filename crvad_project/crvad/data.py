"""Feature datasets. Everything is precomputed to a single .pt dict of tensors.

Keys (N = number of clips/videos, T = segments, D = visual feature dim):
  feats        (N,T,D)  float16/32  frozen-encoder segment features          [required]
  slots        (N,4)    long        event slot indices (-1 = unknown)         [stage1: required]
  video_label  (N,)     long        weak video-level label 0/1                [stage2/eval]
  conf         (N,)     float       pseudo-label confidence in [0,1]          [stage2]
  nuis_feats   (N,T,D)  optional    features of a nuisance-augmented view     [invariance / stability]
  seg_labels   (N,T)    optional    segment-level GT (eval only)
  heldout      (N,)     bool        event uses a held-out (action,object) combo
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import Dataset


def load_pt(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def subset(d: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    n = len(d["feats"])
    return {k: (v[idx] if torch.is_tensor(v) and len(v) == n else v) for k, v in d.items()}


def split_train_val(d: Dict[str, torch.Tensor], val_frac: float, seed: int):
    """Random split of the *training* source into train/val (never touches test benchmarks)."""
    n = len(d["feats"])
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(n * val_frac))
    return subset(d, perm[n_val:]), subset(d, perm[:n_val])


class FeatDataset(Dataset):
    def __init__(self, d: Dict[str, torch.Tensor]):
        self.d = d
        self.n = len(d["feats"])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        out = {}
        for k, v in self.d.items():
            if torch.is_tensor(v) and len(v) == self.n:
                x = v[i]
                out[k] = x.float() if x.dtype == torch.float16 else x
        return out
