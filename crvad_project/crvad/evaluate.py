"""Evaluation: standard VAD metrics + intervention / grounding / nuisance / compositional metrics.

Metrics
  video_auc / video_ap     : video-level detection from top-k segment score.
  seg_auc                  : segment-level AUC (proxy for frame level) when seg_labels exist.
  edit_acc (grounding)     : P[ sim(clip, true event) > sim(clip, single-slot edit) ], per slot and overall.
  flip_acc (intervention)  : on matched pairs, P[ score(abnormal clip) > score(normal clip) ].
  flip_acc_nuis            : same, computed on nuisance-augmented views (robustness of the flip).
  nuis_delta               : mean |score(x) - score(x_nuis)|  (lower = more stable under irrelevant change).
  heldout/seen splits      : every metric above reported separately for unseen vs seen (action,object) combos.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from . import slots as S
from .data import load_pt
from .losses import flat_index
from .model import CRVADModel


@torch.no_grad()
def _embed(model: CRVADModel, feats: torch.Tensor, bs: int = 128) -> torch.Tensor:
    dev = next(model.parameters()).device
    return torch.cat([model.encode_video(feats[i:i + bs].float().to(dev)) for i in range(0, len(feats), bs)])


@torch.no_grad()
def seg_scores(model, text_emb, feats, bs: int = 128) -> torch.Tensor:
    seg = _embed(model, feats, bs)
    return torch.cat([model.posterior_scores(seg[i:i + bs], text_emb) for i in range(0, len(seg), bs)]).cpu()


@torch.no_grad()
def edit_accuracy(model, text_emb, feats, slots: torch.Tensor, bs: int = 128):
    """Grounding under minimal interventions. Returns (overall, per_slot dict)."""
    seg = _embed(model, feats, bs)
    dev = seg.device
    sims = torch.cat([torch.einsum("btd,nd->bnt", seg[i:i + bs], text_emb).max(-1).values
                      for i in range(0, len(seg), bs)])                      # (N, N_EVENTS)
    pos = flat_index(slots.to(dev))
    edits = torch.tensor([[S.encode(e) for e in S.single_slot_edits(tuple(map(int, s)))] for s in slots.tolist()],
                         device=dev)                                         # (N, 22)
    ok = (sims.gather(1, pos[:, None]) > sims.gather(1, edits)).float()       # (N, 22)
    ids = torch.tensor(S.edit_slot_ids(), device=dev)
    per = {S.SLOT_NAMES[j]: ok[:, ids == j].mean().item() for j in range(len(S.RADIX))}
    return ok.mean().item(), per


def _detection(scores_v: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    y, s = y.numpy(), scores_v.numpy()
    if len(set(y.tolist())) < 2:
        return {}
    return {"video_auc": float(roc_auc_score(y, s)), "video_ap": float(average_precision_score(y, s))}


@torch.no_grad()
def evaluate(model: CRVADModel, text_bank: torch.Tensor, videos: Optional[dict] = None, pairs: Optional[dict] = None,
             k_frac: float = 0.125) -> Dict[str, dict]:
    model.eval()
    dev = next(model.parameters()).device
    text_emb = model.encode_text(text_bank.to(dev))
    res: Dict[str, dict] = {}

    if videos is not None:
        sc = seg_scores(model, text_emb, videos["feats"])
        vs = CRVADModel.video_score(sc, k_frac)
        hv = videos.get("heldout")
        groups = {"all": torch.ones(len(vs), dtype=torch.bool)}
        if hv is not None:
            groups.update(seen=~hv.bool(), heldout=hv.bool())
        for name, m in groups.items():
            r = _detection(vs[m], videos["video_label"][m])
            if "seg_labels" in videos and m.any():
                sl = videos["seg_labels"][m].flatten().numpy()
                if len(set(sl.tolist())) > 1:
                    r["seg_auc"] = float(roc_auc_score(sl, sc[m].flatten().numpy()))
            if "nuis_feats" in videos and m.any():
                scn = seg_scores(model, text_emb, videos["nuis_feats"][m])
                r["nuis_delta"] = float((CRVADModel.video_score(scn, k_frac) - vs[m]).abs().mean())
            res[f"videos/{name}"] = r

    if pairs is not None:
        s0 = CRVADModel.video_score(seg_scores(model, text_emb, pairs["feats0"]), k_frac)
        s1 = CRVADModel.video_score(seg_scores(model, text_emb, pairs["feats1"]), k_frac)
        n0 = CRVADModel.video_score(seg_scores(model, text_emb, pairs["nuis0"]), k_frac)
        n1 = CRVADModel.video_score(seg_scores(model, text_emb, pairs["nuis1"]), k_frac)
        held = pairs["heldout"].bool()
        for name, m in {"all": torch.ones_like(held), "seen": ~held, "heldout": held}.items():
            if not m.any():
                continue
            r = {"flip_acc": float((s1[m] > s0[m]).float().mean()),
                 "flip_acc_nuis": float((n1[m] > n0[m]).float().mean()),
                 "score_gap": float((s1[m] - s0[m]).mean())}
            feats = torch.cat([pairs["feats0"][m], pairs["feats1"][m]])
            slots = torch.cat([pairs["slots0"][m], pairs["slots1"][m]])
            overall, per = edit_accuracy(model, text_emb, feats, slots)
            r["edit_acc"] = overall
            r.update({f"edit_acc/{k}": v for k, v in per.items()})
            res[f"pairs/{name}"] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text-bank", required=True)
    ap.add_argument("--videos")
    ap.add_argument("--pairs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = CRVADModel(ck["d_vid"], ck["d_txt"], **ck["cfg"]["model"]).to(a.device)
    model.load_state_dict(ck["model"])
    out = evaluate(model, load_pt(a.text_bank), load_pt(a.videos) if a.videos else None,
                   load_pt(a.pairs) if a.pairs else None)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
