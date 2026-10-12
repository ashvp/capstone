"""Stage 1 (relational pretraining) and Stage 2 (weak-label VAD adaptation) on precomputed features.

    python -m crvad.train --stage 1 --config configs/default.yaml
    python -m crvad.train --stage 2 --config configs/default.yaml

Model selection uses a validation split carved out of the *training* source (never a test benchmark).
Heads are tiny, so a single A40 trains one run; the 2nd GPU runs a different seed / negative-mode ablation.
(Pin with CUDA_VISIBLE_DEVICES=0 / 1.)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .data import FeatDataset, load_pt, split_train_val
from .evaluate import seg_scores
from .losses import (candidate_nce, flat_index, invariance_loss, mil_loss, rule_loss, sample_negatives)
from .model import CRVADModel

DEFAULTS: Dict[str, Any] = dict(
    text_bank="data/text_bank.pt",
    model=dict(d=256, n_layers=2, n_heads=4, max_T=64, dropout=0.1, tau=0.07),
    stage1=dict(train="data/stage1_train.pt", epochs=20, bs=128, lr=3e-4, wd=0.05, K=16, neg_mode="single",
                margin=0.2, lam_margin=0.5, lam_inv=0.5),
    stage2=dict(train="data/stage2_train.pt", init="runs/default/stage1/best.pt", epochs=20, bs=32, lr=1e-4, wd=0.05,
                K=16, neg_mode="single", margin=0.2, lam_nce=0.5, lam_margin=0.25, lam_inv=0.5, lam_rule=0.0,
                k_frac=0.125),
    seed=0, val_frac=0.1, out="runs/default", device="auto", amp=True, workers=4,
)


def merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def seed_all(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _to(batch, dev):
    return {k: v.to(dev) for k, v in batch.items()}


def loss_stage1(model, text_emb, b, c):
    seg = model.encode_video(b["feats"])
    pos = flat_index(b["slots"])
    neg = flat_index(sample_negatives(b["slots"], c["K"], c["neg_mode"]))
    nce, marg, acc = candidate_nce(seg, text_emb, pos, neg, model.scale, c["margin"])
    loss = nce + c["lam_margin"] * marg
    logs = dict(nce=nce.item(), marg=marg.item(), acc=acc.item())
    if "nuis_feats" in b and c["lam_inv"] > 0:
        inv = invariance_loss(seg, model.encode_video(b["nuis_feats"]))
        loss = loss + c["lam_inv"] * inv
        logs["inv"] = inv.item()
    return loss, logs


def loss_stage2(model, text_emb, b, c):
    seg = model.encode_video(b["feats"])
    scores = model.posterior_scores(seg, text_emb)
    mil, _ = mil_loss(scores, b["video_label"], c["k_frac"])
    loss, logs = mil, dict(mil=mil.item())
    valid = b["slots"][:, 0] >= 0
    if valid.any() and c["lam_nce"] > 0:
        sl = b["slots"][valid]
        pos = flat_index(sl)
        neg = flat_index(sample_negatives(sl, c["K"], c["neg_mode"]))
        nce, marg, acc = candidate_nce(seg[valid], text_emb, pos, neg, model.scale, c["margin"],
                                       weights=b["conf"][valid] if "conf" in b else None)
        loss = loss + c["lam_nce"] * nce + c["lam_margin"] * marg
        logs.update(nce=nce.item(), acc=acc.item())
    if "nuis_feats" in b and c["lam_inv"] > 0:
        inv = invariance_loss(seg, model.encode_video(b["nuis_feats"]))
        loss = loss + c["lam_inv"] * inv
        logs["inv"] = inv.item()
    if c["lam_rule"] > 0:
        loss = loss + c["lam_rule"] * rule_loss(model.abnormal_logits())
    return loss, logs


@torch.no_grad()
def validate(model, text_bank, val, stage, c, dev):
    model.eval()
    text_emb = model.encode_text(text_bank.to(dev))
    if stage == 1:
        tot, n = 0.0, 0
        for b in DataLoader(FeatDataset(val), batch_size=256):
            loss, _ = loss_stage1(model, text_emb, _to(b, dev), c)
            tot += loss.item() * len(b["feats"]); n += len(b["feats"])
        return -tot / n                                       # higher is better
    sc = seg_scores(model, text_emb, val["feats"])
    vs = CRVADModel.video_score(sc, c["k_frac"])
    return float(roc_auc_score(val["video_label"].numpy(), vs.numpy()))


def run(cfg: dict, stage: int) -> str:
    c = cfg[f"stage{stage}"]
    seed_all(cfg["seed"])
    dev = torch.device("cuda" if (cfg["device"] == "auto" and torch.cuda.is_available()) else
                       ("cpu" if cfg["device"] == "auto" else cfg["device"]))
    text_bank = load_pt(cfg["text_bank"]).float()
    data = load_pt(c["train"])
    tr, va = split_train_val(data, cfg["val_frac"], cfg["seed"])
    model = CRVADModel(data["feats"].shape[-1], text_bank.shape[-1], **cfg["model"]).to(dev)
    if stage == 2 and c.get("init") and os.path.exists(c["init"]):
        model.load_state_dict(torch.load(c["init"], map_location=dev, weights_only=False)["model"])
        print(f"[stage2] initialised from {c['init']}")
    loader = DataLoader(FeatDataset(tr), batch_size=c["bs"], shuffle=True, drop_last=True,
                        num_workers=cfg["workers"] if dev.type == "cuda" else 0)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["wd"])
    total = c["epochs"] * len(loader)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    use_amp = cfg["amp"] and dev.type == "cuda"
    out_dir = os.path.join(cfg["out"], f"stage{stage}")
    os.makedirs(out_dir, exist_ok=True)
    step_fn = loss_stage1 if stage == 1 else loss_stage2
    best, best_path = -1e9, os.path.join(out_dir, "best.pt")
    for ep in range(c["epochs"]):
        model.train()
        agg: Dict[str, float] = {}
        for b in loader:
            b = _to(b, dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                text_emb = model.encode_text(text_bank.to(dev))
                loss, logs = step_fn(model, text_emb, b, c)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v / len(loader)
        score = validate(model, text_bank, va, stage, c, dev)
        print(f"[stage{stage}] ep {ep + 1}/{c['epochs']} val={score:.4f} " + " ".join(f"{k}={v:.3f}" for k, v in agg.items()))
        if score > best:
            best = score
            torch.save(dict(model=model.state_dict(), cfg=cfg, d_vid=data["feats"].shape[-1], d_txt=text_bank.shape[-1],
                            val=score), best_path)
    json.dump(dict(best_val=best, cfg=cfg), open(os.path.join(out_dir, "summary.json"), "w"), indent=2, default=str)
    return best_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like stage1.neg_mode=random out=runs/x")
    a = ap.parse_args()
    cfg = merge(DEFAULTS, yaml.safe_load(open(a.config)) if a.config else {})
    for kv in a.set:
        k, v = kv.split("=", 1)
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(v)
    print(run(cfg, a.stage))


if __name__ == "__main__":
    main()
