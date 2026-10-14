"""GATE 2b (v2) -- is the relation linearly decodable from frozen per-frame features?

v1 fit a 26-way template classifier on POOLED, UNSTANDARDISED features with 217 samples and
read "below chance" as "no signal". Below-chance accuracy is what leave-few-out CV produces on
pure noise, and the pooled input is order-blind, so directional pairs were unsolvable by design.

v2 asks the binary question the thesis actually needs, once per antonym pair, on features that
keep temporal order:

    mean           mean over frames                      (order-blind control)
    halves         [mean(first half), mean(second half)] (order-aware)
    delta          [mean, second - first]                (order-aware)
    halves_shuf    'halves' after shuffling frame order  (must collapse to ~'mean' for directional pairs)

Reported: macro AUC over pairs, a within-pair label-permutation null (mean, 95th pct, p-value).
Reading it:
  * halves >> mean on directional pairs  => the encoder keeps temporal relation information
  * halves ~ mean ~ 0.5                  => features do not carry it (try a video-native encoder)
  * mean > 0.5 on directional pairs      => appearance leak (object/scene differs between templates)

    python -m crvad.probes.linear_probe --encoders clip --data runs/ssv2_pairs.pt
Needs >= ~15 clips per template per pair to be meaningful; pairs with < 3 clips are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch

from .relation_probe import probe_macro, variants
from .ssv2_data import ANTONYM_PAIRS

CACHE = "runs/feats"


def get_frame_feats(encoder: str, data, dev: str, batch: int = 32) -> torch.Tensor:
    """(N,T,D) per-frame features, cached. The cache stores the labels it was built for and is
    discarded if they differ (a stale cache silently attached to new labels is a classic bug)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{encoder}_frames_{data['clips'].shape[0]}x{data['clips'].shape[1]}.pt")
    if os.path.exists(path):
        c = torch.load(path, weights_only=True)
        if torch.equal(c["label"], data["label"]):
            print(f"  [{encoder}] cached {path}")
            return c["F"].float()
        print(f"  [{encoder}] cache labels differ from data -> re-encoding")
    from .encoder_bakeoff import encode_frames

    print(f"  [{encoder}] encoding {len(data['clips'])} clips ...")
    Fr = encode_frames(encoder, data["clips"], dev, batch).cpu()
    torch.save({"F": Fr.half(), "label": data["label"]}, path)
    return Fr.float()


def pair_feature_sets(X: torch.Tensor, labels: torch.Tensor, captions) -> List[Dict[str, torch.Tensor]]:
    gid = {c: i for i, c in enumerate(captions)}
    out = []
    for a, b in ANTONYM_PAIRS:
        if a in gid and b in gid:
            out.append({"name": f"{a} vs {b}", "A": X[labels == gid[a]], "B": X[labels == gid[b]]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/ssv2_pairs.pt")
    ap.add_argument("--encoders", nargs="*", default=["clip", "xclip"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default="runs/gate2_relation_probe_v2.json")
    a = ap.parse_args()

    data = torch.load(a.data, weights_only=False)
    labels, captions = data["label"], data["captions"]
    print(f"{len(labels)} clips, {len(captions)} templates\n")
    results: Dict[str, Dict] = {}
    for enc in a.encoders:
        print(f"===== {enc} " + "=" * (54 - len(enc)))
        try:
            Fr = get_frame_feats(enc, data, a.device)
        except Exception as e:
            print(f"  SKIPPED: {type(e).__name__}: {e}\n")
            continue
        vs = variants(Fr)
        if Fr.shape[1] >= 2:                       # order-destroyed control
            g = torch.Generator().manual_seed(0)
            perm = torch.stack([torch.randperm(Fr.shape[1], generator=g) for _ in range(len(Fr))])
            vs["halves_shuf"] = variants(torch.gather(Fr, 1, perm[..., None].expand_as(Fr)))["halves"]
        results[enc] = {}
        for name, X in vs.items():
            r = probe_macro(pair_feature_sets(X, labels, captions), n_perm=a.n_perm)
            results[enc][name] = r
            if r.get("n_pairs", 0):
                print(f"  {name:12s} macro AUC {r['macro_auc']:.3f}  null {r['null_mean']:.3f} "
                      f"(p95 {r['null_p95']:.3f})  p={r['p_value']:.3f}  ({r['n_pairs']} pairs)")
            else:
                print(f"  {name:12s} no usable pairs (need >=3 clips per template)")
        print()
    if results:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
