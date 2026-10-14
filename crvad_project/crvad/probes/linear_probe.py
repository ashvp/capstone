"""Linear probe: is the relation information PRESENT in the frozen features?

Zero-shot cosine (encoder_bakeoff.py) measures prompt sensitivity, not information
content. A frozen encoder can score near chance zero-shot and still contain a
perfectly linearly-decodable relation -- which is exactly the situation the temporal
adapter in crvad/model.py exists to exploit.

So: extract features once, cache them, then ask the question the zero-shot number
cannot answer.

    python -m crvad.probes.linear_probe --encoders clip xclip

Reports, per encoder:
  * zero-shot pairwise edit_acc   (from the cached features)
  * cross-validated 41-way template accuracy
  * the same on shuffled labels   (a permutation control -- a probe that cannot
    beat its own control is fitting noise, not signal)
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .ssv2_data import ANTONYM_PAIRS

CACHE = "runs/feats"


# --------------------------------------------------------------------------- features
def get_feats(encoder: str, data, dev: str, batch: int = 32) -> torch.Tensor:
    """Encoded clips (N,D), cached on disk. Encoding is the expensive part (~25 min on
    CPU for 264 clips); everything downstream is seconds, so cache aggressively."""
    os.makedirs(CACHE, exist_ok=True)
    tag = f"{encoder}_{data['clips'].shape[0]}x{data['clips'].shape[1]}"
    path = os.path.join(CACHE, f"{tag}.pt")
    if os.path.exists(path):
        print(f"  [{encoder}] cached {path}")
        return torch.load(path, weights_only=False)

    from .encoder_bakeoff import encode_clip

    print(f"  [{encoder}] encoding {len(data['clips'])} clips (slow, cached afterwards) ...")
    V = encode_clip(encoder, data["clips"], dev, batch).cpu()
    torch.save(V, path)
    print(f"  [{encoder}] saved {tuple(V.shape)} -> {path}")
    return V


# --------------------------------------------------------------------------- probes
def logreg(X: torch.Tensor, y: torch.Tensor, n_cls: int, epochs: int = 300,
           lr: float = 0.05, wd: float = 1e-3, seed: int = 0):
    """Fit multinomial logistic regression, full-batch AdamW. n is tiny here
    (hundreds), so anything fancier is overkill and slower.

    Returns (W, b) so the caller can score a DIFFERENT (held-out) set -- returning
    logits would silently tie the fit to the training fold."""
    torch.manual_seed(seed)
    W = torch.zeros(X.shape[1], n_cls, requires_grad=True)
    b = torch.zeros(n_cls, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(X @ W + b, y).backward()
        opt.step()
    return W.detach(), b.detach()


def cv_probe(X: torch.Tensor, y: torch.Tensor, n_cls: int, folds: int = 5,
             seed: int = 0, shuffle_y: bool = False,
             min_per_class: int = 0) -> Dict[str, float]:
    """Stratified k-fold, restricted to templates with at least `min_per_class` clips.

    `min_per_class` exists because the full 41-way problem is underdetermined: 264
    clips over 41 templates is a median of 6 each, and a 512-d linear head over ~217
    samples in 26 classes simply folds the noise. Restricting to the well-populated
    templates trades classes for samples-per-class, which is the axis that decides
    whether the probe means anything.

    Classes below the threshold are dropped entirely (not merely excluded from scoring)
    so that train and test agree on the label space.
    """
    g = torch.Generator().manual_seed(seed)
    if shuffle_y:                      # permutation control
        y = y[torch.randperm(len(y), generator=g)]

    counts = torch.bincount(y, minlength=n_cls)
    thresh = max(folds, min_per_class)
    keep = y[counts[y] >= thresh]
    sel = (y.unsqueeze(1) == keep.unsqueeze(0)).any(1)
    Xk, yk = X[sel], keep
    usable = torch.unique(yk)
    remap = -torch.ones(n_cls, dtype=torch.long)
    remap[usable] = torch.arange(len(usable))
    yk = remap[yk]

    # stratified fold assignment
    fold = torch.zeros(len(yk), dtype=torch.long)
    for c in usable.tolist():
        idx = (yk == c).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        fold[idx] = torch.arange(len(idx)) % folds

    correct, n = 0, 0
    for k in range(folds):
        tr, te = fold != k, fold == k
        if not te.any() or not tr.any():
            continue
        # Fit on the training folds, then SCORE the held-out fold. Applying the
        # training-fold logits to the test mask would be a shape error at best and
        # a silently mis-folded split at worst.
        W, b = logreg(Xk[tr], yk[tr], len(usable), seed=seed + k)
        logits = Xk[te] @ W + b
        correct += int((logits.argmax(1) == yk[te]).sum())
        n += int(te.sum())
    return {"acc": correct / max(1, n), "n": n, "n_classes": len(usable),
            "chance": 1.0 / max(1, len(usable))}


def zeroshot(V: torch.Tensor, U: torch.Tensor, captions, labels: torch.Tensor) -> float:
    gid = {c: i for i, c in enumerate(captions)}
    s = V @ U.T
    hits = tot = 0
    for a, b in ANTONYM_PAIRS:
        if a not in gid or b not in gid:
            continue
        ma, mb = labels == gid[a], labels == gid[b]
        if not ma.any() or not mb.any():
            continue
        hits += int((s[ma, gid[a]] > s[ma, gid[b]]).sum()) + int((s[mb, gid[b]] > s[mb, gid[a]]).sum())
        tot += int(ma.sum()) + int(mb.sum())
    return hits / max(1, tot)


def sweep(V: torch.Tensor, y: torch.Tensor, n_cls: int, folds: int,
          thresholds, seeds, n_perm: int) -> list:
    """For each min-per-class threshold, average the probe over several CV seeds and
    compare against the MEAN of several independent label permutations.

    A single shuffled control is one draw from a noisy null, and the earlier run read
    signal off exactly that kind of single draw. Using a mean over permutations gives
    the null a spread, so "beats its own control" means something.
    """
    rows = []
    for thr in thresholds:
        accs = [cv_probe(V, y, n_cls, folds, seed=s, min_per_class=thr)["acc"]
                for s in seeds]
        perms = [cv_probe(V, y, n_cls, folds, seed=s, shuffle_y=True,
                          min_per_class=thr)["acc"]
                 for s in range(n_perm)]
        ref = cv_probe(V, y, n_cls, folds, seed=seeds[0], min_per_class=thr)
        real = sum(accs) / len(accs)
        ctrl = sum(perms) / len(perms)
        spread = (max(perms) - min(perms)) / 2
        rows.append({"min_per_class": thr, "n": ref["n"], "n_classes": ref["n_classes"],
                     "chance": ref["chance"], "probe_acc": real, "probe_std": _std(accs),
                     "control_acc": ctrl, "control_spread": spread,
                     "gain": real - ctrl,
                     "beats_null": bool(real - ctrl > max(spread, 0.02))})
    return rows


def _std(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/ssv2_pairs.pt")
    ap.add_argument("--encoders", nargs="*", default=["clip", "xclip"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--thresholds", nargs="*", type=int, default=[0, 5, 8, 10, 12])
    ap.add_argument("--seeds", type=int, default=5, help="CV seeds averaged per threshold")
    ap.add_argument("--perms", type=int, default=8, help="label permutations for the null")
    ap.add_argument("--out", default="runs/gate2_linear_probe_sweep.json")
    a = ap.parse_args()

    data = torch.load(a.data, weights_only=False)
    clips, labels, captions = data["clips"], data["label"], data["captions"]
    counts = torch.bincount(labels, minlength=len(captions))
    print(f"{len(clips)} clips, {len(captions)} templates, folds={a.folds}")
    print(f"clips per template: min {int(counts.min())} median "
          f"{int(counts.median())} max {int(counts.max())}")
    for thr in a.thresholds:
        k = int((counts >= max(a.folds, thr)).sum())
        n = int(counts[counts >= max(a.folds, thr)].sum())
        print(f"  >={max(a.folds, thr):2d} clips: {k:2d} templates, {n:3d} clips, "
              f"{n/max(1,k):.1f} per class")
    print()

    seeds = list(range(a.seeds))
    results: Dict[str, Dict] = {}
    for enc in a.encoders:
        print(f"===== {enc} " + "=" * (54 - len(enc)))
        try:
            V = get_feats(enc, data, a.device)
            from .encoder_bakeoff import encode_text

            U = encode_text(enc, list(captions), a.device).cpu()
        except Exception as e:
            print(f"  SKIPPED: {type(e).__name__}: {e}\n")
            continue

        n_cls = len(captions)
        zs = zeroshot(V, U, captions, labels)
        rows = sweep(V, labels, n_cls, a.folds, a.thresholds, seeds, a.perms)

        print(f"  zero-shot pairwise edit_acc : {zs:.3f}   (chance 0.500)")
        print(f"  {'thr':>4} {'n':>5} {'cls':>4} {'chance':>7} {'probe':>7} {'+-':>6} "
              f"{'null':>7} {'+-':>6} {'gain':>7}  verdict")
        for r in rows:
            v = "SIGNAL" if r["beats_null"] else "-"
            print(f"  {r['min_per_class']:>4} {r['n']:>5} {r['n_classes']:>4} "
                  f"{r['chance']:>7.3f} {r['probe_acc']:>7.3f} {r['probe_std']:>6.3f} "
                  f"{r['control_acc']:>7.3f} {r['control_spread']:>6.3f} "
                  f"{r['gain']:>+7.3f}  {v}")
        best = max(rows, key=lambda r: r["gain"])
        verdict = ("INFORMATION PRESENT -- linearly decodable, Stage 1 is viable"
                   if any(r["beats_null"] for r in rows) else
                   "NO USABLE SIGNAL -- no threshold beats the permutation null")
        print(f"  best gain {best['gain']:+.3f} at thr={best['min_per_class']} "
              f"-> {verdict}\n")
        results[enc] = {"zeroshot_edit_acc": zs, "sweep": rows, "verdict": verdict}

    if results:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()