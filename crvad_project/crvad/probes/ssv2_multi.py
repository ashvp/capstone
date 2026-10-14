"""Build the SSv2 minimal-intervention clip set across SEVERAL mp4 shards.

Why this exists: the gate-2 linear probe was run on 264 clips spread over 41 templates
(a median of 6 per template, one with a single clip). A 512-d linear head over that is
underdetermined, and a threshold sweep confirmed it -- restricting to the 5 best-populated
templates moved the gain over a permutation null from +0.001 to +0.002. So the remaining
confound is CLIPS PER CLASS, not classes, and the only way to remove it is more shards.

Each mp4 in the LeRobot mirror holds ~1000 of 217,838 episodes, so a shard contributes only
the ~264 clips whose template happens to be one of our 41 targets. Shards are additive.

    python -m crvad.probes.ssv2_multi --files 0 1 2 3 4 --out runs/ssv2_multi.pt

Merging is safe because build() derives its `matched` template list from the dataset-wide
meta/tasks.parquet, which every shard shares -- so label ids mean the same thing in every
per-file payload. This asserts that rather than assuming it.
"""
from __future__ import annotations

import argparse
import collections
import os
import subprocess
import sys
import tempfile

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "ssv2_data.py")


def build_shard(f: int, out: str, limit: int, chunk: int, frames: int, cache: str) -> str:
    cmd = [sys.executable, PROBE, "--file", str(f), "--out", out,
           "--limit-per-template", str(limit), "--chunk", str(chunk),
           "--max-frames", str(frames), "--cache", cache]
    print(f"\n{'=' * 70}\n$ {' '.join(cmd[1:])}\n{'=' * 70}", flush=True)
    r = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(HERE)))
    if r.returncode != 0:
        raise SystemExit(f"shard {f} failed with code {r.returncode}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", type=int, required=True)
    ap.add_argument("--out", default="runs/ssv2_multi.pt")
    ap.add_argument("--limit-per-template", type=int, default=24,
                    help="PER SHARD; the global cap is applied after merging")
    ap.add_argument("--global-limit", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--cache", default="hf_cache")
    ap.add_argument("--keep-shards", action="store_true",
                    help="keep the per-shard .pt files (they are large; off by default)")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="ssv2_shards_")
    clips, labels, caps, sources = [], [], None, []

    for f in a.files:
        p = build_shard(f, os.path.join(tmp, f"shard_{f:03d}.pt"),
                        a.limit_per_template, a.chunk, a.max_frames, a.cache)
        d = torch.load(p, weights_only=False)
        if caps is None:
            caps = list(d["captions"])
        elif list(d["captions"]) != caps:
            # Label ids are positional into `captions`. If the ordering differed between
            # shards, concatenating would silently mislabel every clip after the first.
            raise SystemExit(
                f"shard {f} template ordering differs from shard {a.files[0]}; "
                f"label ids are not comparable, refusing to merge")
        clips.append(d["clips"])
        labels.append(d["label"])
        sources.append(d.get("source", f"file-{f:03d}"))
        c = collections.Counter(d["label"].tolist())
        print(f"  -> shard {f}: {len(d['clips'])} clips, {len(c)} templates, "
              f"{d['clips'].shape[1]} frames")

    X = torch.cat(clips)
    y = torch.cat(labels)
    del clips

    # Global cap: --limit-per-template bounds each shard, so the union can overshoot.
    counts = collections.Counter(y.tolist())
    if max(counts.values()) > a.global_limit:
        keep, seen = [], collections.Counter()
        for i, g in enumerate(y.tolist()):
            if seen[g] < a.global_limit:
                keep.append(i)
                seen[g] += 1
        keep = torch.tensor(keep)
        X, y = X[keep], y[keep]
        print(f"\nglobal cap {a.global_limit}: kept {len(keep)}/{len(y)} clips")

    payload = dict(clips=X, label=y, captions=caps, source="+".join(sources))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(payload, a.out)

    c = collections.Counter(y.tolist())
    v = sorted(c.values())
    print(f"\nsaved {tuple(X.shape)} -> {a.out}  ({X.shape[0] * X.shape[1] * 3 * X.shape[2] * X.shape[3] / 1e9:.2f} GB)")
    print(f"templates used: {len(c)}/{len(caps)}")
    print(f"clips per template: min {v[0]} median {v[len(v)//2]} max {v[-1]}")
    for thr in (5, 8, 10, 20, 30, 40):
        k = sum(1 for n in v if n >= thr)
        n = sum(n for n in v if n >= thr)
        print(f"  >={thr:2d} clips: {k:2d} templates, {n:4d} clips, {n/max(1,k):5.1f} per class")
    print("\nper-template counts:")
    for gid, n in sorted(c.items()):
        print(f"  {caps[gid]:55s} {n}")

    if not a.keep_shards:
        for f in a.files:
            p = os.path.join(tmp, f"shard_{f:03d}.pt")
            if os.path.exists(p):
                os.remove(p)
        try:
            os.rmdir(tmp)
        except OSError:
            pass
        print(f"\nremoved shard temp dir {tmp}")


if __name__ == "__main__":
    main()
