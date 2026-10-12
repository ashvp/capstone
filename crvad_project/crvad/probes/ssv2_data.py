"""GATE 2a -- prepare a small SSv2 subset for the encoder bake-off.

Why this exists
---------------
Gate 1 proved the *text* side can separate single-slot edits (AUC 0.979). Gate 2 asks
the harder question: can the *video* side see them? Answering that on the project's
own 1728-hypothesis bank needs a hand-written template -> slot-tuple map for all 174
SSv2 templates. That is days of annotation and it is not needed for a go/no-go.

Instead we probe at the **template level**, using only the SSv2 templates that are
already genuine minimal interventions -- pairs that differ by one antonymous verb or
preposition, with no object placeholders involved:

    Opening something          /  Closing something
    Covering something with ..  /  Uncovering something
    Pouring something into ..  /  Pouring something out of ..
    Pretending to take ...      /  Taking something ...

For a clip from template A, is its embedding closer to A's caption than to its
antonym's? Chance is 0.50. This is `edit_acc` restricted to the pairs where the
single-factor edit is unambiguous, and it needs zero manual mapping.

Data source
-----------
`n3puiol/something_something_v2_lerobot` -- Apache-2.0 (the Qualcomm licence question
on the original SSv2 release is sidestepped), and chunked, so one ~130 MB mp4 gives
1000 clips instead of downloading 19.4 GB.

    python -m crvad.probes.ssv2_data --limit-per-template 20
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import torch

REPO = "n3puiol/something_something_v2_lerobot"

# Minimal-intervention template pairs, verified verbatim against the official
# 174-template list (HuggingFaceM4/something_something_v2 card). Never matched by
# id -- ids are only recorded for the write-up, the matcher works on caption text.
#
# Two families, both single-factor edits:
#   (a) antonymous verb/preposition -- "Opening" vs "Closing"
#   (b) "Pretending or failing to..." -- looks like the action, isn't performed.
#       This is the tightest test of the thesis: the ONLY difference is whether the
#       relation actually happens, with no change of object, scene or camera.
ANTONYM_PAIRS: List[Tuple[str, str]] = [
    # (a) antonymy
    ("Opening something", "Closing something"),
    ("Covering something with something", "Uncovering something"),
    ("Folding something", "Unfolding something"),
    ("Pouring something into something", "Pouring something out of something"),
    ("Moving something up", "Moving something down"),
    ("Moving something closer to something", "Moving something away from something"),
    ("Moving something towards the camera", "Moving something away from the camera"),
    ("Turning the camera upwards while filming something", "Turning the camera downwards while filming something"),
    ("Turning the camera right while filming something", "Turning the camera left while filming something"),
    # (b) pretending vs. actually performing -- the best available counterfactuals
    ("Taking something from somewhere", "Pretending to take something from somewhere"),
    ("Taking something out of something", "Pretending to take something out of something"),
    ("Closing something", "Pretending to close something without actually closing it"),
    ("Opening something", "Pretending to open something without actually opening it"),
    ("Wiping something off of something", "Pretending or failing to wipe something off of something"),
    ("Picking something up", "Pretending to pick something up"),
    ("Throwing something", "Pretending to throw something"),
    ("Turning something upside down", "Pretending to turn something upside down"),
    ("Pouring something out of something", "Pretending to pour something out of something, but something is empty"),
    ("Putting something behind something", "Pretending to put something behind something"),
    ("Putting something into something", "Pretending to put something into something"),
    ("Putting something next to something", "Pretending to put something next to something"),
    ("Putting something underneath something", "Pretending to put something underneath something"),
]

# Official SSv2 template ids for the targets above, for the write-up only.
TEMPLATE_IDS: Dict[str, int] = {
    "Opening something": 46, "Closing something": 5,
    "Covering something with something": 6, "Uncovering something": 171,
    "Folding something": 14, "Unfolding something": 172,
    "Pouring something into something": 59, "Pouring something out of something": 62,
    "Moving something up": 45, "Moving something down": 43,
    "Moving something closer to something": 42, "Moving something away from something": 40,
    "Moving something towards the camera": 44, "Moving something away from the camera": 41,
    "Turning the camera upwards while filming something": 168,
    "Turning the camera downwards while filming something": 165,
    "Turning the camera right while filming something": 167,
    "Turning the camera left while filming something": 166,
    "Taking something from somewhere": 147, "Taking something out of something": 148,
    "Pretending to take something from somewhere": 81,
    "Pretending to take something out of something": 82,
    "Pretending to close something without actually closing it": 66,
    "Pretending to open something without actually opening it": 67,
    "Pretending or failing to wipe something off of something": 63,
    "Picking something up": 47, "Pretending to pick something up": 68,
    "Throwing something": 151, "Pretending to throw something": 83,
    "Wiping something off of something": 173,
    "Turning something upside down": 164, "Pretending to turn something upside down": 84,
    "Pretending to pour something out of something, but something is empty": 70,
    "Putting something behind something": 104, "Pretending to put something behind something": 71,
    "Putting something into something": 106, "Pretending to put something into something": 72,
    "Putting something next to something": 107, "Pretending to put something next to something": 73,
    "Putting something underneath something": 118, "Pretending to put something underneath something": 76,
}

# Each template -> every template it is contrasted against. A template can have more
# than one partner ("Closing something" contrasts with both "Opening something" and
# "Pretending to close something without actually closing it"), so this is a list and
# the evaluation iterates PAIRS, never a single partner.
CONTRASTS: Dict[str, List[str]] = {}
for _a, _b in ANTONYM_PAIRS:
    CONTRASTS.setdefault(_a, [])
    CONTRASTS.setdefault(_b, [])
    if _b not in CONTRASTS[_a]:
        CONTRASTS[_a].append(_b)
    if _a not in CONTRASTS[_b]:
        CONTRASTS[_b].append(_a)


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def resolve_templates(captions: List[str]) -> Tuple[List[str], List[str], List[int]]:
    """Map target templates onto this dataset's actual caption strings.

    SSv2 keeps the literal word "something" in captions for templates with no real
    object noun, so these pairs appear verbatim. Anything that does not match is
    reported rather than silently dropped.
    """
    index: Dict[str, str] = {}
    for c in captions:
        index.setdefault(_norm(c), c)

    matched, missing = [], []
    for t in CONTRASTS:
        if _norm(t) in index:
            matched.append(t)
        else:
            missing.append(t)
    group_ids = {t: i for i, t in enumerate(sorted(matched))}
    return sorted(matched), missing, [group_ids[t] for t in sorted(matched)]


def download(chunk: int, file: int, cache: str) -> Tuple[str, str, str]:
    from huggingface_hub import hf_hub_download

    tasks = hf_hub_download(REPO, "meta/tasks.parquet", repo_type="dataset", cache_dir=cache)
    data = hf_hub_download(REPO, f"data/chunk-{chunk:03d}/file-{file:03d}.parquet",
                           repo_type="dataset", cache_dir=cache)
    video = hf_hub_download(REPO, f"videos/observation.images.image/chunk-{chunk:03d}/file-{file:03d}.mp4",
                            repo_type="dataset", cache_dir=cache)
    return tasks, data, video


def iter_frames(path: str):
    """Yield (index, rgb_uint8 HWC ndarray) sequentially.

    The LeRobot SSv2 mirror is AV1-encoded; PyAV handles it, OpenCV usually does not.
    """
    import av
    import numpy as np

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        i = 0
        for frame in container.decode(stream):
            yield i, frame.to_ndarray(format="rgb24")
            i += 1


def build(out: str, chunk: int = 0, file: int = 0, limit_per_template: int = 24,
          max_frames: int = 16, cache: str = "hf_cache", videos_per_file: int = 1000):
    import numpy as np
    import pandas as pd

    tasks_p, data_p, video_p = download(chunk, file, cache)
    print(f"tasks : {tasks_p}\ndata  : {data_p}\nvideo : {video_p}")

    tasks = pd.read_parquet(tasks_p)
    frames = pd.read_parquet(data_p)

    print(f"\ntasks.parquet  columns={list(tasks.columns)}  rows={len(tasks)}")
    print(f"  head:\n{tasks.head(3).to_string()}")
    print(f"\ndata parquet   columns={list(frames.columns)}  rows={len(frames)}")
    print(f"  head:\n{frames.head(5).to_string()}")

    # --- column discovery: this mirror's names are not guaranteed -------------
    def pick(cols, *keywords, exclude=()):
        for kw in keywords:
            for c in cols:
                lc = c.lower()
                if lc == kw and lc not in exclude:
                    return c
        for kw in keywords:
            for c in cols:
                lc = c.lower()
                if kw in lc and lc not in exclude:
                    return c
        return None

    task_col = pick(tasks.columns, "task", "text", "caption", "instruction", "description", "language")
    ep_col = pick(frames.columns, "episode_index", "episode", "episode_id")
    fidx_col = pick(frames.columns, "frame_index", "frame")
    tidx_col = pick(frames.columns, "task_index", "task_id", "label")
    if not all([task_col, ep_col, fidx_col, tidx_col]):
        raise SystemExit(f"could not resolve columns. tasks={list(tasks.columns)} frames={list(frames.columns)}")
    print(f"\nresolved: caption={task_col!r} episode={ep_col!r} frame={fidx_col!r} task={tidx_col!r}")

    frames = frames[[ep_col, fidx_col, tidx_col]].sort_values([ep_col, fidx_col])
    print(f"episodes in this parquet: {frames[ep_col].nunique()}   rows: {len(frames)}")

    # --- the parquet may cover the whole dataset while the mp4 covers only the
    # first `videos_per_file` episodes. Restrict explicitly rather than decoding
    # 9.8M frames looking for clips that are not in this file.
    all_eps = sorted(frames[ep_col].unique())
    eps_here = all_eps[:videos_per_file]
    print(f"episodes present in THIS mp4 (first {videos_per_file}): {len(eps_here)}")
    frames = frames[frames[ep_col].isin(eps_here)]

    captions = tasks[task_col].astype(str).tolist()
    print(f"distinct captions in tasks.parquet: {len(set(captions))}")
    print(f"  sample captions: {list(dict.fromkeys(captions))[:5]}")
    matched, missing, _ = resolve_templates(captions)
    print(f"\ntarget templates matched : {len(matched)}/{len(CONTRASTS)}")
    if missing:
        print(f"  NOT FOUND ({len(missing)}): " + "; ".join(missing))
    print("  " + "\n  ".join(matched))

    cap_of_task = {int(getattr(t, tidx_col)): _norm(str(getattr(t, task_col)))
               for t in tasks.itertuples()}
    target_norm = {m: _norm(m) for m in matched}
    gid_of_cap = {v: i for i, v in enumerate(target_norm.values())}
    cap_lookup = {v: m for m, v in target_norm.items()}

    # first pass: which episodes to keep, and their frame span
    lab_of_ep: Dict[int, int] = {}
    cap_of_ep: Dict[int, str] = {}
    spans: Dict[int, Tuple[int, int]] = {}
    counts: Dict[int, int] = {}
    for ep, g in frames.groupby(ep_col):
        ep = int(ep)
        cap = cap_of_task[int(g[tidx_col].iloc[0])]
        gid = gid_of_cap.get(cap)
        if gid is None or counts.get(gid, 0) >= limit_per_template:
            continue
        counts[gid] = counts.get(gid, 0) + 1
        lab_of_ep[ep] = gid
        cap_of_ep[ep] = cap_lookup[cap]
        spans[ep] = (int(g[fidx_col].min()), int(g[fidx_col].max()))

    if not lab_of_ep:
        raise SystemExit("no matching episodes -- check template strings against this dataset's captions")
    last_frame = max(spans[ep][1] for ep in lab_of_ep)
    print(f"\nkeeping {len(lab_of_ep)} episodes")
    print(f"need to decode frames 0..{last_frame}")

    # frame index -> owning episode. O(1) lookup, no scan per frame.
    frame_owner: Dict[int, int] = {}
    for ep, (lo, hi) in spans.items():
        for f in range(lo, hi + 1):
            frame_owner[f] = ep

    clips, labels, caps = [], [], []
    buf: List[torch.Tensor] = []
    buf_ep: Optional[int] = None

    def flush():
        if buf_ep is None or not buf:
            return
        arr = torch.from_numpy(np.stack(buf))                    # (F,H,W,3) uint8
        if arr.shape[0] > max_frames:                             # uniform subsample
            idx = torch.linspace(0, arr.shape[0] - 1, max_frames).round().long()
            arr = arr[idx]
        while arr.shape[0] < max_frames:                          # repeat-pad short clips
            arr = torch.cat([arr, arr[-1:]], 0)
        clips.append(arr.permute(0, 3, 1, 2).contiguous())       # (F,3,H,W)
        labels.append(lab_of_ep[buf_ep])
        caps.append(cap_of_ep[buf_ep])

    for i, rgb in iter_frames(video_p):
        if i > last_frame:
            break
        ep = frame_owner.get(i)
        if ep is None:
            continue
        if buf_ep is not None and ep != buf_ep:
            flush()
        buf_ep = ep
        buf.append(torch.from_numpy(rgb.copy()))
    flush()

    if not clips:
        raise SystemExit("decoded no clips -- check AV1 decoder availability")

    payload = dict(
        clips=torch.stack(clips),                    # (N, F, 3, H, W) uint8
        label=torch.tensor(labels),                  # (N,) template group id
        captions=matched,                            # list[str], group id -> template
        source=f"{REPO} chunk-{chunk:03d}/file-{file:03d}",
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(payload, out)
    print(f"\nsaved {payload['clips'].shape[0]} clips {tuple(payload['clips'].shape[1:])} -> {out}")

    import collections
    print("per-template counts:")
    for gid, n in sorted(collections.Counter(labels).items()):
        print(f"  {matched[gid]:55s} {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ssv2_pairs.pt")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--file", type=int, default=0)
    ap.add_argument("--limit-per-template", type=int, default=24)
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--cache", default="hf_cache")
    ap.add_argument("--videos-per-file", type=int, default=1000,
                    help="episodes present in each mp4; the data parquet is dataset-wide")
    a = ap.parse_args()
    build(a.out, a.chunk, a.file, a.limit_per_template, a.max_frames, a.cache, a.videos_per_file)


if __name__ == "__main__":
    main()