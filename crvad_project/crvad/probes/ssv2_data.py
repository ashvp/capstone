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
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

# SSv2 captions and polars' DataFrame repr contain non-ASCII characters; the Windows
# console defaults to cp1252 and raises UnicodeEncodeError on print.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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

# The official 174 SSv2 templates, verbatim, with their official ids.
#
# Needed because this mirror substitutes object placeholders inconsistently:
#   "Turning something upside down"                        -> kept as-is
#   "Turning the camera downwards while filming something"  -> "...while filming guitar"
# so exact caption matching finds nothing. We instead compile every template into an
# anchored regex ("something" -> ".+?") and match captions against the FULL 174, then
# keep the LONGEST matching template. That resolves prefix collisions correctly:
#   "pouring milk into a cup until it overflows" must hit template 60, not 59,
#   because 60's regex is longer and therefore more specific.
OFFICIAL_TEMPLATES: List[str] = [
    "Approaching something with your camera", "Attaching something to something",
    "Bending something so that it deforms", "Bending something until it breaks",
    "Burying something in something", "Closing something",
    "Covering something with something", "Digging something out of something",
    "Dropping something behind something", "Dropping something in front of something",
    "Dropping something into something", "Dropping something next to something",
    "Dropping something onto something",
    "Failing to put something into something because something does not fit",
    "Folding something", "Hitting something with something", "Holding something",
    "Holding something behind something", "Holding something in front of something",
    "Holding something next to something", "Holding something over something",
    "Laying something on the table on its side, not upright",
    "Letting something roll along a flat surface", "Letting something roll down a slanted surface",
    "Letting something roll up a slanted surface, so it rolls back down",
    "Lifting a surface with something on it but not enough for it to slide down",
    "Lifting a surface with something on it until it starts sliding down",
    "Lifting something up completely without letting it drop down",
    "Lifting something up completely, then letting it drop down",
    "Lifting something with something on it",
    "Lifting up one end of something without letting it drop down",
    "Lifting up one end of something, then letting it drop down",
    "Moving away from something with your camera", "Moving part of something",
    "Moving something across a surface until it falls down",
    "Moving something across a surface without it falling down",
    "Moving something and something away from each other",
    "Moving something and something closer to each other",
    "Moving something and something so they collide with each other",
    "Moving something and something so they pass each other",
    "Moving something away from something", "Moving something away from the camera",
    "Moving something closer to something", "Moving something down",
    "Moving something towards the camera", "Moving something up", "Opening something",
    "Picking something up", "Piling something up", "Plugging something into something",
    "Plugging something into something but pulling it right out as you remove your hand",
    "Poking a hole into some substance", "Poking a hole into something soft",
    "Poking a stack of something so the stack collapses",
    "Poking a stack of something without the stack collapsing",
    "Poking something so it slightly moves",
    "Poking something so lightly that it doesn't or almost doesn't move",
    "Poking something so that it falls over", "Poking something so that it spins around",
    "Pouring something into something", "Pouring something into something until it overflows",
    "Pouring something onto something", "Pouring something out of something",
    "Pretending or failing to wipe something off of something",
    "Pretending or trying and failing to twist something",
    "Pretending to be tearing something that is not tearable",
    "Pretending to close something without actually closing it",
    "Pretending to open something without actually opening it",
    "Pretending to pick something up", "Pretending to poke something",
    "Pretending to pour something out of something, but something is empty",
    "Pretending to put something behind something",
    "Pretending to put something into something",
    "Pretending to put something next to something",
    "Pretending to put something on a surface",
    "Pretending to put something onto something",
    "Pretending to put something underneath something",
    "Pretending to scoop something up with something",
    "Pretending to spread air onto something", "Pretending to sprinkle air onto something",
    "Pretending to squeeze something", "Pretending to take something from somewhere",
    "Pretending to take something out of something", "Pretending to throw something",
    "Pretending to turn something upside down", "Pulling something from behind of something",
    "Pulling something from left to right", "Pulling something from right to left",
    "Pulling something onto something", "Pulling something out of something",
    "Pulling two ends of something but nothing happens",
    "Pulling two ends of something so that it gets stretched",
    "Pulling two ends of something so that it separates into two pieces",
    "Pushing something from left to right", "Pushing something from right to left",
    "Pushing something off of something", "Pushing something onto something",
    "Pushing something so it spins", "Pushing something so that it almost falls off but doesn't",
    "Pushing something so that it falls off the table", "Pushing something so that it slightly moves",
    "Pushing something with something", "Putting number of something onto something",
    "Putting something and something on the table", "Putting something behind something",
    "Putting something in front of something", "Putting something into something",
    "Putting something next to something",
    "Putting something on a flat surface without letting it roll",
    "Putting something on a surface",
    "Putting something on the edge of something so it is not supported and falls down",
    "Putting something onto a slanted surface but it doesn't glide down",
    "Putting something onto something",
    "Putting something onto something else that cannot support it so it falls down",
    "Putting something similar to other things that are already on the table",
    "Putting something that can't roll onto a slanted surface, so it slides down",
    "Putting something that can't roll onto a slanted surface, so it stays where it is",
    "Putting something that cannot actually stand upright upright on the table, so it falls on its side",
    "Putting something underneath something", "Putting something upright on the table",
    "Putting something, something and something on the table",
    "Removing something, revealing something behind", "Rolling something on a flat surface",
    "Scooping something up with something", "Showing a photo of something to the camera",
    "Showing something behind something", "Showing something next to something",
    "Showing something on top of something", "Showing something to the camera",
    "Showing that something is empty", "Showing that something is inside something",
    "Something being deflected from something",
    "Something colliding with something and both are being deflected",
    "Something colliding with something and both come to a halt",
    "Something falling like a feather or paper", "Something falling like a rock",
    "Spilling something behind something", "Spilling something next to something",
    "Spilling something onto something", "Spinning something so it continues spinning",
    "Spinning something that quickly stops spinning", "Spreading something onto something",
    "Sprinkling something onto something", "Squeezing something",
    "Stacking number of something", "Stuffing something into something",
    "Taking one of many similar things on the table", "Taking something from somewhere",
    "Taking something out of something", "Tearing something into two pieces",
    "Tearing something just a little bit", "Throwing something",
    "Throwing something against something", "Throwing something in the air and catching it",
    "Throwing something in the air and letting it fall", "Throwing something onto a surface",
    "Tilting something with something on it slightly so it doesn't fall down",
    "Tilting something with something on it until it falls off", "Tipping something over",
    "Tipping something with something in it over, so something in it falls out",
    "Touching (without moving) part of something",
    "Trying but failing to attach something to something because it doesn't stick",
    "Trying to bend something unbendable so nothing happens",
    "Trying to pour something into something, but missing so it spills next to it",
    "Turning something upside down", "Turning the camera downwards while filming something",
    "Turning the camera left while filming something",
    "Turning the camera right while filming something",
    "Turning the camera upwards while filming something",
    "Twisting (wringing) something wet until water comes out", "Twisting something",
    "Uncovering something", "Unfolding something", "Wiping something off of something",
]

TEMPLATE_IDS: Dict[str, int] = {t: i for i, t in enumerate(OFFICIAL_TEMPLATES)}

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
    return " ".join(str(s).strip().lower().split())


# SSv2 templates do not only use "something" as the object slot. Template 147 is
# "Taking something from somewhere", template 51 is "Poking a hole into some
# substance". Treating those as literal text makes their captions unmatchable, so all
# three forms are treated as wildcards.
#
# Known limitation: template 113 ("...onto something else that cannot support it...")
# drops "else" when its slots are substituted and is therefore not reliably
# recoverable. It is not used as a contrast target, so it does not affect the gate.
# NOTE: the group must be NON-capturing. re.split() returns captured groups as extra
# elements, which injects the literal word "thing" into every generated regex.
_PLACEHOLDER = re.compile(r"\b(?:something|somewhere|some\s+substance)\b")


def compile_template(template: str) -> re.Pattern:
    """'Taking something from somewhere' -> ^taking\\ .+?\\ from\\ .+?$"""
    parts = _PLACEHOLDER.split(_norm(template))
    return re.compile("^" + ".+?".join(re.escape(p) for p in parts) + "$")


class TemplateMatcher:
    """Maps a caption back to its SSv2 template, longest match wins.

    Longest-match is what makes this safe: several templates are prefixes of longer
    ones ("Pouring something into something" vs "...until it overflows"), and the
    longer regex is always the more specific one.

    A word index avoids running 174 regexes against each of ~124k captions. A caption
    can only match a template that shares at least one of its fixed (non-"something")
    words with it, so we only test those candidates.
    """

    def __init__(self, templates: List[str] = None):
        self.templates = list(templates if templates is not None else OFFICIAL_TEMPLATES)
        self.rx = [compile_template(t) for t in self.templates]
        self.idx: Dict[str, List[int]] = {}
        self.always: List[int] = []               # templates with no fixed words
        for i, t in enumerate(self.templates):
            fixed = [w for w in _PLACEHOLDER.sub(" ", _norm(t)).split() if len(w) > 2]
            if not fixed:
                self.always.append(i)
            for w in fixed:
                self.idx.setdefault(w, []).append(i)

    def candidates(self, caption: str) -> List[int]:
        cap = set(_norm(caption).split())
        out = set(self.always)
        for w in cap:
            if w in self.idx:
                out.update(self.idx[w])
        return list(out)

    def match(self, caption: str) -> Optional[int]:
        """Return the official template id, or None."""
        c = _norm(caption)
        best, best_len = None, -1
        for i in self.candidates(c):
            if self.rx[i].match(c) and len(self.templates[i]) > best_len:
                best, best_len = i, len(self.templates[i])
        return best

    def match_many(self, captions: List[str]) -> Dict[int, int]:
        """caption position -> template id."""
        return {i: tid for i, c in enumerate(captions) if (tid := self.match(c)) is not None}


def read_captions(tasks) -> List[str]:
    """Captions live in this mirror's serialised pandas INDEX; the column holds int ids.

    polars drops that index unless include_index=True, hence the caller. Prefer a
    caption-named column, then the index, then any all-string column. Raises rather
    than silently returning integers -- that silent failure is what produced an
    unexplained 0/41 target match earlier.
    """
    def all_str(col):
        vals = tasks[col].to_list()[:200]
        return bool(vals) and all(isinstance(v, str) and v.strip() for v in vals)

    order = [c for c in tasks.columns if c in ("task", "text", "caption", "instruction")]
    order += [c for c in tasks.columns if c.startswith("__index_level_0__")]
    order += [c for c in tasks.columns if c not in order]

    for c in order:
        if c in tasks.columns and all_str(c):
            vals = [_norm(v) for v in tasks[c].to_list()]
            print(f"  captions read from column {c!r} ({len(vals)} rows)")
            return vals

    raise SystemExit(f"no textual column in tasks.parquet. columns={tasks.columns} "
                     f"head={tasks.head(3).to_dicts()}")


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
          max_frames: int = 16, cache: str = "hf_cache", videos_per_file: int = 1000,
          offset_col: Optional[str] = None):
    import numpy as np
    import polars as pl

    def read_parquet(path: str):
        # polars >=2 keeps a serialised pandas index as __index_level_0__ by default;
        # older versions need include_index=True. Try both.
        try:
            return pl.read_parquet(path, include_index=True)
        except TypeError:
            return pl.read_parquet(path)

    tasks_p, data_p, video_p = download(chunk, file, cache)
    print(f"tasks : {tasks_p}\ndata  : {data_p}\nvideo : {video_p}")

    tasks = read_parquet(tasks_p)
    frames = read_parquet(data_p)

    print(f"\ntasks.parquet  columns={tasks.columns}  rows={len(tasks)}")
    print(f"  head:\n{tasks.head(3)}")
    print(f"\ndata parquet   columns={frames.columns}  rows={len(frames)}")
    print(f"  head:\n{frames.head(5)}")

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
        raise SystemExit(f"could not resolve columns. tasks={tasks.columns} frames={frames.columns}")

    # Which column is the ABSOLUTE frame offset into the concatenated mp4?
    # `frame_index` resets to 0 in every episode, so it cannot be used to seek --
    # keying a frame->episode map by it makes hundreds of episodes collide onto the
    # same keys and silently collapses to a handful of clips. `index` is global.
    #
    # Detect it rather than assume: a global offset column is strictly increasing in
    # row order (it counts frames across the whole concatenated file). A per-episode
    # counter drops back to 0 at each episode boundary.
    def is_global(col: str) -> bool:
        return bool(frames.select((pl.col(col).diff().min() > 0).alias("g")).item())

    cand = [c for c in (offset_col or "", "index", "global_index", "frame_offset") if c and c in frames.columns]
    offset_col = next((c for c in cand if is_global(c)), None)
    if offset_col is None:
        raise SystemExit(
            f"no global frame-offset column among {cand}. "
            f"frame_index={fidx_col!r} restarts per episode and cannot be used to seek."
        )
    print(f"\nresolved: caption={task_col!r} episode={ep_col!r} task={tidx_col!r}")
    print(f"  intra-episode frame number: {fidx_col!r}")
    print(f"  GLOBAL mp4 offset column  : {offset_col!r}  (used for seeking)")
    print(f"episodes in this parquet: {frames[ep_col].n_unique()}   rows: {len(frames)}")

    # --- the parquet may cover the whole dataset while the mp4 covers only the
    # first `videos_per_file` episodes. Restrict explicitly rather than decoding
    # 9.8M frames looking for clips that are not in this file.
    all_eps = sorted(frames[ep_col].unique().to_list())
    eps_here = set(all_eps[:videos_per_file])
    print(f"episodes present in THIS mp4 (first {videos_per_file}): {len(eps_here)}")
    frames = frames.filter(pl.col(ep_col).is_in(list(eps_here)))

    captions = read_captions(tasks)
    print(f"distinct captions: {len(set(captions))}")

    # --- caption -> official template id, longest match wins ------------------
    matcher = TemplateMatcher()
    print(f"matching {len(captions)} captions against {len(OFFICIAL_TEMPLATES)} templates ...")
    task_ids = tasks[tidx_col].to_list()
    tid_of_task: Dict[int, int] = {}
    example: Dict[int, str] = {}
    for tid in task_ids:
        tid = int(tid)
        if not (0 <= tid < len(captions)):
            continue
        m = matcher.match(captions[tid])
        if m is not None and m not in example:
            example[m] = captions[tid]
        if m is not None:
            tid_of_task[tid] = m
    unparsed = len(set(task_ids)) - len(tid_of_task)
    print(f"  captions matching no template: {unparsed}/{len(set(task_ids))} "
          f"({100*unparsed/max(1,len(set(task_ids))):.1f}%) -- this mirror drops some prepositions")

    hit = Counter(tid_of_task.values())
    matched_set = {OFFICIAL_TEMPLATES[t] for t in hit} & set(CONTRASTS)
    matched = sorted(matched_set, key=lambda t: TEMPLATE_IDS[t])
    missing = sorted(set(CONTRASTS) - matched_set)
    print(f"\ntarget templates present in this dataset: {len(matched)}/{len(CONTRASTS)}")
    for t in matched:
        print(f"  [{TEMPLATE_IDS[t]:3d}] {t:52s} {hit[TEMPLATE_IDS[t]]:6d} captions")
    if missing:
        print(f"\n  NOT PRESENT ({len(missing)}): " + "; ".join(missing))

    # Verify before trusting: real captions per target, so mis-mappings are visible.
    print("\n--- VERIFICATION: sample caption per matched target (eyeball these) ---")
    for name in matched:
        tid = TEMPLATE_IDS[name]
        print(f"  {name:52s} <- {example.get(tid, '(none)')}")

    if len(matched) < 4:
        raise SystemExit(
            f"only {len(matched)} target templates found -- too few for a gate. "
            "Try more --file indices; each mp4 holds only ~1000 of 217838 episodes."
        )

    # gid = position in `matched`; keys are normalised captions of matched templates
    gid_of_cap = {_norm(m): i for i, m in enumerate(matched)}
    cap_lookup = {v: k for k, v in gid_of_cap.items()}

    # first pass: which episodes to keep, and their mp4 offset span
    lab_of_ep: Dict[int, int] = {}
    cap_of_ep: Dict[int, str] = {}
    spans: Dict[int, Tuple[int, int]] = {}
    counts: Dict[int, int] = {}
    for row in (frames.group_by(ep_col)
                    .agg(pl.col(offset_col).min().alias("lo"),
                         pl.col(offset_col).max().alias("hi"),
                         pl.col(tidx_col).first().alias("tidx"))
                    .sort(ep_col)
                    .iter_rows()):
        ep, lo, hi, tidx = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        tid = tid_of_task.get(tidx)
        if tid is None:
            continue
        name = OFFICIAL_TEMPLATES[tid]
        gid = gid_of_cap.get(_norm(name))
        if gid is None or counts.get(gid, 0) >= limit_per_template:
            continue
        counts[gid] = counts.get(gid, 0) + 1
        lab_of_ep[ep] = gid
        cap_of_ep[ep] = name
        spans[ep] = (lo, hi)

    if not lab_of_ep:
        raise SystemExit("no matching episodes -- captions mapped but none in this mp4")

    # Guard against the per-episode-frame_index bug: with a local offset column the
    # spans of different episodes overlap, which would silently yield a few clips.
    owner: Dict[int, int] = {}
    overlaps = 0
    for ep, (lo, hi) in spans.items():
        for f in range(lo, hi + 1):
            if f in owner:
                overlaps += 1
            owner[f] = ep
    last_frame = max(hi for _, hi in spans.values())
    print(f"\nkeeping {len(lab_of_ep)} episodes")
    print(f"  span {min(lo for lo,_ in spans.values())}..{last_frame} using {offset_col!r}")
    if overlaps:
        raise SystemExit(
            f"{overlaps} colliding frame offsets across {len(lab_of_ep)} episodes -- "
            f"{offset_col!r} is not a global mp4 offset. Cannot seek reliably."
        )
    if last_frame > 200_000:
        print(f"  WARNING: last_frame={last_frame} implies {last_frame/45:.0f} episodes; "
              f"this mp4 may not be the chunk the parquet describes.")
    print(f"need to decode frames 0..{last_frame}")

    # `owner` (built for the collision guard above) is the frame -> episode map.
    frame_owner = owner

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
    ap.add_argument("--offset-col", default=None,
                    help="column holding the global mp4 frame offset (auto-detected; usually 'index')")
    a = ap.parse_args()
    build(a.out, a.chunk, a.file, a.limit_per_template, a.max_frames, a.cache,
          a.videos_per_file, a.offset_col)


if __name__ == "__main__":
    main()