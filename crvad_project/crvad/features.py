"""Frozen-encoder feature extraction (GPU). NOT exercised by the smoke test (needs weights + videos).

  * `extract_windows`: video -> windows of T segments; each segment = mean CLIP image feature of `seg_frames`
    frames. Optionally also a nuisance-augmented view (colour jitter + random-resized crop, same params per clip).
  * `build_text_bank`: embeds all 1728 hypothesis sentences with the CLIP text tower.

    CUDA_VISIBLE_DEVICES=0 python -m crvad.features videos --list v.txt --out feats_0.pt --shard 0/2 --nuisance
    CUDA_VISIBLE_DEVICES=1 python -m crvad.features videos --list v.txt --out feats_1.pt --shard 1/2 --nuisance
    python -m crvad.features text --out data/text_bank.pt
"""
from __future__ import annotations

import argparse
from typing import List

import torch

from . import slots as S

CLIP = "openai/clip-vit-base-patch16"


def _unwrap(out):
    """transformers>=5 returns BaseModelOutputWithPooling from get_*_features; older returned a tensor."""
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(out, attr):
            return getattr(out, attr)
    return out


def window_starts(n_frames: int, per: int) -> List[int]:
    """Start index of every window of `per` sampled frames. The last window is right-aligned
    (overlapping the previous one) so the video tail is always covered -- v1 dropped it, and on
    surveillance video that is often where the anomaly is. Videos shorter than `per` give one
    window that the caller pads."""
    if n_frames <= per:
        return [0]
    starts = list(range(0, n_frames - per + 1, per))
    if starts[-1] + per < n_frames:
        starts.append(n_frames - per)
    return starts


def _load_clip(dev):
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained(CLIP, torch_dtype=torch.float16).to(dev).eval()
    return m, CLIPProcessor.from_pretrained(CLIP)


@torch.no_grad()
def build_text_bank(out: str, dev: str = "cuda"):
    m, p = _load_clip(dev)
    sents = [S.to_text(e) for e in S.all_events()]
    embs = []
    for i in range(0, len(sents), 256):
        t = p(text=sents[i:i + 256], return_tensors="pt", padding=True).to(dev)
        embs.append(torch.nn.functional.normalize(_unwrap(m.get_text_features(**t)).float(), dim=-1).cpu())
    torch.save(torch.cat(embs), out)


def _read_frames(path: str, fps_sample: float):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / fps_sample)))
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(torch.from_numpy(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
        i += 1
    cap.release()
    return frames


@torch.no_grad()
def _encode(m, p, frames: List[torch.Tensor], dev) -> torch.Tensor:
    x = p(images=[f for f in frames], return_tensors="pt", do_rescale=True).pixel_values.to(dev, torch.float16)
    return torch.nn.functional.normalize(_unwrap(m.get_image_features(pixel_values=x)).float(), dim=-1).cpu()


def extract_windows(paths: List[str], out: str, T: int = 32, seg_frames: int = 8, fps_sample: float = 2.0,
                    nuisance: bool = False, shard: str = "0/1", dev: str = "cuda"):
    from torchvision.transforms import v2
    m, p = _load_clip(dev)
    aug = v2.Compose([v2.RandomResizedCrop(224, scale=(0.5, 1.0), antialias=True), v2.ColorJitter(0.4, 0.4, 0.4, 0.1)])
    i, n = map(int, shard.split("/"))
    feats, nuis, meta = [], [], []
    for k, path in enumerate(paths):
        if k % n != i:
            continue
        fr = _read_frames(path, fps_sample)
        per = T * seg_frames
        for w0 in window_starts(len(fr), per):
            win = fr[w0:w0 + per]
            if len(win) < per:                       # pad the tail by repeating the last frame
                win = win + [win[-1]] * (per - len(win))
            f = _encode(m, p, win, dev).view(T, seg_frames, -1).mean(1)
            feats.append(f.half())
            if nuisance:
                a = aug(torch.stack(win))            # one random draw applied to the whole clip
                nuis.append(_encode(m, p, list(a), dev).view(T, seg_frames, -1).mean(1).half())
            meta.append((path, w0))
    d = dict(feats=torch.stack(feats))
    if nuisance:
        d["nuis_feats"] = torch.stack(nuis)
    torch.save(d, out)
    with open(out + ".meta.txt", "w") as f:
        f.writelines(f"{p_}\t{w}\n" for p_, w in meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("videos"); v.add_argument("--list", required=True); v.add_argument("--out", required=True)
    v.add_argument("--shard", default="0/1"); v.add_argument("--nuisance", action="store_true")
    t = sub.add_parser("text"); t.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "text":
        build_text_bank(a.out)
    else:
        extract_windows([l.strip() for l in open(a.list) if l.strip()], a.out, nuisance=a.nuisance, shard=a.shard)
