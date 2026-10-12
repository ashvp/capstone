"""GATE 2b -- zero-shot encoder bake-off on SSv2 minimal-intervention pairs.

Tests Condition B: can the *video* side see a single-factor change?

For every clip from template A, ask whether its embedding is closer to A's caption
than to A's antonym partner B's caption:

    edit_acc = P[ sim(clip, A) > sim(clip, B) ]        chance = 0.50

Nothing is trained. If an encoder scores ~0.50 the video representation does not
carry the relation and Stage 1 cannot work with it -- whatever happens on the text
side. If it scores well above 0.50, that encoder can see relations and Stage 1 is
worth running.

    python -m crvad.probes.encoder_bakeoff --data runs/ssv2_pairs.pt

Also reports:
  * multi-class template retrieval accuracy over every matched template (a harder
    task than the pairwise one, and directly comparable to published SSv2 numbers)
  * a bootstrap CI, because 0.52 vs 0.55 on a few hundred clips is noise
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .ssv2_data import ANTONYM_PAIRS


# --------------------------------------------------------------------------- encoders
def encode_clip(name: str, clips: torch.Tensor, dev: str, batch: int = 32):
    """(N,F,3,H,W) uint8 -> (N,D) L2-normalised, max-pooled over frames."""
    if name == "clip":
        from transformers import CLIPModel, CLIPProcessor

        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch16",
                                      torch_dtype=torch.float32).to(dev).eval()
        p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        mean = torch.tensor(p.image_processor.image_mean, device=dev).view(1, 3, 1, 1)
        std = torch.tensor(p.image_processor.image_std, device=dev).view(1, 3, 1, 1)

        outs = []
        with torch.no_grad():
            for i in range(0, len(clips), batch):
                x = clips[i : i + batch].to(dev).float().div_(255.0)
                x = x.flatten(0, 1)                                     # (B*F,3,H,W)
                x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
                x = ((x - mean) / std).clamp(0, 1)
                f = m.get_image_features(pixel_values=x)
                for attr in ("image_embeds", "pooler_output"):
                    if hasattr(f, attr):
                        f = getattr(f, attr)
                        break
                f = F.normalize(f.float(), dim=-1).view(-1, clips.shape[1], -1)
                outs.append(f.max(dim=1).values)                      # max over frames
        del m
        torch.cuda.empty_cache()
        return torch.cat(outs)

    if name in ("egovlpv2", "egovlp"):
        return _encode_egovlp(clips, dev, batch)

    raise ValueError(f"unknown encoder {name!r}")


def encode_text(name: str, sentences: List[str], dev: str) -> torch.Tensor:
    if name == "clip":
        from transformers import CLIPModel, CLIPTokenizer

        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch16",
                                      torch_dtype=torch.float32).to(dev).eval()
        tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        with torch.no_grad():
            f = m.get_text_features(**tok(sentences, return_tensors="pt", padding=True))
            for attr in ("text_embeds", "pooler_output"):
                if hasattr(f, attr):
                    f = getattr(f, attr)
                    break
            out = F.normalize(f.float(), dim=-1)
        del m
        torch.cuda.empty_cache()
        return out

    if name in ("egovlpv2", "egovlp"):
        return _egovlp_text(sentences, dev)

    raise ValueError(f"unknown encoder {name!r}")


_EG = {}


def _load_egovlp(dev: str):
    if "m" not in _EG:
        from egovlpv2.models import EgoVLPv2  # pip install from the official repo
        m = EgoVLPv2(pretrained=True)
        m = m.to(dev).eval()
        _EG["m"] = m
    return _EG["m"]


def _encode_egovlp(clips: torch.Tensor, dev: str, batch: int) -> torch.Tensor:
    m = _load_egovlp(dev)
    outs = []
    with torch.no_grad():
        for i in range(0, len(clips), batch):
            x = clips[i : i + batch].to(dev).float().div_(255.0)
            x = F.interpolate(x.flatten(0, 1), size=(224, 224), mode="bicubic", align_corners=False)
            x = x.view(-1, clips.shape[1], 3, 224, 224)
            outs.append(F.normalize(m.encode_video(x).float(), dim=-1))
    return torch.cat(outs)


def _egovlp_text(sentences: List[str], dev: str) -> torch.Tensor:
    m = _load_egovlp(dev)
    with torch.no_grad():
        return F.normalize(m.encode_text(list(sentences)).float(), dim=-1)


# --------------------------------------------------------------------------- eval
def bootstrap_ci(x: torch.Tensor, n: int = 2000, seed: int = 0) -> tuple:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(x), (n, len(x)), generator=g)
    means = x[idx].mean(dim=1)
    return means.quantile(0.025).item(), means.quantile(0.975).item()


def evaluate(data, encoder: str, dev: str) -> Dict:
    clips, labels, captions = data["clips"], data["label"], data["captions"]
    gid = {c: i for i, c in enumerate(captions)}

    V = encode_clip(encoder, clips, dev)
    Uall = encode_text(encoder, list(captions), dev)

    # Pairwise discrimination: for each (A,B), clips of A must prefer A's caption and
    # clips of B must prefer B's. Both directions pooled so an unbalanced pair cannot
    # score well by having more clips on one side.
    per_pair = {}
    obs: List[float] = []                      # one entry per (clip, pair) test
    for a, b in ANTONYM_PAIRS:
        if a not in gid or b not in gid:
            continue
        ua, ub = Uall[gid[a]], Uall[gid[b]]
        ma, mb = labels == gid[a], labels == gid[b]
        if not ma.any() or not mb.any():
            continue
        ca = ((V[ma] * ua).sum(-1) > (V[ma] * ub).sum(-1)).float()
        cb = ((V[mb] * ub).sum(-1) > (V[mb] * ua).sum(-1)).float()
        obs += ca.tolist() + cb.tolist()       # both directions, so imbalance cannot help
        n_a, n_b = int(ma.sum()), int(mb.sum())
        pair_acc = (ca.sum().item() + cb.sum().item()) / (n_a + n_b)
        margin = (((V[ma] * ua).sum(-1) - (V[ma] * ub).sum(-1)).mean().item()
                  + ((V[mb] * ub).sum(-1) - (V[mb] * ua).sum(-1)).mean().item()) / 2
        per_pair[f"{a}  vs  {b}"] = {"edit_acc": pair_acc, "margin": margin,
                                     "n_a": n_a, "n_b": n_b}

    correct = torch.tensor(obs) if obs else torch.tensor([float("nan")])
    lo, hi = bootstrap_ci(correct)
    retrieval = (V @ Uall.T).argmax(1) == labels

    return {
        "encoder": encoder,
        "n_clips": int(len(clips)),
        "n_templates": len(captions),
        "n_pairs": len(per_pair),
        "n_observations": len(obs),
        "edit_acc": correct.mean().item(),
        "edit_acc_ci95": [lo, hi],
        "template_retrieval": retrieval.float().mean().item(),
        "retrieval_chance": 1.0 / len(captions),
        "per_pair": per_pair,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/ssv2_pairs.pt")
    ap.add_argument("--encoders", nargs="*", default=["clip", "egovlpv2"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/gate2_encoder_bakeoff.json")
    a = ap.parse_args()

    data = torch.load(a.data, weights_only=False)
    print(f"{data['clips'].shape[0]} clips x {data['clips'].shape[1]} frames, "
          f"{len(data['captions'])} template pairs\n")

    results = {}
    for enc in a.encoders:
        print(f"===== {enc} " + "=" * (60 - len(enc)))
        try:
            r = evaluate(data, enc, a.device)
        except Exception as e:                       # missing package, OOM, AV1, ...
            print(f"  SKIPPED: {type(e).__name__}: {e}\n")
            continue
        results[enc] = r
        verdict = ("DEAD  -- video features do not carry the relation"
                   if r["edit_acc_ci95"][1] < 0.60 else
                   "WEAK  -- barely above chance" if r["edit_acc"] < 0.65 else
                   "OK    -- encoder can see single-factor changes")
        print(f"  edit_acc (pairwise)   : {r['edit_acc']:.3f}   95% CI [{r['edit_acc_ci95'][0]:.3f}, {r['edit_acc_ci95'][1]:.3f}]"
              f"   ({r['n_pairs']} pairs, {r['n_observations']} tests)")
        print(f"  template retrieval    : {r['template_retrieval']:.3f}   (chance {r['retrieval_chance']:.3f})")
        print(f"  -> {verdict}\n")
        for t, v in sorted(r["per_pair"].items(), key=lambda kv: kv[1]["edit_acc"]):
            print(f"    {v['edit_acc']:.3f}  (margin {v['margin']:+.4f}, n={v['n_a']}+{v['n_b']})  {t}")

    if results:
        import os
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()