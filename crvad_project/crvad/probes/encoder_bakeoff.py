"""GATE 2 -- zero-shot encoder bake-off on SSv2 minimal-intervention pairs (v2, corrected).

What changed versus v1 (and why the v1 Gate-2 numbers must not be used for decisions)
  1. PREPROCESSING. v1 normalised CLIP input and then clamp(0,1)-ed it (about half the pixels
     become 0), and fed X-CLIP raw [0,1] pixels with NO normalisation at all (NORM["xclip"]
     was defined but never applied). Both are fixed by `normalise`, which never clamps.
  2. POOLING. v1 took a per-dimension MAX over frames of CLIP features (cached clips then had
     0.955 mean pairwise cosine and 97% positive entries). Default is now the mean.
  3. METRIC. v1 pooled accuracy rewards a video-blind classifier that always prefers the caption
     of the larger class. v2 reports per-pair AUC and balanced accuracy, a clip-level bootstrap
     CI on the macro AUC, and a within-pair permutation p-value (see metrics.py).
  4. TEMPORAL ORDER. Zero-shot cosine on a pooled embedding cannot separate order-defined
     pairs (opening/closing ...). Per-frame features are cached so relation_probe.py can test
     order-aware probes.

    python -m crvad.probes.encoder_bakeoff --data runs/ssv2_pairs.pt --encoders clip xclip
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .metrics import bootstrap_macro_auc, macro_auc, pair_report, permutation_p
from .ssv2_data import ANTONYM_PAIRS

# clip   openai/clip-vit-base-patch16   image tower per frame. No temporal modelling.
# xclip  microsoft/xclip-base-patch16   CLIP fine-tuned on video, temporal transformer.
# egovlpv2  NOT pip-installable; see _load_egovlp.
HF_IDS = {"clip": "openai/clip-vit-base-patch16", "xclip": "microsoft/xclip-base-patch16"}
# Fixed per checkpoint. NOT the same: X-CLIP inherits ImageNet stats, CLIP its own.
NORM = {
    "clip":   ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    "xclip":  ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "viclip": ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
}
EGOVLP = ("egovlpv2", "egovlp")
XCLIP_FRAMES = 8          # X-CLIP is trained with 8 frames
# viclip  OpenGVLab/ViCLIP-B-16-hf  CLIP initialised then trained on 10M video-text pairs
#       (InternVid, arXiv:2307.06942) with spatiotemporal attention in the ViT itself, so
#       it is NOT order-blind the way an image tower + mean-pool is. Local snapshot, loaded
#       via trust_remote_code; see _load_viclip. Uses CLIP's normalisation constants because
#       it is initialised from CLIP.
VICLIP_DIR = os.environ.get("VICLIP_DIR", "hf_cache/models--OpenGVLab--ViCLIP-B-16-hf")
_VC = {}


def _unwrap(out):
    """transformers 5.x returns BaseModelOutputWithPooling; older returned a tensor."""
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(out, attr):
            return getattr(out, attr)
    return out


def _prep(clips: torch.Tensor, dev: str, size: int = 224) -> torch.Tensor:
    """(N,F,3,H,W) uint8 -> (N,F,3,size,size) float in [0,1] (resize only; NOT normalised)."""
    x = clips.to(dev).float().div_(255.0)
    b, f = x.shape[:2]
    x = F.interpolate(x.flatten(0, 1), size=(size, size), mode="bicubic", align_corners=False)
    return x.view(b, f, 3, size, size).clamp(0, 1)


def normalise(x: torch.Tensor, name: str) -> torch.Tensor:
    """(N,F,3,H,W) in [0,1] -> per-checkpoint mean/std normalised. NEVER clamp afterwards:
    normalised values legitimately span roughly [-2, 2.6]."""
    m, s = NORM[name]
    m = torch.tensor(m, device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
    s = torch.tensor(s, device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
    return (x - m) / s


# --------------------------------------------------------------------------- encoders
def encode_frames(name: str, clips: torch.Tensor, dev: str, batch: int = 32) -> torch.Tensor:
    """(N,F,3,H,W) uint8 -> (N,T,D) L2-normalised. CLIP: one vector per frame (T=F).
    X-CLIP / EgoVLP: one video-level vector (T=1)."""
    if name == "clip":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(HF_IDS[name], torch_dtype=torch.float32).to(dev).eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(clips), batch):
                x = normalise(_prep(clips[i : i + batch], dev), "clip")
                b, f = x.shape[:2]
                feat = F.normalize(_unwrap(model.get_image_features(pixel_values=x.flatten(0, 1))).float(), dim=-1)
                outs.append(feat.view(b, f, -1).cpu())
        del model
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        return torch.cat(outs)

    if name == "xclip":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(HF_IDS[name], torch_dtype=torch.float32).to(dev).eval()
        k = min(XCLIP_FRAMES, clips.shape[1])
        sel = torch.linspace(0, clips.shape[1] - 1, k).round().long()
        outs = []
        with torch.no_grad():
            for i in range(0, len(clips), batch):
                x = normalise(_prep(clips[i : i + batch][:, sel], dev), "xclip")   # <- v1 skipped this
                outs.append(F.normalize(_unwrap(model.get_video_features(pixel_values=x)).float(), dim=-1).cpu())
        del model
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        return torch.cat(outs).unsqueeze(1)

    if name == "viclip":
        return _encode_viclip(clips, dev, batch)

    if name in EGOVLP:
        return _encode_egovlp(clips, dev, batch).cpu().unsqueeze(1)

    raise ValueError(f"unknown encoder {name!r}")


# --------------------------------------------------------------------------- viclip
def _load_viclip(dev: str):
    """ViCLIP-B-16 (InternVid, arXiv:2307.06942). No pip package; fetch with
    `python tools/fetch_viclip.py`, then load from the local snapshot via trust_remote_code.

    Two patches are required to run this 2023 remote code under transformers 5.x. Both fail
    AFTER the weights are loaded, so neither raises a useful error on its own:

    1. `all_tied_weights_keys` is read while finalising weight loading; the remote code
       predates it and only defines `_tied_weights_keys`, so __getattr__ raises
       AttributeError once the safetensors are in place. Set the base-class default.

    2. The text tower's causal attention mask is built in __init__ by an lru_cache'd method
       and stored as a PLAIN attribute, not a registered buffer. transformers 5.x builds on
       the meta device, so the mask is allocated on meta, is never replaced by the checkpoint
       (it has no key), and the first forward dies with "Cannot copy out of meta tensor".
       Rebuild it on a real device after loading.

    VERIFIED CLEAN: 303/303 checkpoint tensors load bit-exact, and encode_vision is
    batch-invariant (cosine(B=1, B=16) = 1.0000). The `x[:B]` slice at viclip_vision.py:171
    looks wrong for B>1 but a candidate fix was measured to change nothing, so no
    monkey-patch is applied. Do not "fix" it without a failing test.
    """
    if "m" in _VC:
        return _VC["m"]
    from transformers import AutoConfig, AutoModel
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    if not os.path.isdir(VICLIP_DIR):
        raise RuntimeError(
            f"ViCLIP snapshot not found at {os.path.abspath(VICLIP_DIR)!r}. "
            f"Run: python tools/fetch_viclip.py  (or set VICLIP_DIR)")
    cfg = AutoConfig.from_pretrained(VICLIP_DIR, trust_remote_code=True)
    # config ships tokenizer_path="./bpe_simple_vocab_16e6.txt.gz", resolved against the CWD --
    # but trust_remote_code executes the model from ~/.cache/huggingface/modules/, where the
    # vocab file does not exist. Must be absolute.
    cfg.tokenizer_path = os.path.abspath(
        os.path.join(VICLIP_DIR, "bpe_simple_vocab_16e6.txt.gz"))
    m = AutoModel.from_pretrained(VICLIP_DIR, config=cfg, trust_remote_code=True)

    m.vision_encoder.transformer.checkpoint_num = 0    # training-only; warns under no_grad
    type(m.text_encoder).build_attention_mask.cache_clear()
    mask = m.text_encoder.build_attention_mask()
    for blk in m.text_encoder.transformer.resblocks:
        blk.attn_mask = mask
    print(f"    viclip: {m.vision_encoder_name}/{m.text_encoder_name}, "
          f"{m.video_input_num_frames} frames, ctx={m.max_txt_l}, mask on {mask.device}")
    _VC["m"] = m = m.to(dev).eval()
    return m


def _encode_viclip(clips: torch.Tensor, dev: str, batch: int) -> torch.Tensor:
    """(N,F,3,H,W) -> (N,1,D). One pooled vector per clip: ViCLIP's spatiotemporal attention
    already integrates the frames, so there is nothing to keep per-frame (T=1)."""
    m = _load_viclip(dev)
    # temporal_positional_embedding is (1, num_frames, width) and is BROADCAST, never checked,
    # so the wrong frame count either errors or silently reuses one frame. Resample to exactly
    # what the checkpoint was built with.
    t = m.video_input_num_frames
    sel = torch.linspace(0, clips.shape[1] - 1, t).round().long()
    outs = []
    with torch.no_grad():
        for i in range(0, len(clips), batch):
            x = normalise(_prep(clips[i : i + batch][:, sel], dev), "viclip")   # never clamp
            # encode_vision permutes (B,T,C,H,W) -> (B,C,T,H,W) for the 3D conv and returns a
            # single pooled (B,512); the docstring's promised tuple is not what the code returns.
            outs.append(F.normalize(m.encode_vision(x, test=True).float(), dim=-1).cpu())
    return torch.cat(outs).unsqueeze(1)


def _viclip_text(sentences: List[str], dev: str) -> torch.Tensor:
    m = _load_viclip(dev)
    with torch.no_grad():
        return F.normalize(m.encode_text(list(sentences)).float(), dim=-1).cpu()


def pool_frames(Fr: torch.Tensor, pool: str = "mean") -> torch.Tensor:
    """(N,T,D) -> (N,D) normalised. Mean is the standard choice; 'max' is kept only to reproduce v1."""
    p = Fr.mean(1) if pool == "mean" else Fr.max(1).values
    return F.normalize(p, dim=-1)


def encode_clip(name: str, clips: torch.Tensor, dev: str, batch: int = 32, pool: str = "mean") -> torch.Tensor:
    return pool_frames(encode_frames(name, clips, dev, batch), pool)


def encode_text(name: str, sentences: List[str], dev: str) -> torch.Tensor:
    if name in HF_IDS:
        from transformers import AutoModel, AutoTokenizer

        mid = HF_IDS[name]
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModel.from_pretrained(mid, torch_dtype=torch.float32).to(dev).eval()
        with torch.no_grad():
            enc = tok(list(sentences), return_tensors="pt", padding=True, truncation=True).to(dev)
            out = F.normalize(_unwrap(model.get_text_features(**enc)).float(), dim=-1)
        del model
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        return out.cpu()
    if name == "viclip":
        return _viclip_text(sentences, dev)
    if name in EGOVLP:
        return _egovlp_text(sentences, dev).cpu()
    raise ValueError(f"unknown encoder {name!r}")



_EG = {}


def _load_egovlp(dev: str):
    """EgoVLPv2 (ICCV'23). NOT pip-installable -- the repo has no setup.py/pyproject.

    Do NOT `pip install -r requirements.txt`: it pins torch==1.13.1+cu116 and
    numpy==1.23.5, neither of which has Python 3.13 wheels, and both would break the
    Kaggle image. Install only the light deps and put the repo on sys.path:

        !git clone https://github.com/facebookresearch/EgoVLPv2.git
        !pip install -q timm einops yacs
        # then: export EGOVLP_REPO=/kaggle/working/EgoVLPv2

    The package directory is capitalised (`EgoVLPv2/`) and must be imported from the
    repo ROOT, so the module path is `EgoVLPv2.model`, not `egovlpv2.models`.
    """
    if "m" in _EG:
        return _EG["m"]
    repo = os.environ.get("EGOVLP_REPO", "/kaggle/working/EgoVLPv2")
    if not os.path.isdir(repo):
        raise RuntimeError(
            f"EgoVLPv2 repo not found at {repo!r}. Clone facebookresearch/EgoVLPv2 and set "
            f"EGOVLP_REPO, or just run: --encoders clip xclip"
        )
    sys.path.insert(0, repo)
    try:
        from EgoVLPv2 import model as ego_model
    except Exception as e:
        raise RuntimeError(f"could not import EgoVLPv2.model from {repo}: {e}")

    url = os.environ.get(
        "EGOVLP_CKPT",
        "http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2.pth",
    )
    path = url
    if url.startswith("http"):
        path = os.path.join(repo, "EgoVLPv2.pth")
        if not os.path.exists(path):
            torch.hub.download_url_to_file(url, path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("config")
    if cfg is None:
        raise RuntimeError(f"checkpoint {path} has no 'config' key; keys={list(ck)[:8]}")

    # Find the class rather than hardcoding a name -- the repo ships several variants
    # (EgoMQ, EgoNLQ, Charades) and only the pretraining one takes a config.
    cls = None
    for name, obj in vars(ego_model).items():
        if not (isinstance(obj, type) and issubclass(obj, torch.nn.Module)):
            continue
        if "egovlpv2" in name.lower() and "config" in inspect.signature(obj.__init__).parameters:
            cls = obj
            break
    if cls is None:
        raise RuntimeError("no config-accepting EgoVLPv2 class in EgoVLPv2.model; found: "
                           f"{[k for k in vars(ego_model) if 'Ego' in k]}")
    m = cls(cfg)
    missing = m.load_state_dict(ck.get("state_dict", ck), strict=False)
    print(f"    egovlpv2: {cls.__name__}, ckpt keys={list(ck)[:4]}, "
          f"missing={len(missing.missing_keys)}, unexpected={len(missing.unexpected_keys)}")
    _EG["m"] = m = m.to(dev).eval()
    return m


def _encode_egovlp(clips: torch.Tensor, dev: str, batch: int) -> torch.Tensor:
    m = _load_egovlp(dev)
    outs = []
    with torch.no_grad():
        for i in range(0, len(clips), batch):
            outs.append(F.normalize(m.encode_video(_prep(clips[i : i + batch], dev)).float(), dim=-1))
    return torch.cat(outs)


def _egovlp_text(sentences: List[str], dev: str) -> torch.Tensor:
    m = _load_egovlp(dev)
    with torch.no_grad():
        return F.normalize(m.encode_text(list(sentences)).float(), dim=-1)



# --------------------------------------------------------------------------- eval
def build_pairs(V: torch.Tensor, U: torch.Tensor, labels: torch.Tensor, captions) -> List[dict]:
    """Per antonym pair: s = sim(clip,A) - sim(clip,B) for the clips of each template."""
    gid = {c: i for i, c in enumerate(captions)}
    pairs = []
    for a, b in ANTONYM_PAIRS:
        if a not in gid or b not in gid:
            continue
        ia, ib = (labels == gid[a]).nonzero().flatten(), (labels == gid[b]).nonzero().flatten()
        if len(ia) == 0 or len(ib) == 0:
            continue
        s = V @ (U[gid[a]] - U[gid[b]])
        pairs.append({"name": f"{a}  vs  {b}", "ta": gid[a], "tb": gid[b], "sA": s[ia], "sB": s[ib]})
    return pairs


def summarise(pairs: List[dict], n_boot: int = 2000, n_perm: int = 2000) -> Dict:
    per_pair = {p["name"]: pair_report(p["sA"], p["sB"]) for p in pairs}
    lo, hi = bootstrap_macro_auc(pairs, n_boot)
    return {
        "n_pairs": len(pairs),
        "macro_auc": macro_auc(pairs),
        "macro_auc_ci95": [lo, hi],
        "perm_p": permutation_p(pairs, n_perm),
        "macro_acc_balanced": sum(r["acc_balanced"] for r in per_pair.values()) / max(1, len(per_pair)),
        "pooled_acc_raw_v1_style": sum(r["acc_raw"] * (r["n_a"] + r["n_b"]) for r in per_pair.values())
                                   / max(1, sum(r["n_a"] + r["n_b"] for r in per_pair.values())),
        "n_constant_preference_pairs": sum(r["constant_preference"] for r in per_pair.values()),
        "per_pair": per_pair,
    }


def evaluate(data, encoder: str, dev: str, pool: str = "mean") -> Dict:
    clips, labels, captions = data["clips"], data["label"], data["captions"]
    V = encode_clip(encoder, clips, dev, pool=pool)
    U = encode_text(encoder, list(captions), dev)
    r = summarise(build_pairs(V, U, labels, captions))
    r.update(encoder=encoder, pool=pool, n_clips=int(len(clips)), n_templates=len(captions),
             template_retrieval=((V @ U.T).argmax(1) == labels).float().mean().item(),
             retrieval_chance=1.0 / len(captions))
    return r


def verdict(r: Dict) -> str:
    lo = r["macro_auc_ci95"][0]
    if lo > 0.55 and r["perm_p"] < 0.05:
        return "SIGNAL -- zero-shot text-conditioned relation sensitivity above chance"
    if r["perm_p"] < 0.05:
        return "WEAK SIGNAL -- above chance, CI too wide or too low to rely on"
    return "NO ZERO-SHOT SIGNAL (says little: pooled zero-shot cannot see order; run relation_probe)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/ssv2_pairs.pt")
    ap.add_argument("--encoders", nargs="*", default=["clip", "xclip"])
    ap.add_argument("--pool", choices=["mean", "max"], default="mean")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/gate2_encoder_bakeoff_v2.json")
    a = ap.parse_args()

    data = torch.load(a.data, weights_only=False)
    print(f"{data['clips'].shape[0]} clips x {data['clips'].shape[1]} frames, {len(data['captions'])} templates\n")
    results = {}
    for enc in a.encoders:
        print(f"===== {enc} " + "=" * (60 - len(enc)))
        try:
            r = evaluate(data, enc, a.device, a.pool)
        except Exception as e:
            print(f"  SKIPPED: {type(e).__name__}: {e}\n")
            continue
        results[enc] = r
        print(f"  macro AUC (bias-invariant): {r['macro_auc']:.3f}  95% CI [{r['macro_auc_ci95'][0]:.3f}, "
              f"{r['macro_auc_ci95'][1]:.3f}]  perm p={r['perm_p']:.3f}  ({r['n_pairs']} pairs)")
        print(f"  macro balanced acc: {r['macro_acc_balanced']:.3f}   v1-style pooled acc: {r['pooled_acc_raw_v1_style']:.3f}"
              f"   constant-preference pairs: {r['n_constant_preference_pairs']}/{r['n_pairs']}")
        print(f"  template retrieval: {r['template_retrieval']:.3f} (chance {r['retrieval_chance']:.3f})")
        print(f"  -> {verdict(r)}\n")
        for t, v in sorted(r["per_pair"].items(), key=lambda kv: kv[1]["auc"]):
            flag = " [const]" if v["constant_preference"] else ""
            print(f"    AUC {v['auc']:.3f} balAcc {v['acc_balanced']:.3f} (n={v['n_a']}+{v['n_b']}){flag}  {t}")
    if results:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
