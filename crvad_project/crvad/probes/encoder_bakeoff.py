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
import inspect
import json
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .ssv2_data import ANTONYM_PAIRS

# clip   openai/clip-vit-base-patch16    image tower, max over frames. No temporal
#                                       modelling at all -- the control.
# xclip  microsoft/xclip-base-patch16   CLIP fine-tuned ON VIDEO, temporal transformer
#                                       added. Same family, so clip-vs-xclip isolates
#                                       "does video pretraining buy relation
#                                       sensitivity?". No extra dependencies.
# viclip  OpenGVLab/ViCLIP-B-16-hf    CLIP fine-tuned on 10M video-text pairs
#                                       (InternVid) with spatiotemporal attention built
#                                       into the ViT, replacing per-frame max-pooling.
#                                       LOCAL SNAPSHOT, not pip-installable: see
#                                       _load_viclip for the fetch and patch notes.
# egovlpv2  egocentric VLP with fusion in the backbone. NOT pip-installable; see
#           _load_egovlp for the install note.
HF_IDS = {"clip": "openai/clip-vit-base-patch16", "xclip": "microsoft/xclip-base-patch16"}
# Hardcoded rather than read from AutoProcessor: transformers>=5 requires Pillow /
# torchvision for the image processor, and these constants are fixed per checkpoint.
# NOTE they are NOT the same -- X-CLIP inherits VideoMAE's ImageNet stats, not CLIP's.
# Using CLIP's numbers on X-CLIP would silently degrade it and flatter the comparison.
NORM = {
    "clip":   ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    "xclip":  ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ViCLIP is initialised from OpenAI CLIP and inherits its preprocessing; its own
    # frames2tensor() helper uses exactly these constants.
    "viclip": ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
}
EGOVLP = ("egovlpv2", "egovlp")
XCLIP_FRAMES = 8          # X-CLIP is trained with 8 frames; our clips have 16


def _unwrap(out):
    """transformers 5.x returns BaseModelOutputWithPooling; older returned a tensor."""
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(out, attr):
            return getattr(out, attr)
    return out


# --------------------------------------------------------------------------- encoders
def encode_clip(name: str, clips: torch.Tensor, dev: str, batch: int = 32):
    """(N,F,3,H,W) uint8 -> (N,D) L2-normalised."""
    if name in HF_IDS:
        from transformers import AutoModel, AutoProcessor

        mid = HF_IDS[name]
        model = AutoModel.from_pretrained(mid, torch_dtype=torch.float32).to(dev).eval()
        outs = []
        with torch.no_grad():
            if name == "xclip":
                k = min(XCLIP_FRAMES, clips.shape[1])
                sel = torch.linspace(0, clips.shape[1] - 1, k).round().long()
                for i in range(0, len(clips), batch):
                    x = _prep(clips[i : i + batch][:, sel], dev)
                    outs.append(F.normalize(_unwrap(model.get_video_features(
                        pixel_values=x)).float(), dim=-1))
            else:
                mean, std = _norm_stats(name, dev)
                for i in range(0, len(clips), batch):
                    chunk = clips[i : i + batch].to(dev).float().div_(255.0)
                    b, f = chunk.shape[:2]
                    x = F.interpolate(chunk.flatten(0, 1), size=(224, 224),
                                      mode="bicubic", align_corners=False)
                    x = ((x.view(b, f, 3, 224, 224) - mean) / std).clamp(0, 1)
                    feat = F.normalize(_unwrap(
                        model.get_image_features(pixel_values=x.flatten(0, 1))).float(), dim=-1)
                    # (B*F, D) -> (B, F, D). Two -1s is illegal, torch infers only one.
                    outs.append(F.normalize(feat.view(b, f, -1).max(dim=1).values, dim=-1))
        del model
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        return torch.cat(outs)

    if name == "viclip":
        return _encode_viclip(clips, dev, batch)

    if name in EGOVLP:
        return _encode_egovlp(clips, dev, batch)

    raise ValueError(f"unknown encoder {name!r}")


def _prep(clips: torch.Tensor, dev: str, size: int = 224):
    """(N,F,3,H,W) uint8 -> (N,F,3,size,size) float in [0,1]."""
    x = clips.to(dev).float().div_(255.0)
    b, f = x.shape[:2]
    x = F.interpolate(x.flatten(0, 1), size=(size, size), mode="bicubic", align_corners=False)
    return x.view(b, f, 3, size, size).clamp(0, 1)


def _norm_stats(name: str, dev: str):
    m, s = NORM[name]
    return (torch.tensor(m, device=dev).view(1, 1, 3, 1, 1),
            torch.tensor(s, device=dev).view(1, 1, 3, 1, 1))


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
        return out

    if name == "viclip":
        return _viclip_text(sentences, dev)

    if name in EGOVLP:
        return _egovlp_text(sentences, dev)

    raise ValueError(f"unknown encoder {name!r}")


# --------------------------------------------------------------------------- viclip
VICLIP_DIR = os.environ.get(
    "VICLIP_DIR", "hf_cache/models--OpenGVLab--ViCLIP-B-16-hf")
_VC = {}


def _load_viclip(dev: str):
    """ViCLIP-B-16 (InternVid, arXiv:2307.06942). Loaded from a LOCAL snapshot via
    trust_remote_code, because there is no pip package -- fetch it with:

        python tools/fetch_viclip.py

    Two patches are needed to run this 2023 remote code under transformers 5.x, both
    of which fail *after* the weights are already loaded, so neither raises a useful
    error on its own:

    1. `all_tied_weights_keys` -- transformers 5.x reads this attribute while
       finalising weight loading. The remote code predates it, so __getattr__ raises
       AttributeError once the safetensors are in place. Give the base class the empty
       default it expects rather than editing the vendored source.

    2. The text tower's causal attention mask is built in __init__ by an
       lru_cache'd method and stored as a PLAIN attribute, not a registered buffer.
       transformers 5.x builds the model on the meta device, so the mask is
       allocated on meta, is never replaced by the checkpoint (it has no key), and
       fails at the first forward with "Cannot copy out of meta tensor". Rebuild it on
       a real device after loading.

    VERIFIED CLEAN: 303/303 checkpoint tensors load bit-exact, and encode_vision is
    batch-invariant -- cosine(B=1, B=16) = 1.0000 on the same 24 clips, and
    identical with and without a hand-written CLS-token workaround. The `x[:B]`
    indexing in viclip_vision.py:171 *looks* wrong for B>1 (it slices the first B
    rows of a B*T-row tensor) but a candidate fix was measured to change nothing, so
    no monkey-patch is applied. Do not "fix" it without a failing test.
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
            f"Run: python tools/fetch_viclip.py  (or set VICLIP_DIR)"
        )
    cfg = AutoConfig.from_pretrained(VICLIP_DIR, trust_remote_code=True)
    # The repo's config carries tokenizer_path="./bpe_simple_vocab_16e6.txt.gz",
    # resolved relative to the CWD -- but trust_remote_code executes the model from
    # ~/.cache/huggingface/modules/, where that vocab file does not exist. Absolute.
    cfg.tokenizer_path = os.path.abspath(
        os.path.join(VICLIP_DIR, "bpe_simple_vocab_16e6.txt.gz"))
    m = AutoModel.from_pretrained(VICLIP_DIR, config=cfg, trust_remote_code=True)

    m.vision_encoder.transformer.checkpoint_num = 0   # training-time only; warns under no_grad
    type(m.text_encoder).build_attention_mask.cache_clear()
    mask = m.text_encoder.build_attention_mask()
    for blk in m.text_encoder.transformer.resblocks:
        blk.attn_mask = mask
    print(f"    viclip: {m.vision_encoder_name}/{m.text_encoder_name}, "
          f"{m.video_input_num_frames} frames, ctx={m.max_txt_l}, mask on {mask.device}")
    _VC["m"] = m = m.to(dev).eval()
    return m


def _encode_viclip(clips: torch.Tensor, dev: str, batch: int) -> torch.Tensor:
    m = _load_viclip(dev)
    # temporal_positional_embedding is (1, num_frames, width) and is BROADCAST, not
    # checked, so feeding the wrong frame count either errors or silently reuses a
    # single frame's embedding. Resample to exactly what the checkpoint was built with.
    t = m.video_input_num_frames
    sel = torch.linspace(0, clips.shape[1] - 1, t).round().long()
    mean, std = _norm_stats("viclip", dev)
    outs = []
    with torch.no_grad():
        for i in range(0, len(clips), batch):
            x = _prep(clips[i : i + batch][:, sel], dev)     # (B,t,3,224,224) in [0,1]
            x = (x - mean) / std                              # CLIP stats, NOT clamped
            # encode_vision permutes (B,T,C,H,W) -> (B,C,T,H,W) for the 3D conv and
            # returns a single pooled (B,512) vector -- the docstring's promised tuple
            # is not what the code returns.
            outs.append(F.normalize(m.encode_vision(x, test=True).float(), dim=-1))
    return torch.cat(outs)


def _viclip_text(sentences: List[str], dev: str) -> torch.Tensor:
    m = _load_viclip(dev)
    with torch.no_grad():
        return F.normalize(m.encode_text(list(sentences)).float(), dim=-1)


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
    ap.add_argument("--encoders", nargs="*", default=["clip", "xclip"])
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