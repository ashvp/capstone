"""GATE 1 -- text-space separability of minimal (single-slot) interventions.

The whole thesis rests on one premise:

    changing exactly one slot produces a *different, harder* negative
    than drawing a hypothesis at random.

Negatives are indices into a text bank, so that premise lives entirely in the
text encoder's embedding space. If the encoder maps

    "A person covers a bag with a box in a store."      (true)
    "A person uncovers a bag from a box in a store."    (single-slot edit)

to nearly the same point, then a single-slot negative IS a random negative and
the proposed sampler is an empty phrase. No amount of GPU time, no video-side
encoder swap and no training run recovers a distinction that was never in the
representation.

This probe answers that with ~1700 short sentences and one CPU. Run it before
downloading anything.

    python -m crvad.probes.text_separability --encoder clip

Two banks are compared:

  sentences : the 1728 rendered hypothesis sentences (what crvad/features.py does)
  slots     : 26 per-slot phrases composed additively -- the structural fallback,
              where the factorisation holds by construction

Verdict (AUC for telling single-slot edits apart from random pairs):

    > 0.85  green      the text space really represents slot changes
    0.70-0.85  ok      workable, but consider a stronger text encoder
    0.55-0.70  warn    single-slot negatives are only mildly harder than random
    < 0.55  FAIL      chance. The text space is mostly template. Redesign.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .. import slots as S

# Slot phrases are embedded bare ("covering", "bag") rather than inside the
# template: the template dominates CLIP's text embedding and would swamp the
# one or two words that actually carry the slot identity.
SLOT_PROMPTS = {
    name: [f"a person {v}" for v in S.SCHEMA[name]] for name in S.SLOT_NAMES
}
SLOT_PROMPTS["scene"] = [f"a scene of {v}" for v in S.SCHEMA["scene"]]


_CACHE: Dict[str, torch.Tensor] = {}


@torch.no_grad()
def embed(sentences: List[str], model_id: str, batch: int = 256) -> torch.Tensor:
    """L2-normalised text embeddings, (len(sentences), d). CPU, fp32.

    Cached by (model_id, tuple(sentences)) -- re-running the probe is then instant,
    which matters because this gets re-run whenever the text encoder changes.
    """
    key = f"{model_id}||" + "\x00".join(sentences)
    if key in _CACHE:
        return _CACHE[key]

    from transformers import CLIPModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float32).eval()

    out = []
    for i in range(0, len(sentences), batch):
        enc = tok(sentences[i : i + batch], return_tensors="pt", padding=True, truncation=True)
        feats = model.get_text_features(**enc)
        # transformers 5.x returns BaseModelOutputWithPooling; older returns a tensor
        for attr in ("text_embeds", "pooler_output"):
            if hasattr(feats, attr):
                feats = getattr(feats, attr)
                break
        out.append(F.normalize(feats.float(), dim=-1))
    _CACHE[key] = torch.cat(out)
    return _CACHE[key]


def bank_sentences(model_id: str) -> torch.Tensor:
    return embed([S.to_text(e) for e in S.all_events()], model_id)


def bank_slots(model_id: str) -> torch.Tensor:
    """Compose each hypothesis from additive per-slot embeddings (the fallback)."""
    events = torch.tensor(S.all_events(), dtype=torch.long)  # (1728, 4)
    parts: List[torch.Tensor] = []
    for j, name in enumerate(S.SLOT_NAMES):
        v = embed(SLOT_PROMPTS[name], model_id)             # (r_j, d)
        parts.append(v[events[:, j]])                       # (1728, d)
    return F.normalize(sum(parts), dim=-1)


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Mann-Whitney U: P(pos > neg), ties counted as half."""
    n, m = len(pos), len(neg)
    combined = torch.cat([pos, neg])
    ranks = combined.argsort().argsort().float() + 1.0
    r_pos = ranks[:n].sum().item()
    u = r_pos - n * (n + 1) / 2.0
    return u / (n * m)


def cohen_d(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.mean() - b.mean()) / (0.5 * (a.var(unbiased=True) + b.var(unbiased=True))).sqrt()).item()


def analyse(U: torch.Tensor, rng: torch.Generator, n_random: int) -> Dict:
    """Compare single-slot-edit similarity against random-pair similarity."""
    N = U.shape[0]
    sim = U @ U.T# (N, N) cosine

    # --- signal: each event against its 22 single-slot edits ---
    edits = torch.tensor(
        [[S.encode(e) for e in S.single_slot_edits(t)] for t in S.all_events()], dtype=torch.long
    )  # (1728, 22)
    assert edits.shape[1] == S.N_SINGLE_EDITS, edits.shape
    slot_ids = torch.tensor(S.edit_slot_ids(), dtype=torch.long)

    ar = torch.arange(N).unsqueeze(1)
    edit_sim = sim[ar, edits]  # (1728, 22)

    # --- noise: uniformly random hypotheses, excluding self and all 22 edits ---
    forbidden = torch.cat([edits, torch.arange(N).unsqueeze(1)], dim=1)  # (1728, 23)
    neg = torch.randint(0, N, (N, n_random), generator=rng)
    # reject self / edit collisions by resampling until clean
    for _ in range(20):
        bad = (neg.unsqueeze(2) == forbidden.unsqueeze(1)).any(-1)  # (N, n_random)
        if not bad.any():
            break
        neg[bad] = torch.randint(0, N, (int(bad.sum()),), generator=rng)
    rand_sim = sim[ar.expand(-1, n_random), neg]  # (1728, n_random)

    pos_f, neg_f = edit_sim.flatten(), rand_sim.flatten()
    res = {
        "n_events": N,
        "n_edit_pairs": len(pos_f),
        "n_random_pairs": len(neg_f),
        "edit_mean": pos_f.mean().item(),
        "edit_std": pos_f.std().item(),
        "random_mean": neg_f.mean().item(),
        "random_std": neg_f.std().item(),
        "gap": (neg_f.mean() - pos_f.mean()).item(),
        "cohen_d": cohen_d(pos_f, neg_f),
        "auc_edit_vs_random": auc(pos_f, neg_f),
        "per_slot": {},
    }
    for j, name in enumerate(S.SLOT_NAMES):
        m = slot_ids == j
        p = edit_sim[:, m].flatten()
        res["per_slot"][name] = {
            "mean": p.mean().item(),
            "auc": auc(p, neg_f),
            "cohen_d": cohen_d(p, neg_f),
        }

    # context: how spread out is the whole bank?
    off = sim[~torch.eye(N, dtype=torch.bool)]
    res["bank_offdiag_mean"] = off.mean().item()
    res["bank_offdiag_std"] = off.std().item()
    return res


def verdict(a: float) -> str:
    if a > 0.85:
        return "GREEN  -- text space represents slot changes; proceed"
    if a > 0.70:
        return "OK     -- workable; consider a stronger text encoder"
    if a > 0.55:
        return "WARN   -- single-slot negatives only mildly harder than random"
    return "FAIL   -- text space is mostly template; single-slot negatives ~= random"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="openai/clip-vit-base-patch16")
    ap.add_argument("--banks", nargs="*", default=["sentences", "slots"])
    ap.add_argument("--n-random", type=int, default=64, help="random comparisons per event")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional JSON dump path")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = torch.Generator().manual_seed(a.seed)
    all_res: Dict[str, Dict] = {}

    print(f"encoder: {a.encoder}   device: cpu   events: {S.N_EVENTS}   edits/event: {S.N_SINGLE_EDITS}\n")

    for kind in a.banks:
        print(f"--- bank: {kind} " + "-" * (58 - len(kind)))
        U = bank_sentences(a.encoder) if kind == "sentences" else bank_slots(a.encoder)
        r = analyse(U, rng, a.n_random)
        all_res[kind] = r
        print(f"  similarity, single-slot edit : {r['edit_mean']:.4f} +/- {r['edit_std']:.4f}")
        print(f"  similarity, random pair      : {r['random_mean']:.4f} +/- {r['random_std']:.4f}")
        print(f"  gap                          : {r['gap']:.4f}   Cohen's d = {r['cohen_d']:.3f}")
        print(f"  AUC edit-vs-random           : {r['auc_edit_vs_random']:.4f}   {verdict(r['auc_edit_vs_random'])}\n")
        print("  per-slot (relation is the one ABNORMAL_RULES depends on):")
        for name, v in r["per_slot"].items():
            flag = "  <-- critical" if name == "relation" else ""
            print(f"    {name:9s} mean={v['mean']:.4f}  AUC={v['auc']:.4f}  d={v['cohen_d']:.3f}{flag}")
        print(f"\n  bank off-diagonal cosine: mean={r['bank_offdiag_mean']:.4f} std={r['bank_offdiag_std']:.4f}")
        print("  (a tiny std means all 1728 sentences sit in a tight cluster -- template bias)\n")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(all_res, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()