"""Synthetic feature world used ONLY to smoke-test the code path (no scientific claims).

Video features are additive slot vectors + a nuisance 'style' vector + noise; text-bank embeddings are
additive slot vectors in a different space. Abnormality follows slots.ABNORMAL_RULES. Videos are T-segment
sequences; abnormal videos contain a short abnormal window inside a normal background event.
"""
from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

import torch

from . import slots as S

ALL = S.all_events()
ABN = [e for e in ALL if S.is_abnormal(e)]
NRM = [e for e in ALL if not S.is_abnormal(e)]


class World:
    def __init__(self, d_vid: int = 64, d_txt: int = 48, n_styles: int = 6, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.vec = [torch.randn(r, d_vid, generator=g) for r in S.RADIX]
        self.styles = torch.randn(n_styles, d_vid, generator=g)
        tv = [torch.randn(r, d_txt, generator=g) for r in S.RADIX]
        ev = torch.tensor(ALL)
        self.text_bank = sum(tv[i][ev[:, i]] for i in range(4)) + 0.05 * torch.randn(len(ALL), d_txt, generator=g)
        self.n_styles = n_styles
        self.d_vid = d_vid

    def render(self, ev: S.Event, style: int, T: int, gen: torch.Generator, noise: float = 0.3) -> torch.Tensor:
        base = sum(self.vec[i][ev[i]] for i in range(4)) + 0.5 * self.styles[style]
        return base[None].repeat(T, 1) + noise * torch.randn(T, self.d_vid, generator=gen)


def _pick(rng: random.Random, pool: List[S.Event], held: Set[Tuple[int, int]], want_held: bool) -> S.Event:
    cand = [e for e in pool if S.is_heldout(e, held) == want_held]
    return rng.choice(cand)


def make_stage1(world: World, n: int, T: int, held, seed: int):
    rng, g = random.Random(seed), torch.Generator().manual_seed(seed)
    feats, nuis, slots_ = [], [], []
    for _ in range(n):
        ev = _pick(rng, ALL, held, False)
        s0, s1 = rng.randrange(world.n_styles), rng.randrange(world.n_styles)
        feats.append(world.render(ev, s0, T, g))
        nuis.append(world.render(ev, s1, T, g))
        slots_.append(ev)
    return dict(feats=torch.stack(feats), nuis_feats=torch.stack(nuis), slots=torch.tensor(slots_))


def make_videos(world: World, n: int, T: int, held, seed: int, want_held: bool = False, pseudo_noise: float = 0.15):
    """Weakly-labelled long videos with an abnormal window; pseudo-slots mimic a noisy VLM."""
    rng, g = random.Random(seed), torch.Generator().manual_seed(seed)
    F_, N_, SL, Y, C, SEG, H = [], [], [], [], [], [], []
    for i in range(n):
        y = i % 2
        base = _pick(rng, NRM, held, want_held)
        s0, s1 = rng.randrange(world.n_styles), rng.randrange(world.n_styles)
        f, fn = world.render(base, s0, T, g), world.render(base, s1, T, g)
        seg = torch.zeros(T, dtype=torch.long)
        salient = base
        if y:
            ab = _pick(rng, ABN, held, want_held)
            L = rng.randint(max(2, T // 8), max(3, T // 4))
            a = rng.randrange(0, T - L)
            f[a:a + L] = world.render(ab, s0, L, g)
            fn[a:a + L] = world.render(ab, s1, L, g)
            seg[a:a + L] = 1
            salient = ab
        conf = 1.0
        if rng.random() < pseudo_noise:
            salient, conf = rng.choice(ALL), 0.3
        F_.append(f); N_.append(fn); SL.append(salient); Y.append(y); C.append(conf); SEG.append(seg)
        H.append(S.is_heldout(base if not y else ab, held))
    return dict(feats=torch.stack(F_), nuis_feats=torch.stack(N_), slots=torch.tensor(SL),
                video_label=torch.tensor(Y), conf=torch.tensor(C), seg_labels=torch.stack(SEG),
                heldout=torch.tensor(H))


def make_pairs(world: World, n: int, T: int, held, seed: int):
    """Controlled matched pairs: abnormal event vs the same event with ONE slot (relation) edited to normal."""
    rng, g = random.Random(seed), torch.Generator().manual_seed(seed)
    out: Dict[str, List] = {k: [] for k in ["feats0", "feats1", "nuis0", "nuis1", "slots0", "slots1", "heldout"]}
    for i in range(n):
        e1 = rng.choice(ABN)
        rel = rng.choice([r for r in range(S.RADIX[3]) if not S.is_abnormal((e1[0], e1[1], e1[2], r))])
        e0 = (e1[0], e1[1], e1[2], rel)
        s0, s1 = rng.randrange(world.n_styles), rng.randrange(world.n_styles)
        out["feats0"].append(world.render(e0, s0, T, g)); out["feats1"].append(world.render(e1, s0, T, g))
        out["nuis0"].append(world.render(e0, s1, T, g)); out["nuis1"].append(world.render(e1, s1, T, g))
        out["slots0"].append(e0); out["slots1"].append(e1); out["heldout"].append(S.is_heldout(e1, held))
    return {k: torch.stack(v) if k.startswith(("feats", "nuis")) else torch.tensor(v) for k, v in out.items()}
