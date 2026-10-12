"""End-to-end smoke test on synthetic features (CPU, ~minutes). Verifies the code path, not the science.

    python tests/smoke_test.py
It also runs the headline ablation in miniature (negative mode: random vs multi vs single).
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from crvad import pseudo_label as PL
from crvad import slots as S
from crvad import synthetic as SY
from crvad import train as TR
from crvad.evaluate import evaluate
from crvad.model import CRVADModel

torch.set_num_threads(max(1, os.cpu_count() or 1))
tmp = tempfile.mkdtemp()
held = S.compositional_split(seed=0, frac=0.2)
world = SY.World(seed=0)

# ---- unit checks ------------------------------------------------------------------------------
assert len(S.all_events()) == S.N_EVENTS == 1728
assert all(S.encode(e) == i for i, e in enumerate(S.all_events()[:2000]))
ev = S.all_events()[123]
assert all(sum(a != b for a, b in zip(ev, e)) == 1 for e in S.single_slot_edits(ev))
assert len(S.single_slot_edits(ev)) == S.N_SINGLE_EDITS == 22
assert len({a for a, _ in held}) <= S.RADIX[0] and len(held) == 10
from crvad.losses import flat_index, sample_negatives
pos = torch.tensor([ev] * 64)
for mode, n_changed in [("single", {1}), ("multi", {2, 3, 4})]:
    neg = sample_negatives(pos, 8, mode)
    diff = (neg != pos[:, None]).sum(-1)
    assert set(diff.flatten().tolist()) <= n_changed, (mode, diff.unique())
assert (flat_index(pos) == S.encode(ev)).all()
print("unit checks ok;", len(held), "held-out (action,object) pairs")

# ---- pseudo-label plumbing (mock VLM) ---------------------------------------------------------
rows = PL.run(PL.MockBackend(seed=1), [dict(id=str(i), frames=[], video_label=i % 2) for i in range(40)], n_samples=5)
assert any(r["slots"] for r in rows) and all(0 <= r["conf"] <= 1 for r in rows)
assert PL.parse_response('x {"action": "Takes", "object": "bag", "scene": "store", "relation": "from a shelf"} y')["action"] == 0
PL.audit_csv(rows, os.path.join(tmp, "audit.csv"), n=10)
print("pseudo-label plumbing ok")

# ---- data -------------------------------------------------------------------------------------
tb = world.text_bank
s1 = SY.make_stage1(world, 3000, 8, held, seed=1)
s2 = SY.make_videos(world, 600, 32, held, seed=2)
ev_seen = SY.make_videos(world, 200, 32, held, seed=3, want_held=False, pseudo_noise=0)
ev_held = SY.make_videos(world, 200, 32, held, seed=4, want_held=True, pseudo_noise=0)
videos = {k: torch.cat([ev_seen[k], ev_held[k]]) for k in ev_seen}
pairs = SY.make_pairs(world, 300, 8, held, seed=5)
torch.save(tb, f"{tmp}/tb.pt"); torch.save(s1, f"{tmp}/s1.pt"); torch.save(s2, f"{tmp}/s2.pt")

results = {}
for mode in ["random", "multi", "single"]:
    cfg = TR.merge(TR.DEFAULTS, dict(
        text_bank=f"{tmp}/tb.pt", out=f"{tmp}/run_{mode}", workers=0, device="cpu", amp=False,
        model=dict(d=128, n_layers=1, n_heads=4),
        stage1=dict(train=f"{tmp}/s1.pt", epochs=8, bs=128, neg_mode=mode),
        stage2=dict(train=f"{tmp}/s2.pt", init=f"{tmp}/run_{mode}/stage1/best.pt", epochs=8, bs=32, neg_mode=mode)))
    p1 = TR.run(cfg, 1)
    p2 = TR.run(cfg, 2)
    ck = torch.load(p2, map_location="cpu", weights_only=False)
    model = CRVADModel(ck["d_vid"], ck["d_txt"], **ck["cfg"]["model"])
    model.load_state_dict(ck["model"])
    results[mode] = evaluate(model, tb, videos, pairs)
    r = results[mode]
    print(f"== {mode}: video_auc={r['videos/all']['video_auc']:.3f} seg_auc={r['videos/all'].get('seg_auc', float('nan')):.3f} "
          f"edit_acc={r['pairs/all']['edit_acc']:.3f} flip={r['pairs/all']['flip_acc']:.3f} "
          f"flip_nuis={r['pairs/all']['flip_acc_nuis']:.3f} | heldout flip={r['pairs/heldout']['flip_acc']:.3f} "
          f"edit={r['pairs/heldout']['edit_acc']:.3f}")

json.dump(results, open(os.path.join(tmp, "results.json"), "w"), indent=1)
print("results saved to", tmp)
print("SMOKE TEST PASSED")
