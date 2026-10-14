"""Fast unit tests (no model weights, no network). Each test pins down a failure that produced a
silently wrong number in the first Gate-2 run.   python tests/test_probes_v2.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from crvad.features import window_starts
from crvad.losses import flat_index
from crvad.probes.encoder_bakeoff import NORM, _prep, normalise, pool_frames
from crvad.probes.metrics import bootstrap_macro_auc, macro_auc, pair_report, permutation_p
from crvad.probes.relation_probe import cv_pair_auc, probe_macro, variants
from crvad.pseudo_label import parse_response


def test_normalise_does_not_clamp():
    torch.manual_seed(0)
    clips = (torch.rand(4, 8, 3, 64, 64) * 255).to(torch.uint8)       # uniform pixels
    x = _prep(clips, "cpu")
    assert 0 <= x.min() and x.max() <= 1
    for name in ("clip", "xclip"):
        z = normalise(x, name)
        assert z.min() < -1.5 and z.max() > 1.5, f"{name}: range {z.min():.2f}..{z.max():.2f} looks clamped"
        assert abs(z.mean()) < 0.3 and 0.8 < z.std() < 1.4
    # the two checkpoints use different constants
    assert not torch.allclose(normalise(x, "clip"), normalise(x, "xclip"))
    assert NORM["clip"] != NORM["xclip"]


def test_pool_is_mean_by_default_and_normalised():
    Fr = torch.nn.functional.normalize(torch.randn(5, 16, 32), dim=-1)
    p = pool_frames(Fr)
    assert torch.allclose(p.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(p, torch.nn.functional.normalize(Fr.mean(1), dim=-1), atol=1e-6)


def test_window_starts_cover_the_tail():
    assert window_starts(10, 32) == [0]
    assert window_starts(32, 32) == [0]
    assert window_starts(96, 32) == [0, 32, 64]
    for n in (33, 70, 100, 257):
        st = window_starts(n, 32)
        assert st[0] == 0 and st[-1] + 32 == n, (n, st)      # last window ends exactly at the video end


def test_parse_response():
    ok = '{"action": "takes", "object": "bag", "scene": "store", "relation": "from a shelf"}'
    assert parse_response("noise " + ok + " trailing")["action"] == 0
    nested = '{"note": {"x": 1}, "action": "throws", "object": "box", "scene": "plaza", "relation": "toward a person"}'
    assert parse_response(nested)["action"] == 3                  # v1's non-greedy regex truncated this
    two = '{"junk": 1} ' + ok
    assert parse_response(two)["action"] == 0                       # unrelated {"junk": 1} is skipped
    assert parse_response("no json here") is None


def test_negative_slot_is_rejected():
    try:
        flat_index(torch.tensor([[0, 0, 0, -1]]))
    except ValueError:
        return
    raise AssertionError("flat_index accepted a -1 slot")


def test_video_blind_classifier_is_not_rewarded():
    """Always prefer caption A (s = large constant + noise): v1 pooled accuracy = n_A/(n_A+n_B)."""
    g = torch.Generator().manual_seed(0)
    sA, sB = 0.5 + 1e-3 * torch.randn(12, generator=g), 0.5 + 1e-3 * torch.randn(4, generator=g)
    r = pair_report(sA, sB)
    assert abs(r["acc_raw"] - 0.75) < 1e-9 and r["constant_preference"]
    assert abs(r["acc_balanced"] - 0.5) < 1e-9
    assert abs(r["auc"] - 0.5) < 0.25                            # chance-level, not 0.75
    pairs = [{"ta": i, "tb": 100 + i, "sA": 0.5 + 1e-3 * torch.randn(12, generator=g),
              "sB": 0.5 + 1e-3 * torch.randn(4, generator=g)} for i in range(15)]
    assert 0.4 < macro_auc(pairs) < 0.6
    assert permutation_p(pairs, 300) > 0.05


def test_real_signal_is_detected():
    g = torch.Generator().manual_seed(1)
    pairs = [{"ta": i, "tb": 100 + i, "sA": 0.3 + 0.2 * torch.randn(10, generator=g),
              "sB": -0.3 + 0.2 * torch.randn(10, generator=g)} for i in range(8)]
    assert macro_auc(pairs) > 0.9
    lo, hi = bootstrap_macro_auc(pairs, 300)
    assert lo > 0.8 and permutation_p(pairs, 300) < 0.01


def _order_pairs(signal_order=True, n=14, T=8, D=24, seed=0, scale=1.0, offset=0.0):
    g = torch.Generator().manual_seed(seed)
    u, v = torch.randn(D, generator=g), torch.randn(D, generator=g)
    def clip(forward):
        a, b = (u, v) if forward else (v, u)
        t = torch.linspace(0, 1, T)[:, None]
        return (1 - t) * a + t * b + 0.5 * torch.randn(T, D, generator=g)
    A = torch.stack([clip(True) for _ in range(n)])
    B = torch.stack([clip(signal_order) if not signal_order else clip(False) for _ in range(n)])
    return A * scale + offset, B * scale + offset


def test_order_only_pairs_need_order_aware_features():
    A, B = _order_pairs(True)
    vA, vB = variants(A), variants(B)
    mean_auc = cv_pair_auc(vA["mean"], vB["mean"])
    halves_auc = cv_pair_auc(vA["halves"], vB["halves"])
    assert mean_auc < 0.7, mean_auc                              # order-blind cannot solve it
    assert halves_auc > 0.9, halves_auc                          # order-aware can


def test_probe_is_scale_and_offset_invariant():
    """CLIP clip features have per-dim std ~0.009 and a huge common offset; v1 died on that."""
    A, B = _order_pairs(True, seed=3)
    base = cv_pair_auc(variants(A)["halves"], variants(B)["halves"])
    tiny = cv_pair_auc(variants(A * 0.009 + 0.03)["halves"], variants(B * 0.009 + 0.03)["halves"])
    assert abs(base - tiny) < 0.05 and tiny > 0.9


def test_no_signal_gives_chance_and_calibrated_null():
    g = torch.Generator().manual_seed(5)
    pf = [{"A": torch.randn(10, 40, generator=g), "B": torch.randn(10, 40, generator=g)} for _ in range(6)]
    r = probe_macro(pf, n_perm=60)
    assert 0.35 < r["macro_auc"] < 0.65 and r["p_value"] > 0.05, r
    assert abs(r["null_mean"] - 0.5) < 0.08


if __name__ == "__main__":
    fails = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            try:
                f(); print("ok  ", k)
            except Exception as e:
                fails += 1; print("FAIL", k, "->", type(e).__name__, e)
    sys.exit(1 if fails else 0)
