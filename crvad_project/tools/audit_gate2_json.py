"""Re-audit a v1-format Gate-2 bake-off JSON (runs/gate2_encoder_bakeoff.json).

Flags what the v1 pooled accuracy cannot tell apart from a video-blind classifier:
  * pairs whose #correct equals one template's clip count (constant caption preference),
  * the range of pooled accuracy a constant-preference classifier can reach on these pair sizes,
  * the bias-cancelled per-pair margin (mean, SE, sign count),
  * a pair-level cluster bootstrap CI next to the (iid) one reported.

    python tools/audit_gate2_json.py runs/gate2_encoder_bakeoff.json
"""
import json, random, sys

path = sys.argv[1] if len(sys.argv) > 1 else "runs/gate2_encoder_bakeoff.json"
for enc, r in json.load(open(path)).items():
    pp = r["per_pair"]; Ns = [v["n_a"] + v["n_b"] for v in pp.values()]
    ks = [round(v["edit_acc"] * N) for v, N in zip(pp.values(), Ns)]
    tot = sum(Ns)
    const = sum(k in (v["n_a"], v["n_b"]) for k, v in zip(ks, pp.values()))
    best = sum(max(v["n_a"], v["n_b"]) for v in pp.values()) / tot
    worst = sum(min(v["n_a"], v["n_b"]) for v in pp.values()) / tot
    ms = [v["margin"] for v in pp.values()]
    mu = sum(ms) / len(ms); sd = (sum((m - mu) ** 2 for m in ms) / (len(ms) - 1)) ** .5
    random.seed(0); bs = []
    for _ in range(5000):
        idx = [random.randrange(len(Ns)) for _ in Ns]
        bs.append(sum(ks[i] for i in idx) / sum(Ns[i] for i in idx))
    bs.sort()
    print(f"== {enc}: pooled acc {r['edit_acc']:.3f}, reported iid CI {r['edit_acc_ci95'][0]:.3f}-{r['edit_acc_ci95'][1]:.3f}")
    print(f"   constant-preference signature: {const}/{len(pp)} pairs; a constant-preference classifier "
          f"reaches {worst:.3f}..{best:.3f} on these pair sizes")
    print(f"   bias-cancelled margin: mean {mu:+.5f}, SE {sd / len(ms) ** .5:.5f}, t {mu / (sd / len(ms) ** .5):+.2f}, "
          f"positive in {sum(m > 0 for m in ms)}/{len(ms)} pairs")
    print(f"   pair-cluster bootstrap 95% CI: [{bs[125]:.3f}, {bs[4875]:.3f}]")
