# CRVAD — Implementation Plan

**Counterfactual Relational Video Anomaly Detection through Minimal-Intervention Contrastive Learning**

Status: draft for review · supersedes `crvad_pipeline.tex` where they conflict
Last updated: 12 October 2026

---

## 0. Decisions taken and still open

| # | Decision | Status |
|---|---|---|
| 1 | Frozen video encoder, train heads only | **Settled** — keeps ablation clean and compute small |
| 2 | Candidate encoder: EgoVLPv2 | **Provisional** — Gate 2 decides |
| 3 | Stage 1 data: EgoClip + VLM pseudo-labels | **Provisional** — see §6 |
| 4 | Stage 2 data: UCF-Crime train + XD-Violence train | **Settled** (multi-source, to limit domain gap) |
| 5 | Supervision: weakly-supervised (video labels only) | **Settled** — required for comparability |
| 6 | Controlled pair set: mined from UCA | **Settled** — replaces manual collection |
| 7 | Text encoder for the 1728 hypotheses | **Open** — depends on Gate 1 |
| 8 | Add an embedding-hard-negative arm to the ablation | **Recommended, not yet agreed** — see §2.3 |

---

## 1. Thesis and contribution

### 1.1 The question

> Which **single factor change** turns a normal event into an abnormal one?

### 1.2 Why current work cannot answer it

Weakly-supervised VAD produces a scalar anomaly score. It cannot say *why* a clip scored high, and it cannot distinguish "genuinely alarming" from "this particular camera angle is unusual." A model can score well by latching onto scene appearance, camera position, or object frequency — none of which are the event.

### 1.3 The three capabilities required

| # | Capability | Metric | Exists today? |
|---|---|---|---|
| 1 | Tell which factors are present | `edit_acc` — is the true description closer than any of its 22 single-slot edits? | Partially (VADCLIP, Fine-VAD) |
| 2 | Tell which factor *combination* is abnormal | `flip_acc` on matched pairs | No |
| 3 | Show the score changed *because of that factor* | `flip_acc` + `nuis_delta` | No |

**Contribution 3 is the real novelty.** Anyone can swap in a better feature extractor. Almost nobody in VAD can point at a metric that isolates *which factor* drove a score change.

### 1.4 The mechanism

Minimal-intervention (single-slot) contrastive negatives make the video–description alignment sensitive to every individual factor. Without that sensitivity, abilities 2 and 3 cannot be measured at all.

> "Compatibility is not anomaly." — put this in the first paragraph of the introduction.

### 1.5 What this is NOT

- Not a detection-SOTA paper. Weakly-supervised UCF-Crime frame AUC sits at 88–90 (VadCLIP 88.02, STPrompt 88.08, Alert-CLIP 89.32, LAP 90.4). A frozen encoder with 2.3M trainable parameters **will not approach that**. State this up front, not in the risk section, or a failed AUC table reads as a failed capstone.
- Not a new architecture. See §2.1.
- Not a claim about training from scratch. Nothing large is ever trained.

---

## 2. Positioning

### 2.1 The architecture already exists — and it is CLM

The Stanford/NVIDIA **Contrastive Language Model** (Sept 2026, Apache-2.0, 2,989 GitHub stars) and TypeSafe.ai's commercial **Jev** are both "System 1" models: frozen encoder + small projection heads + InfoNCE over a **closed, supplied candidate set**, returning a probability distribution instead of generating tokens.

`crvad/model.py` is structurally the same machine, in a different modality:

| CLM | CRVAD |
|---|---|
| State encoder: frozen Qwen3-8B + trainable projection | `encode_video()`: `in_proj` → temporal transformer → `v_out` |
| Action head over supplied candidates | `encode_text()`: `t_proj` over 1728 hypotheses |
| Bidirectional InfoNCE, diagonal positive | `candidate_nce()`, positive + K sampled negatives |
| Softmax over candidates → distribution | `posterior_scores()`: softmax over 1728 |
| Cannot generate new candidates | Closed 1728 lattice — same limit taken to its extreme |
| Actions cached, encoded once | Text bank precomputed once |

**How to use this:** cite CLM and Jev as System-1 prior art and frame CRVAD as a **transfer of that recipe to video anomaly detection**, not as an invention. "We ported a 2026 System-1 architecture to VAD and it holds up" is a far stronger and more achievable claim than "we designed a head."

- CLM: `Contrastive-LM/CLM-v0.1-8B`, Apache-2.0. **Text-only and encoder-locked** to Qwen3-8B last-token-pooled embeddings — it cannot be used directly. A multimodal CLM-35B was announced for "early October" and **had not shipped as of 12 Oct 2026**. Check again periodically; if it lands, it is both a competitor and a tool.
- Jev: proprietary, no weights. Cite only. arXiv 2609.24395 (third-party evaluation) reports Pearson 0.737 vs VMAF, on par with ITU-T P.1204.1 (0.733). Its **pixel-only variant failed completely** — a direct, citable warning that generic visual features are insufficient, which motivates §5 Gate 2.

### 2.2 Prior art that must be cited and differentiated

| Work | What it does | How we differ |
|---|---|---|
| **CPL** (arXiv 2210.10362) | "Counterfactual by identifying **minimal non-spurious feature change** … learned via contrastive learning" — same conceptual thesis, on CLIP prompt tuning, image tasks | Not video, not anomaly detection; counterfactuals generated on the visual side, not structured text-side slot edits |
| **CRCL** (arXiv 2503.18808, TIP 2025) | Unsupervised VAD, structural causal models, **scene-debiasing** | Closest VAD-side work on scene bias. Our `scene`-slot shortcut risk is the same problem; CRCL is both related work and the hedge against it |
| **CIVC** (Pattern Recognition 2026, doi 10.1016/j.patcog.2026.113809) | Counterfactual + prototype intervention for causally-invariant VAD | **Paywalled, no preprint, `abstract: null` in Crossref/OpenAlex/S2.** Do not characterise its mechanism. Cite only as a counterfactual causal-invariance framework |
| **Alert-CLIP** (CVPR 2026, pp. 35545–35554) | **Fine-tunes** OpenCLIP ViT-L/32. 3 Qwen-VL free-form captions per region caption that reverse normal/abnormal polarity. Margin-softmax loss. **No factor schema, no factor counting** | Our structured factorised lattice is a real gap vs. the closest SOTA hard-negative method |
| **Fine-VAD** (CVPR 2026, pp. 35514–35523) | **Frozen CLIP ViT-B/16 + LGT-Adapter** + multi-granularity text alignment. Deliberately *removes* contrastive loss at the fine level | Architecturally our closest neighbour; required citation and natural baseline. Their warning ("push-pull may disrupt intra-class compactness") is a real risk for `single` negatives |
| **LAVIDA** (CVPR 2026, arXiv 2602.19248) | End-to-end MLLM + **SAM2** + Anomaly Exposure Sampler; irrelevant categories sampled `j ~ Unif(...)`. Zero VAD data | Its "random decoys" are prompt-injected category strings, not contrastive negatives. Baseline, not competitor |
| **CR-VAD?** AdaVid, LAVITI, V-JEPA 2 | Video pretraining backbones | See §4.2 |

**Unoccupied:** no prior art found on **held-out (action, object) compositional splits for VAD**. This is the strongest available novelty axis and should be the headline.

### 2.3 The ablation's missing control

`single` vs `random` is *hard negatives vs. easy negatives* — a known result. It will pass and prove nothing.

**Recommended fourth arm: `hard`** — negatives chosen as the top-K most similar hypotheses in the *current embedding space* (mining at temperature 0 with periodic refresh).

- `single`: shares 3 of 4 slots with the anchor (structured, hard)
- `multi`: changes 2–4 slots (accidentally easy — any single changed slot identifies it)
- `random`: shares almost nothing (easy)
- `hard`: shares whatever the model already confuses (adaptive, matched difficulty)

Only `single` vs `hard` is a real scientific question: *is structured minimality better than adaptive difficulty at the same difficulty?* Without it, G2 is predictable.

This also places the work against **EgoNCE**, EgoVLP's own objective, which "mines egocentric-aware positive and negative samples" — a real, published, non-trivial negative-selection method.

---

## 3. Architecture

```
frozen video encoder  ──►  trainable adapter  ──►  v_t  (unit-norm segment embeddings)
                                                 │
frozen text encoder   ──►  text bank (1728, once) ──►  u_h
                                                 │
                          InfoNCE + margin, negatives per neg_mode   ◄── CONTRIBUTION
                                                 │
   bank_slots ──► slot_emb per slot ──► norm_head ──► a(h) ∈ [0,1]
                                                 │
        p(h|t) = softmax(λ⟨v_t, u_h⟩);   score_t = Σ_h p(h|t)·a(h)
                                                 │
              video score = top-k mean of score_t        (k = T/8)
```

Trainable: `in_proj`, `pos`, `temporal`, `v_out`, `t_proj`, `logit_scale` ≈ **2.25M parameters** (recomputed and confirmed).

### 3.1 The grounding/normality split

The video **never touches the normality head directly**. The head is a pure function of the slot tuple (`model.py:50-53`). The video influences the score through exactly one channel: the posterior `p(h|t)`.

This is the elegant part of the design and it must be preserved. It is also its weakness — see §3.2.

### 3.2 Known architectural gap: the posterior is never supervised

`candidate_nce` supervises a **similarity** (`max_t⟨v_t,u_h⟩`), never the posterior. A model can produce a high anomaly score by concentrating on *any* abnormal hypothesis regardless of pixel content. Nothing in the loss requires `p(h|t)` to be **slot-attributive**.

**Fix (recommended, high leverage):** add a posterior-matching term — KL from `p(h|t)` to a sharpened target over the pseudo-hypothesis, or better, a target concentrated on the true tuple *relative to its 22 single-slot edits*.

This converts "knowing which factor caused the anomaly" from an emergent hope into an optimised objective. It is the single highest-leverage change to the codebase.

### 3.3 Upgrade borrowed from CLM: `Score` instead of sigmoid

CLM returns a **distribution over an ordered scale with stated criteria**, not a scalar. `norm_head`'s sigmoid is a strict special case.

Upgrading to a distribution over something like `[normal, unusual, suspicious, alarming]` with written criteria makes the model say **why**, not just how much. Nearly free, and directly on the interpretability axis this capstone cares about.

---

## 4. Data plan

### 4.1 Roles

| Dataset | Role | Notes |
|---|---|---|
| **EgoClip** (Ego4D, 3.8M clip-text pairs) | Stage 1 training | Pretraining set of EgoVLP. Egocentric — matches the HOI domain. **Requires licence verification** |
| **SSv2** | Stage 1 *evaluation* only | 220,847 clips, 174 templates 1:1 with classes. **Test labels withheld → use val split** |
| **Something-Else annotations** | Compositional split + ROI cropping | 180,049 videos, 16.96M boxes, **assumes 12 fps** |
| **UCA** (arXiv 2309.13925) | Stage 2 supervision + controlled pair set | 23,542 human sentences, 0.1 s temporal grounding, 110.7 h, ~1,854 videos |
| **UCF-Crime train** | Stage 2 training | Weakly-supervised |
| **XD-Violence train** | Stage 2 training | Second scene distribution, limits domain gap |
| **UCF-Crime test / XD test / ShanghaiTech / UBnormal** | Evaluation | Standard protocol |

### 4.2 Encoder candidates

| Candidate | Fits | Risk |
|---|---|---|
| **EgoVLPv2** (arXiv 2307.05463, ICCV 2023) | Egocentric VLP, **cross-modal fusion inside the backbone**, matching text tower. Right for Stage 1 | **Egocentric.** UCF-Crime has Explosion/Arson/Burglary — no visible hand-object interaction. No published result on UCF-Crime |
| **Qwen3-VL-8B** (Apache-2.0, 17.53 GB bf16) | Genuinely general video+text, joint space. Solves the text-degeneracy risk of §5 Gate 1 better than CLIP. Already needed for pseudo-labelling | Extracting clean *segment-level* features from a VLM vision tower is non-standard; `features.py` needs real work |
| **InternVideo2** | Best-in-class | **6B — too large.** Smaller variants exist if needed |
| CLIP ViT-B/16 (current) | Baseline | **4.9% on SSv2 base-to-novel vs 68.9% on K400.** No temporal modelling. Retain only as a baseline row |
| V-JEPA 2 | Self-supervised, no text tower | Cannot do text alignment. Motion-feature probe only |

### 4.3 Why UCA matters

Three deliverables collapse into one:

1. **Stage 2 supervision** — human, not VLM-guessed.
2. **Controlled pair set** — anomalous segment + background segment *from the same video*. Same camera, lighting, style. This deletes the largest manual deliverable in the current doc (staged recordings, two annotators, IAA).
3. **Human audit** (gate G1) — already human.

**UCA does NOT give slots.** It gives free-text sentences + time boundaries. Parsing text → slots is a **text-only** LLM task using the existing `parse_response` from `pseudo_label.py`. Seconds, not hours. Do not confuse this with Qwen3-VL video inference.

---

## 5. The two gates

Both run in week 1. Either can kill the project while it is still cheap.

### Gate 1 — Text-space separability (≈1 hour, no downloads)

Embed all 1728 hypothesis sentences with the candidate text tower. For each tuple, compute cosine similarity to its 22 single-slot edits. Compare that distribution against random pairs.

- **Pass:** single-slot edits are clearly separable from random pairs.
- **Fail:** the text space has lost the information. **No encoder fixes this** — the negatives are *defined* in this space. Stop and redesign the text side.

> Why this is fatal if it fails: a "minimal intervention" only means something if the edit is visible to the model. If all 1728 sentences are near-collinear, a single-slot edit is a no-op and the proposed negative sampler does not exist.

This also produces a motivating figure: "frozen CLIP's text space cannot represent relation flips" — legitimate evidence for why a different encoder is needed.

### Gate 2 — Encoder bake-off (≈2 days)

Extract features for ~500 SSv2 clips and ~500 UCF-Crime windows under 2–3 candidates. Measure, **with no training**:

| Probe | Answers |
|---|---|
| Zero-shot `edit_acc` on SSv2 | Can the encoder see *relations* at all? |
| Zero-shot video AUC on UCF-Crime | Does it transfer to surveillance? |

- **Decision rule:** pick the encoder acceptable on both. EgoVLPv2 at ~0.50 on probe 1 (chance) means Stage 1 is dead — find out in days, not a month.
- **If EgoVLP transfers poorly to surveillance:** use Qwen3-VL's vision tower for Stage 2 and accept losing Stage 1 → Stage 2 weight transfer. The *scientific finding* still stands on its own.

---

## 6. Stage 1 — Relational pretraining

**Purpose:** test whether the negative sampler produces factor-sensitive grounding. Not a general-purpose pretraining stage.

- Normalise hypothesis embeddings → `candidate_nce` with `neg_mode` ∈ {single, multi, random, hard}
- **The normality head is not trained here** (`abnormal_logits()` is never called in `loss_stage1`; `slot_emb` and `norm_head` receive zero gradient and sit at init). This is intentional and matches the doc.
- Add invariance loss if nuisance views are available
- Select checkpoints on a validation split of the training source — never on a test benchmark

**Evaluation:** `edit_acc` overall and per slot, seen vs. held-out (action, object).

**Open decision — Stage 1 supervision:**

| Option | Cost | Verdict |
|---|---|---|
| **A. EgoClip + Qwen3-VL pseudo-labels** | One sharded VLM run (5k windows ≈ 1–4 h wall on 2×A40, ≈$10–40). No manual annotation | **Recommended.** Removes the mapping problem entirely |
| B. SSv2 + 174 hand-written template→tuple mappings | ~1 week manual, plus licence ambiguity | Most templates have 3+ objects, **no `scene` slot exists in SSv2**, so a large fraction would be guesses and `scene` contributes nothing |
| C. Targeted hand-labels (the ~15 genuine antonym template pairs + the "Pretending or failing to…" family, ids 63–83) | ~40 mappings | Trains the `relation` slot only — the only slot `ABNORMAL_RULES` depends on |

**Most valuable unexploited asset:** the **"Pretending or failing to…"** family. A clip that *looks like* an action but does not perform it is a literal counterfactual — visual evidence of a relation change with no change of object or scene. It is the tightest available match to this thesis.

---

## 7. Stage 2 — Weak-label VAD adaptation

Initialise from Stage 1. Train on UCF-Crime + XD-Violence train splits, **video labels only**.

- **MIL** BCE on the top-k segment score ← the normality head learns *only* from weak labels
- **Confidence-weighted InfoNCE** on the pseudo-hypothesis
- **Invariance** term on nuisance views
- `lam_rule = 0` — do not inject a hand-written definition of abnormality into the main results

**Guardrails:**
- Checkpoints selected on a validation split carved from the **training source**. UCF-Crime *test* has frame annotations — using them for any selection decision is leakage.
- Do **not** train on UCA frame labels in the headline run. That would make the task strongly-supervised and the numbers incomparable to the 88–90 band. If a strongly-supervised result is wanted, report it separately and label it.

---

## 8. Evaluation

### 8.1 Standard protocol

Train on UCF-Crime + XD-Violence train. Report:

| Test set | Purpose |
|---|---|
| UCF-Crime test | **The comparable number** (band: 88–90) |
| XD-Violence test | Second comparable number |
| ShanghaiTech | Cross-scene generalisation |
| UBnormal | Cross-domain, fully synthetic |

**Do not** drop the UCF-Crime test set in favour of only cross-dataset testing — that destroys comparability with every published number.

⚠️ **Metric trap:** ShanghaiTech figures of 97–98 in the CLIP-VAD literature come from VadCLIP/STPrompt's WSVAD re-split (238/199), **not** the classic semi-supervised protocol (~70–86). Never mix them in one table.

### 8.2 New metrics — the actual contribution

| Metric | Definition |
|---|---|
| `edit_acc` | P[s(x, h_true) > s(x, h_edit)] over all 22 single-slot edits, per slot |
| `flip_acc` | On matched pairs, P[score(abnormal) > score(normal)] |
| `flip_acc_nuis` | Same, on nuisance-augmented views |
| `nuis_delta` | E\|score(x) − score(x̃)\| |
| compositional split | All of the above, seen vs held-out (action, object) |

### 8.3 Metrics that do not exist yet and should

| New metric | Why |
|---|---|
| **Specificity** (non-flipping edits) | `make_pairs` currently builds only pairs that *flip* normality. Add pairs where the edit does **not** change normality — the score should stay flat. Without this the model can win `flip_acc` by raising every score |
| **Slot-ablation shortcut probe** | Hold the video, swap one slot in the tuple, measure score delta. If swapping `scene` moves the score a lot, the model has learned a scene shortcut. Directly tests the CRCL failure mode |
| **`hard` negative arm** | §2.3 |

### 8.4 Baselines the doc promises but code lacks

None of these exist in `crvad/` today:

- Text-only (no video) — hypothesis-only bias
- Shuffled-video — video-label-only bias
- Zero-shot CLIP similarity floor
- No-normality-head (compatibility only) — the "compatibility is not anomaly" demonstration
- No-invariance-loss ablation

---

## 9. File-level changes

### `crvad/features.py`
- **BUG:** the window loop `range(0, max(1, len(fr) - per + 1), per)` **silently drops the final partial window** of every video. On surveillance this is frequently where the anomaly sits. Fix first.
- Make the encoder configurable (`--encoder egovlpv2 | qwen3vl | clip`) rather than hardcoded `CLIP = "openai/clip-vit-base-patch16"`
- Optional Something-Else box cropping → ROI before encoding (exploits CLIP's known object bias)

### `crvad/losses.py`
- Add `hard` negative mode (embedding-space top-K, periodic refresh)
- Add posterior-attribution loss (§3.2)
- **BUG:** nothing validates `-1` slot values. `flat_index` yields negative indices and PyTorch **silently wraps them** (`text_emb[-122]` returns a valid, wrong hypothesis)

### `crvad/train.py`
- **BUG:** `loss_stage2` validates only `slots[:, 0] >= 0` (`train.py:77`). Any other slot at `-1` produces a wrapped index. Validate all four
- Support multi-source Stage 2 (concatenate UCF + XD into one `.pt`)
- Clamp `logit_scale` earlier — it is only clamped at 100, allowing an over-peaked posterior that destabilises MIL

### `crvad/model.py`
- `norm_head`: optional `Score`-style distribution over an ordered abnormality scale (§3.3)
- Expose `p(h|t)` for the attribution loss

### `crvad/pseudo_label.py`
- Move to **Qwen3-VL-8B-Instruct** (Apache-2.0, 17.53 GB). The current `Qwen2-VL-7B-Instruct` is superseded
- Sample N answers in a **loop with `do_sample=True`**, not `num_return_sequences` — beam-style multi-return gives near-duplicate captions, useless as contrastive negatives. (`num_return_sequences>1` with video was also a real transformers bug, fixed Nov 2025)
- **BUG:** `parse_response` uses non-greedy `re.search(r"\{.*?\}", text)` — breaks on nested braces or two JSON objects. Use `json.JSONDecoder().raw_decode`
- **Add:** a text-only UCA-sentence → slot parser (reuses `parse_response`)

### `crvad/evaluate.py`
- Add specificity, slot-ablation probe, and the four baselines in §8.4
- `evaluate()` currently requires `pairs["nuis0"]`/`["nuis1"]` unconditionally — `KeyError` if nuisance views absent

### New files
```
crvad/probes/text_separability.py   Gate 1
crvad/probes/encoder_bakeoff.py     Gate 2
crvad/build/datasets.py             Stage 1/2 .pt assembly
crvad/build/uca_pairs.py            controlled pair mining from UCA
```

### Unchanged
`crvad/slots.py` (schema + edit enumeration), `crvad/data.py`, the shape of `model.py`, and the whole metric suite are all reusable as-is. The `data.py` abstraction of `(N,T,D)` is what makes the encoder swap cheap.

---

## 10. Milestones

| Weeks | Work | Exit criterion |
|---|---|---|
| **1** | **Gate 1** text separability. **Gate 2** encoder bake-off. Download EgoClip/UCF-Crime | Both gates passed; encoder chosen |
| **2** | EgoVLPv2 (or Qwen3-VL) feature extraction, two views. Build text bank | `feats`/`nuis_feats` caches exist for both sources |
| **3–4** | Qwen3-VL pseudo-labelling → `stage1_train.pt`. Parse UCA → `stage2_train.pt` | Coverage + confidence distributions reported |
| **4** | **Gate G1:** human audit, 300–500 windows | Per-slot agreement; drop slots below 0.7 |
| **5** | Build controlled pair set from UCA. Run text-only + shuffled baselines | Pair set passes both checks at near chance |
| **6** | Stage 1 training. Per-slot `edit_acc`, seen vs held-out | Held-out edit accuracy clearly above zero-shot |
| **7** | **Gate G2 (run EARLY, on a subset):** negative-mode ablation | See §11 |
| **8–9** | Stage 2. Full ablation grid, ≥3 seeds, matched volume and steps | Complete checkpoint set |
| **10** | Remaining ablations (no-invariance, no-normality-head, `hard` arm) | Full table |
| **11** | All benchmarks. Text bank separability measured on the *final* encoder | Comparable numbers reported |
| **12** | Failure analysis, shortcut probe, write-up | — |
| **13–14** | Reproducibility check, code cleanup, report | Regenerated by one script |

---

## 11. Risks and failure modes

| # | Risk | Threatens contribution? | Response |
|---|---|---|---|
| 1 | **Text space degenerate** (Gate 1 fails) | **Yes — fatal** | Stop. Redesign text side. Do not proceed |
| 2 | **`single` does not beat `multi`/`random`** | **Yes** | G2 gate. Pivot to a compositional-grounding study (already written into the doc) |
| 3 | EgoVLP egocentric → surveillance | **No** | Separate claim, separate evidence. Swap encoder for Stage 2 |
| 4 | SSv2 licence (Qualcomm research vs. community Apache-2.0 claim) | Blocks Option B | Resolve before relying on SSv2. Favors EgoClip (Option A) |
| 5 | `single` negatives disrupt intra-class compactness | Partially | Fine-VAD removed contrastive loss for exactly this reason. Watch `flip_acc` on non-flipping edits |
| 6 | Scene shortcut (`parking lot` → abnormal) | Partially | Slot-ablation probe (§8.3); CRCL as reference |
| 7 | Pseudo-label noise | Partially | Confidence weighting, human audit, second VLM as cross-check |
| 8 | Benchmarks show no gain | **No** | By design. Intervention and compositional metrics carry the contribution |
| 9 | Schema ceiling — closed schema cannot express all anomalies | Partially | State as an explicit scope limitation |
| 10 | Multimodal CLM-35B ships mid-project | Partially | Monitor. Could be a tool *or* a competitor |

---

## 12. Corrections required to `crvad_pipeline.tex`

These were verified during planning and the existing doc is wrong on each point.

| Doc claim | Correct position |
|---|---|
| Alert-CLIP uses "curated semantic hard negatives" changing 2–4 semantic factors | **Refuted.** MLLM free-form caption polarity inversion, 3 per region caption, **no factor schema, no factor counting** |
| LAVIDA-style negatives as "uniformly random decoys" | **Correct**, with nuance: prompt-injected irrelevant category strings, not contrastive negatives; end-to-end with SAM2 on segmentation data |
| "Nothing-Else Matters", Mottaghi et al., ECCV 2022 | **Does not exist.** Canonical protocol is **Something-Else**, Materzynska et al., CVPR 2020, arXiv 1912.09930 |
| Something-Else = verb–object pair labels | It provides **dense per-frame bounding boxes** (180,049 videos, 16.96M boxes, **assumes 12 fps**). `gt_placeholders` recovers nouns |
| "~20–25% held-out verb–object pairs" as the Something-Else protocol | Not the protocol. It is a 50/50 flip of two noun groups × two action groups → **54,919 train / 57,876 val** |
| `slots.compositional_split()` is "Something-Else style" | It is a reasonable **original** design, not the published protocol. Rename or re-attribute |
| SSv2 example: "Putting something on top of something else" / "Taking something off something else" | **Neither string exists in SSv2.** Real pair: `Covering something with something` (id 6) ↔ `Uncovering something` (id 171). ~15 genuine antonym pairs exist |
| VLM captioner = Qwen2-VL-7B-Instruct | **Superseded** by Qwen3-VL-8B-Instruct (Apache-2.0). Video throughput is **unpublished — must be measured on 50 windows** |
| SSv2 ≈220k clips | Correct (220,847). Add: 19.4 GB, VP9/webm, 174 templates 1:1 with classes, **test labels withheld** |
| Implicit: CLIP is weak for VAD | **False.** VadCLIP and CLIP-TSA beat SOTA with CLIP features. Sell "frozen + small heads" on **cost and ablation cleanliness**. The correct citable gap is AnomalyCLIP's: *"CLIP's original pre-training focuses on aligning with object semantics instead of the abnormality and normality within images"* |

---

## 13. Report outline

1. **Introduction** — "compatibility is not anomaly"; which factor made this abnormal?
2. **Related work** — System-1 models (CLM, Jev); CLIP-VAD (VadCLIP, Fine-VAD, Alert-CLIP); counterfactual/causal VAD (CPL, CRCL, CIVC)
3. **Method** — slot schema; grounding vs. normality separation; the three negative modes (+ `hard`); posterior-attribution term
4. **Benchmark protocol** — data, splits, no test-set selection, multi-source training
5. **Results** — Stage 1 compositional table; Stage 2 benchmark table; **negative-mode ablation as the headline**
6. **Intervention analysis** — flip, specificity, nuisance stability, slot-ablation
7. **Limitations** — schema ceiling; pseudo-label noise; encoder domain gap; weak-label protocol
8. **Failure analysis** — including the G2 gate outcome and any pivot

---

## 14. Open decisions requiring sign-off

1. **Add the `hard` negative arm?** Without it the central ablation is predictable. *Recommended: yes.*
2. **Posterior-attribution loss?** Highest-leverage code change; makes ability 3 trainable rather than merely measured. *Recommended: yes.*
3. **Stage 1 supervision** — EgoClip + pseudo-labels (recommended), SSv2 hand-mapped, or targeted ~40 hand-labels.
4. **Text encoder** — EgoVLP's text tower or Qwen3-VL's? Depends on Gate 1.
5. **`Score` head** (§3.3) — in scope for this capstone or deferred?