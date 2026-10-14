# CRVAD — Work Log and Findings

Companion to `PLAN.md`. That document is the *plan*; this one records **what was actually
built, run, and measured**, including the things that failed.

Last updated: 13 October 2026
Status: Gate 1 passed · Gate 2 **inconclusive / measurement-limited**

---

## 1. What was built

Four executable probes under `crvad/probes/`, each designed to answer one question and
each able to kill the project cheaply if the answer is bad.

| File | Question | Cost |
|---|---|---|
| `text_separability.py` | **Gate 1** — does the *text* space separate single-slot edits? | 57 s, CPU |
| `ssv2_data.py` | Build a minimal-intervention SSv2 clip set | 262 s, CPU |
| `encoder_bakeoff.py` | **Gate 2a** — zero-shot: can the *video* side separate them? | ~53 min, CPU |
| `linear_probe.py` | **Gate 2b** — is the relation *linearly decodable* from frozen features? | ~71 min first run, then seconds |

Supporting changes: `.gitignore` extended for `runs/`, `data/`, `*.pt`, `hf_cache/`.

---

## 2. Results

### Gate 1 — text space: **PASS**

`openai/clip-vit-base-patch16` text tower, 1728 hypothesis sentences, 38,016
single-slot-edit pairs vs 110,592 random pairs.

| | |
|---|---|
| similarity, single-slot edit | 0.9226 ± 0.0401 |
| similarity, random pair | 0.7507 ± 0.0732 |
| gap | 0.1720 |
| Cohen's *d* | 2.91 |
| **AUC edit-vs-random** | **0.9793** → GREEN |

Per slot:

| slot | mean | AUC | *d* |
|---|---|---|---|
| action | 0.9569 | 0.9972 | 3.79 |
| **relation** ← `ABNORMAL_RULES` depends on this | 0.9338 | **0.9906** | 3.29 |
| object | 0.8982 | 0.9690 | 2.65 |
| scene | 0.8880 | 0.9535 | 2.42 |

**Interpretation.** Template bias *is* present (all 1728 sentences sit at mean cosine
0.7527, std 0.0755) but the distributions are cleanly separable at *d* ≈ 2.9. The
fatal risk — a collapsed text space where a "minimal intervention" is a no-op — did
not materialise. `relation`, the slot the abnormality rules hinge on, is the
second-best represented.

**Secondary finding.** An additive slot-composition bank (26 per-slot phrases summed
rather than 1728 rendered sentences) gives a slightly *better* AUC (0.9950) but
compresses everything into a 0.97–0.99 band (gap 0.0242 vs 0.1720). At τ=0.07
(λ≈14.3) the sentence bank gives ~2.5 logits of separation between an edit and a
random hypothesis; the additive bank gives ~0.35. **Use the plain sentence bank.**
The fallback is not needed — one fewer thing to build.

### Gate 2a — zero-shot: **negative**

264 clips, 22 usable minimal-intervention pairs, 286 tests. No training.

| encoder | edit_acc | 95% CI | template retrieval | chance |
|---|---|---|---|---|
| CLIP | 0.622 | [0.566, 0.678] | 0.045 | 0.024 |
| X-CLIP | 0.552 | [0.497, 0.612] | **0.015** | 0.024 |

Both CI lower bounds sit at or below 0.500. Per-pair cosine margins are ±0.001–0.004 —
the model is not *barely* preferring the right caption, it is producing two
near-identical numbers and winning on noise. Well-populated pairs cluster at
0.54–0.67; `Folding vs Unfolding` (n=13) sits at **0.462**, i.e. below chance.

X-CLIP's retrieval being *below* chance looked like a wiring bug, so it was
investigated (prompt templates, normalisation, frame count, video/text space
alignment). No wiring bug found. See §4.6.

### Gate 2b — linear probe: **underdetermined, not negative**

5-fold CV, logistic head, 41→26 usable classes (classes with <5 clips dropped).

| encoder | probe acc | shuffled control | gain |
|---|---|---|---|
| CLIP | 0.018 | 0.041 | **−0.023** |
| X-CLIP | 0.009 | 0.009 | +0.000 |

The probe fails to beat its own permutation control. But this **should not be read as
"no signal"**:

- **217 samples, 26 classes, 512 dims.** The head has more parameters than data.
- Probe scored **0.018 against a chance of 0.038** — *half* of chance — while the
  shuffled control sat at 0.041, right on chance. That asymmetry is the signature of a
  badly underdetermined fit, not of absent signal.
- **A direct feature-space check contradicts it.** Computing class-centroid separation
  on the cached CLIP features gives a **mean pairwise *d* of 12.04 across 210
  comparable template pairs** — i.e. the classes are not overlapping at all. (The
  absolute scale is inflated because the divisor was `std = 0.0086`; treat it as
  "very separated", not a calibrated effect size.)

**Conclusion: the two measurements disagree and the probe is the less trustworthy of
the two.** The honest summary of Gate 2 is *"underdetermined"*, not *"no signal"*.

### Data set

| | |
|---|---|
| clips | 264 × 16 frames × 224², uint8 |
| templates matched | **41/41** |
| captions matching no template | 6,319 / 124,032 (**5.1%**) |
| cache size / build time | 606 MB / 262 s |
| clips per template | min **1**, median **6**, max 12 |
| templates with ≥5 clips | 26 of 41 |

**The distribution is badly skewed** and this is the root of the Gate 2b problem. Only
264 of the 1,000 episodes in `chunk-000/file-000` belong to the 41 target templates;
the remaining 736 belong to other SSv2 templates. `--limit-per-template 12` was never
the binding constraint — availability was.

---

## 3. Environment

Local venv at `D:\SNU\Capstone\our-model\.venv`.

```
torch 2.14.1+cpu · transformers 5.19.0 · polars 2.0.0 · av 19.0.1 · huggingface_hub
```

**Constraints discovered:**

| Constraint | Consequence |
|---|---|
| **No GPU on this machine** | All four probes run on CPU. Verified: nothing in the probe path needs a GPU. |
| **pyarrow blocked by Windows AppLoader** | `ImportError: DLL load failed`. Parquet reading rewritten to use **polars**. torch's own DLLs load fine, so the block is targeted, not blanket. |
| **Console is cp1252** | Non-ASCII captions and polars' box-drawing repr raise `UnicodeEncodeError`. stdout/stderr now reconfigured to UTF-8. |
| Kaggle 2× T4 (16 GB) available | Reserved for Qwen3-VL inference and bulk extraction. **Not needed for any probe.** |
| T4 has no bf16 (Turing) | `configs/default.yaml` `amp: true` + `train.py:139` hardcoded `torch.bfloat16` will crash. Must switch to fp16. |

Data source: `n3puiol/something_something_v2_lerobot` — **Apache-2.0** (resolves the
Qualcomm licence ambiguity on the original SSv2 release) and chunked, so one 130 MB
mp4 yields 1,000 clips instead of a 19.4 GB download.

---

## 4. Bugs found and fixed

Recorded because each one silently produced a wrong number rather than an error.

1. **`transformers ≥ 5` API break.** `get_text_features` returns
   `BaseModelOutputWithPooling`, not a tensor. `features.py:36` and `:60` will both
   raise `AttributeError` on first run. Fixed in the probes via a `_unwrap` helper;
   **the main `features.py` is still unpatched.**

2. **Captions live in the parquet index, not a column.** `tasks.parquet` has one column
   named `task_index` containing *integers*; the caption strings are the serialised
   pandas index (`__index_level_0__`). Reading the column returned `['0','1','2',...]`
   and matched **0/41** templates with no error raised.

3. **`frame_index` is per-episode, not global.** It resets to 0 in every episode, so
   keying a frame→episode map by it made 264 episodes collide onto ~70 shared keys.
   **264 episodes silently collapsed to 6 clips.** Fixed by auto-detecting the global
   offset column (`index`, verified by strict monotonicity in row order) plus a
   collision guard that now raises instead of degrading quietly.

4. **`.view(-1, 16, -1)`** — two inferred dimensions. torch can infer only one.
   `RuntimeError: only one dimension can be inferred`. CLIP row crashed entirely.

5. **`re.split` with a capturing group** returns the capture as an extra element,
   injecting the literal word `"thing"` into every generated regex.

6. **Wrong placeholder regex.** Wrote `\bsome\s+thing\b`, which requires whitespace —
   but `"something"` is a single word. Every template stopped matching.

7. **`CONTRASTS` was not a bijection.** `Closing something` legitimately contrasts with
   both `Opening something` and `Pretending to close something…`. A single-partner dict
   dropped a contrast and would have crashed evaluation. Now a list; evaluation
   iterates pairs with both directions pooled.

8. **Bootstrap on a constant-expanded tensor** — not a real bootstrap. Now collects one
   observation per (clip, pair) test.

9. **X-CLIP uses ImageNet normalisation, not CLIP's** (0.485/0.456/0.406, 0.229/0.224/
   0.225 vs CLIP's 0.481/0.458/0.408, 0.269/0.261/0.276). Feeding CLIP's constants
   would have degraded X-CLIP and made video pretraining look worse than it is.

10. **Linear probe fit on the training fold, then indexed those logits with the test
    mask.** Shape error caught by a synthetic test; `logreg` now returns `(W, b)` so
    the held-out fold is scored by a model that never saw it.

---

## 5. Method decisions worth keeping

- **The probe must be run at the *template* level, not the 1728-slot level.** Mapping
  all 174 SSv2 templates to slot tuples is days of annotation and is *not needed for a
  gate*. `edit_acc` restricted to pairs where the single-factor edit is unambiguous
  answers the same question with zero manual mapping.
- **Longest-match-wins resolves prefix collisions.** Matching against all 174 official
  templates (not just the 41 targets) means `"pouring milk into a cup until it
  overflows"` correctly hits template 60 rather than 59, because 60's regex is longer
  and therefore more specific. Verified by test.
- **Zero-shot cosine measures prompt sensitivity, not information content.** A frozen
  encoder can score near chance zero-shot and still hold a linearly decodable relation
  — which is exactly what the temporal adapter in `model.py` exists to exploit. Gate
  2a alone cannot justify discarding an encoder.
- **The 22 minimal-intervention pairs** lean heavily on the *"Pretending or failing
  to…"* family (13 of 22), where the only difference between clips is whether the
  relation is actually performed. That is the tightest available test of the thesis.

---

## 6. Timing corrections

I asserted runtime estimates before measuring and was wrong twice. Recording them so
the estimates aren't trusted uncritically next time.

| Task | Predicted | Actual |
|---|---|---|
| Gate 1 text separability | "~1 hour" | **57 s** |
| Bakeoff, CLIP | 5–15 min | **53 min** (both encoders) |
| Linear probe | ~25 min/encoder | **71 min** (both, first run) |

Encoding dominates everything. Feature caching (`runs/feats/*.pt`) was added so
probes re-run in seconds — this is the single most important cost control, and the
reason iteration was painful before it.

---

## 7. Open items

**Uncommitted.** `git status` shows `crvad/probes/{ssv2_data,encoder_bakeoff}.py`
modified and `linear_probe.py` untracked. `origin/main` has the caption fixes but
**not** the frame-offset fix, X-CLIP, the EgoVLPv2 loader, or the probe — a stale
Kaggle checkout has already cost two debugging cycles.

**Immediate next step (seconds, features already cached):** re-run the probe restricted
to the ~15 templates with ≥20 clips — a 15-way problem instead of 41-way, far more data
per class. If that clears chance comfortably, Gate 2b was data-starved and the answer
was always "more clips". If it still cannot beat its own control, the negative is real
and EgoVLPv2 becomes justified.

**Then, if more data is needed:** each `--file N` is 4.4 min and ~130 MB. Five files →
~1,300 clips at median ~30 per template, which is the regime where a linear probe means
something.

**Also outstanding:**

- `features.py:36,60` still has the transformers-5 bug and will crash on first run.
- EgoVLPv2 is **not pip-installable** (no `setup.py`, archived, MIT). Its
  `requirements.txt` pins `torch==1.13.1+cu116` and `numpy==1.23.5`, neither of which
  has Python 3.13 wheels — do **not** install it. Correct path is
  `git clone` + `pip install timm einops yacs` + `EGOVLP_REPO` env var. The loader in
  `encoder_bakeoff.py` is written but untested.
- The UCF-Crime probe (does the encoder transfer to surveillance?) has not been run. It
  is the second half of Gate 2 and the one that decides the Stage 2 encoder.

---

## 8. Prior-art corrections to `crvad_pipeline.tex`

Verified during research; the existing document is wrong on these points.

| Claim | Correct position |
|---|---|
| Alert-CLIP uses hard negatives changing 2–4 semantic factors | **Refuted.** MLLM free-form caption polarity inversion, 3 per region caption, **no factor schema, no factor counting** |
| LAVIDA-style negatives are random decoys | **Correct**, with nuance: prompt-injected category strings, not contrastive negatives |
| "Nothing-Else Matters", Mottaghi, ECCV 2022 | **Does not exist.** Canonical protocol is **Something-Else**, Materzynska et al., **CVPR 2020**, arXiv 1912.09930 |
| Something-Else gives verb–object pairs | It gives **dense per-frame bounding boxes** (180,049 videos, 16.96M boxes, **assumes 12 fps**) |
| ~20–25% held-out verb–object pairs | Not the protocol. 50/50 flip of two noun × two action groups → **54,919 train / 57,876 val** |
| SSv2 put/take example | **Neither string exists in SSv2.** Real pair: `Covering something with something` (6) ↔ `Uncovering something` (171) |
| Implicit: CLIP is weak for VAD | **False.** VadCLIP and CLIP-TSA beat SOTA with CLIP features. The citable gap is AnomalyCLIP's: *"CLIP's original pre-training focuses on aligning with object semantics instead of the abnormality and normality within images"* |

Newly relevant: **CLM** (Contrastive Language Model, Stanford/NVIDIA, Apache-2.0, 2,989
GitHub stars) and **Jev** (TypeSafe.ai, arXiv 2609.24395) are "System 1" models —
frozen encoder + small projection heads + InfoNCE over a *closed, supplied candidate
set*, returning a probability distribution instead of generating tokens.
`crvad/model.py` is structurally the same machine in a different modality, so CRVAD
should be framed as a **transfer of that recipe to VAD**, not as an invented
architecture. CLM is text-only and encoder-locked to Qwen3-8B embeddings; a multimodal
CLM-35B was announced for "early October" and had not shipped as of 12 Oct 2026.