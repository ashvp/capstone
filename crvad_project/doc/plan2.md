# CRVAD ‚Äî Work Log and Findings

Companion to `PLAN.md`. That document is the *plan*; this one records **what was actually
built, run, and measured**, including the things that failed.

Last updated: 14 October 2026
Status: Gate 1 passed (on a placeholder schema) ¬∑ Gate 2 **negative across 3 encoders √ó 2 machines**

> **Read ¬ß9 before using any number in ¬ß2.** Sections 2‚Äì8 record the first Gate-2 run and
> its bugs. Every Gate-2 figure in them is **void** ‚Äî the cached features were degenerate.
> ¬ß9 supersedes them.

---

## 1. What was built

Four executable probes under `crvad/probes/`, each designed to answer one question and
each able to kill the project cheaply if the answer is bad.

| File | Question | Cost |
|---|---|---|
| `text_separability.py` | **Gate 1** ‚Äî does the *text* space separate single-slot edits? | 57 s, CPU |
| `ssv2_data.py` | Build a minimal-intervention SSv2 clip set | 262 s, CPU |
| `encoder_bakeoff.py` | **Gate 2a** ‚Äî zero-shot: can the *video* side separate them? | ~53 min, CPU |
| `linear_probe.py` | **Gate 2b** ‚Äî is the relation *linearly decodable* from frozen features? | ~71 min first run, then seconds |

Supporting changes: `.gitignore` extended for `runs/`, `data/`, `*.pt`, `hf_cache/`.

---

## 2. Results

### Gate 1 ‚Äî text space: **PASS**

`openai/clip-vit-base-patch16` text tower, 1728 hypothesis sentences, 38,016
single-slot-edit pairs vs 110,592 random pairs.

| | |
|---|---|
| similarity, single-slot edit | 0.9226 ¬± 0.0401 |
| similarity, random pair | 0.7507 ¬± 0.0732 |
| gap | 0.1720 |
| Cohen's *d* | 2.91 |
| **AUC edit-vs-random** | **0.9793** ‚Üí GREEN |

Per slot:

| slot | mean | AUC | *d* |
|---|---|---|---|
| action | 0.9569 | 0.9972 | 3.79 |
| **relation** ‚Üê `ABNORMAL_RULES` depends on this | 0.9338 | **0.9906** | 3.29 |
| object | 0.8982 | 0.9690 | 2.65 |
| scene | 0.8880 | 0.9535 | 2.42 |

**Interpretation.** Template bias *is* present (all 1728 sentences sit at mean cosine
0.7527, std 0.0755) but the distributions are cleanly separable at *d* ‚âà 2.9. The
fatal risk ‚Äî a collapsed text space where a "minimal intervention" is a no-op ‚Äî did
not materialise. `relation`, the slot the abnormality rules hinge on, is the
second-best represented.

**Secondary finding.** An additive slot-composition bank (26 per-slot phrases summed
rather than 1728 rendered sentences) gives a slightly *better* AUC (0.9950) but
compresses everything into a 0.97‚Äì0.99 band (gap 0.0242 vs 0.1720). At œÑ=0.07
(Œª‚âà14.3) the sentence bank gives ~2.5 logits of separation between an edit and a
random hypothesis; the additive bank gives ~0.35. **Use the plain sentence bank.**
The fallback is not needed ‚Äî one fewer thing to build.

### Gate 2a ‚Äî zero-shot: **negative**

264 clips, 22 usable minimal-intervention pairs, 286 tests. No training.

| encoder | edit_acc | 95% CI | template retrieval | chance |
|---|---|---|---|---|
| CLIP | 0.622 | [0.566, 0.678] | 0.045 | 0.024 |
| X-CLIP | 0.552 | [0.497, 0.612] | **0.015** | 0.024 |

Both CI lower bounds sit at or below 0.500. Per-pair cosine margins are ¬±0.001‚Äì0.004 ‚Äî
the model is not *barely* preferring the right caption, it is producing two
near-identical numbers and winning on noise. Well-populated pairs cluster at
0.54‚Äì0.67; `Folding vs Unfolding` (n=13) sits at **0.462**, i.e. below chance.

X-CLIP's retrieval being *below* chance looked like a wiring bug, so it was
investigated (prompt templates, normalisation, frame count, video/text space
alignment). No wiring bug found. See ¬ß4.6.

### Gate 2b ‚Äî linear probe: **underdetermined, not negative**

5-fold CV, logistic head, 41‚Üí26 usable classes (classes with <5 clips dropped).

| encoder | probe acc | shuffled control | gain |
|---|---|---|---|
| CLIP | 0.018 | 0.041 | **‚àí0.023** |
| X-CLIP | 0.009 | 0.009 | +0.000 |

The probe fails to beat its own permutation control. But this **should not be read as
"no signal"**:

- **217 samples, 26 classes, 512 dims.** The head has more parameters than data.
- Probe scored **0.018 against a chance of 0.038** ‚Äî *half* of chance ‚Äî while the
  shuffled control sat at 0.041, right on chance. That asymmetry is the signature of a
  badly underdetermined fit, not of absent signal.
- **A direct feature-space check contradicts it.** Computing class-centroid separation
  on the cached CLIP features gives a **mean pairwise *d* of 12.04 across 210
  comparable template pairs** ‚Äî i.e. the classes are not overlapping at all. (The
  absolute scale is inflated because the divisor was `std = 0.0086`; treat it as
  "very separated", not a calibrated effect size.)

**Conclusion: the two measurements disagree and the probe is the less trustworthy of
the two.** The honest summary of Gate 2 is *"underdetermined"*, not *"no signal"*.

### Data set

| | |
|---|---|
| clips | 264 √ó 16 frames √ó 224¬≤, uint8 |
| templates matched | **41/41** |
| captions matching no template | 6,319 / 124,032 (**5.1%**) |
| cache size / build time | 606 MB / 262 s |
| clips per template | min **1**, median **6**, max 12 |
| templates with ‚â•5 clips | 26 of 41 |

**The distribution is badly skewed** and this is the root of the Gate 2b problem. Only
264 of the 1,000 episodes in `chunk-000/file-000` belong to the 41 target templates;
the remaining 736 belong to other SSv2 templates. `--limit-per-template 12` was never
the binding constraint ‚Äî availability was.

---

## 3. Environment

Local venv at `D:\SNU\Capstone\our-model\.venv`.

```
torch 2.14.1+cpu ¬∑ transformers 5.19.0 ¬∑ polars 2.0.0 ¬∑ av 19.0.1 ¬∑ huggingface_hub
```

**Constraints discovered:**

| Constraint | Consequence |
|---|---|
| **No GPU on this machine** | All four probes run on CPU. Verified: nothing in the probe path needs a GPU. |
| **pyarrow blocked by Windows AppLoader** | `ImportError: DLL load failed`. Parquet reading rewritten to use **polars**. torch's own DLLs load fine, so the block is targeted, not blanket. |
| **Console is cp1252** | Non-ASCII captions and polars' box-drawing repr raise `UnicodeEncodeError`. stdout/stderr now reconfigured to UTF-8. |
| Kaggle 2√ó T4 (16 GB) available | Reserved for Qwen3-VL inference and bulk extraction. **Not needed for any probe.** |
| T4 has no bf16 (Turing) | `configs/default.yaml` `amp: true` + `train.py:139` hardcoded `torch.bfloat16` will crash. Must switch to fp16. |

Data source: `n3puiol/something_something_v2_lerobot` ‚Äî **Apache-2.0** (resolves the
Qualcomm licence ambiguity on the original SSv2 release) and chunked, so one 130 MB
mp4 yields 1,000 clips instead of a 19.4 GB download.

---

## 4. Bugs found and fixed

Recorded because each one silently produced a wrong number rather than an error.

1. **`transformers ‚â• 5` API break.** `get_text_features` returns
   `BaseModelOutputWithPooling`, not a tensor. `features.py:36` and `:60` will both
   raise `AttributeError` on first run. Fixed in the probes via a `_unwrap` helper;
   **the main `features.py` is still unpatched.**

2. **Captions live in the parquet index, not a column.** `tasks.parquet` has one column
   named `task_index` containing *integers*; the caption strings are the serialised
   pandas index (`__index_level_0__`). Reading the column returned `['0','1','2',...]`
   and matched **0/41** templates with no error raised.

3. **`frame_index` is per-episode, not global.** It resets to 0 in every episode, so
   keying a frame‚Üíepisode map by it made 264 episodes collide onto ~70 shared keys.
   **264 episodes silently collapsed to 6 clips.** Fixed by auto-detecting the global
   offset column (`index`, verified by strict monotonicity in row order) plus a
   collision guard that now raises instead of degrading quietly.

4. **`.view(-1, 16, -1)`** ‚Äî two inferred dimensions. torch can infer only one.
   `RuntimeError: only one dimension can be inferred`. CLIP row crashed entirely.

5. **`re.split` with a capturing group** returns the capture as an extra element,
   injecting the literal word `"thing"` into every generated regex.

6. **Wrong placeholder regex.** Wrote `\bsome\s+thing\b`, which requires whitespace ‚Äî
   but `"something"` is a single word. Every template stopped matching.

7. **`CONTRASTS` was not a bijection.** `Closing something` legitimately contrasts with
   both `Opening something` and `Pretending to close something‚Ä¶`. A single-partner dict
   dropped a contrast and would have crashed evaluation. Now a list; evaluation
   iterates pairs with both directions pooled.

8. **Bootstrap on a constant-expanded tensor** ‚Äî not a real bootstrap. Now collects one
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
  ‚Äî which is exactly what the temporal adapter in `model.py` exists to exploit. Gate
  2a alone cannot justify discarding an encoder.
- **The 22 minimal-intervention pairs** lean heavily on the *"Pretending or failing
  to‚Ä¶"* family (13 of 22), where the only difference between clips is whether the
  relation is actually performed. That is the tightest available test of the thesis.

---

## 6. Timing corrections

I asserted runtime estimates before measuring and was wrong twice. Recording them so
the estimates aren't trusted uncritically next time.

| Task | Predicted | Actual |
|---|---|---|
| Gate 1 text separability | "~1 hour" | **57 s** |
| Bakeoff, CLIP | 5‚Äì15 min | **53 min** (both encoders) |
| Linear probe | ~25 min/encoder | **71 min** (both, first run) |

Encoding dominates everything. Feature caching (`runs/feats/*.pt`) was added so
probes re-run in seconds ‚Äî this is the single most important cost control, and the
reason iteration was painful before it.

---

## 7. Open items

**Uncommitted.** `git status` shows `crvad/probes/{ssv2_data,encoder_bakeoff}.py`
modified and `linear_probe.py` untracked. `origin/main` has the caption fixes but
**not** the frame-offset fix, X-CLIP, the EgoVLPv2 loader, or the probe ‚Äî a stale
Kaggle checkout has already cost two debugging cycles.

**Immediate next step (seconds, features already cached):** re-run the probe restricted
to the ~15 templates with ‚â•20 clips ‚Äî a 15-way problem instead of 41-way, far more data
per class. If that clears chance comfortably, Gate 2b was data-starved and the answer
was always "more clips". If it still cannot beat its own control, the negative is real
and EgoVLPv2 becomes justified.

**Then, if more data is needed:** each `--file N` is 4.4 min and ~130 MB. Five files ‚Üí
~1,300 clips at median ~30 per template, which is the regime where a linear probe means
something.

**Also outstanding:**

- `features.py:36,60` still has the transformers-5 bug and will crash on first run.
- EgoVLPv2 is **not pip-installable** (no `setup.py`, archived, MIT). Its
  `requirements.txt` pins `torch==1.13.1+cu116` and `numpy==1.23.5`, neither of which
  has Python 3.13 wheels ‚Äî do **not** install it. Correct path is
  `git clone` + `pip install timm einops yacs` + `EGOVLP_REPO` env var. The loader in
  `encoder_bakeoff.py` is written but untested.
- The UCF-Crime probe (does the encoder transfer to surveillance?) has not been run. It
  is the second half of Gate 2 and the one that decides the Stage 2 encoder.

---

## 8. Prior-art corrections to `crvad_pipeline.tex`

Verified during research; the existing document is wrong on these points.

| Claim | Correct position |
|---|---|
| Alert-CLIP uses hard negatives changing 2‚Äì4 semantic factors | **Refuted.** MLLM free-form caption polarity inversion, 3 per region caption, **no factor schema, no factor counting** |
| LAVIDA-style negatives are random decoys | **Correct**, with nuance: prompt-injected category strings, not contrastive negatives |
| "Nothing-Else Matters", Mottaghi, ECCV 2022 | **Does not exist.** Canonical protocol is **Something-Else**, Materzynska et al., **CVPR 2020**, arXiv 1912.09930 |
| Something-Else gives verb‚Äìobject pairs | It gives **dense per-frame bounding boxes** (180,049 videos, 16.96M boxes, **assumes 12 fps**) |
| ~20‚Äì25% held-out verb‚Äìobject pairs | Not the protocol. 50/50 flip of two noun √ó two action groups ‚Üí **54,919 train / 57,876 val** |
| SSv2 put/take example | **Neither string exists in SSv2.** Real pair: `Covering something with something` (6) ‚Üî `Uncovering something` (171) |
| Implicit: CLIP is weak for VAD | **False.** VadCLIP and CLIP-TSA beat SOTA with CLIP features. The citable gap is AnomalyCLIP's: *"CLIP's original pre-training focuses on aligning with object semantics instead of the abnormality and normality within images"* |

Newly relevant: **CLM** (Contrastive Language Model, Stanford/NVIDIA, Apache-2.0, 2,989
GitHub stars) and **Jev** (TypeSafe.ai, arXiv 2609.24395) are "System 1" models ‚Äî
frozen encoder + small projection heads + InfoNCE over a *closed, supplied candidate
set*, returning a probability distribution instead of generating tokens.
`crvad/model.py` is structurally the same machine in a different modality, so CRVAD
should be framed as a **transfer of that recipe to VAD**, not as an invented
architecture. CLM is text-only and encoder-locked to Qwen3-8B embeddings; a multimodal
CLM-35B was announced for "early October" and had not shipped as of 12 Oct 2026.
---

## 9. Gate 2 v2 ó the first run was invalid, and the corrected result is negative

Appended 14 Oct 2026. This section supersedes the Gate-2 numbers in ß2.

### 9.1 Why every earlier Gate-2 number is void

The v1 feature caches were degenerate. Measured on the cached tensors:

| | v1 cache | healthy |
|---|---|---|
| fraction of entries > 0 | **0.968** (CLIP) / 0.521 (X-CLIP) | ~0.5 |
| mean pairwise cosine between clips | **0.955** (CLIP) / 0.844 (X-CLIP) | ~0.6ñ0.7 |

Mean pairwise cosine 0.955 means every clip was ~95% identical to every other clip. Three
independent bugs produced that:

1. **Per-dimension MAX over frames.** v1 did `feat.view(b,f,-1).max(dim=1).values` over 16
   independently-normalised frame vectors, driving ~97% of coordinates positive and
   inflating all inter-clip cosine toward 1.
2. **`clamp(0,1)` applied AFTER normalisation**, zeroing roughly half of CLIP's pixels.
3. **X-CLIP was never normalised at all.** `NORM["xclip"]` was defined and never called.

Separately, the pooled `edit_acc` metric was **not imbalance-invariant**: a classifier that
ignores the video and always prefers the larger template scores `n_A/(n_A+n_B)`, and on these
pair sizes that range is **0.360ñ0.640**. CLIP's reported 0.622 sat *inside* it.
`tools/audit_gate2_json.py` reproduces this on the archived v1 json.

The corrected CLIP number is 0.531 ó mid-band, i.e. chance.

### 9.2 Corrected zero-shot bakeoff (v2, mean pooling, proper normalisation)

Two machines, same code: local CPU on 264 clips, remote A40 on 275 clips. (The clip-count
difference comes from the AV1 decode boundary ó `av` 14 on py3.11 vs `av` 19 on py3.12 ó and
means the two runs are not strictly comparable; see ß9.5.)

| encoder | macro AUC | 95% CI | perm p | balanced acc | video-blind pairs |
|---|---|---|---|---|---|
| CLIP (A40) | 0.524 | [0.451, 0.597] | 0.287 | 0.501 | 19/22 |
| X-CLIP (A40) | 0.570 | [0.500, 0.636] | 0.053 | 0.539 | 8/22 |

"video-blind pairs" counts pairs where the number correct exactly equals one class's clip
count ó the signature of a classifier that never looks at the video.

The v1 ranking (CLIP > X-CLIP) was **backwards for two reasons**, both v1 bugs: CLIP's was
inflated by max-pooling, X-CLIP's depressed by having no normalisation. Fixed both and X-CLIP
leads on the bias-invariant metric, though p=0.053 is not significant.

### 9.3 Relation probe ó null on every encoder, every variant

Standardised ridge with exact-LOO ?, binary per-pair, scored by macro AUC against a
within-pair label-permutation null (200 permutations). A p-value here is the fraction of
shuffled-label runs that matched or beat the observed score ó so 0.318 means "a third of
noise runs beat you", and 0.861 means "most of them did".

**A40, 275 clips:**

| encoder | variant | macro AUC | null | p95 | p |
|---|---|---|---|---|---|
| clip | mean | 0.472 | 0.496 | 0.580 | 0.667 |
| clip | halves | 0.481 | 0.491 | 0.578 | 0.547 |
| clip | delta | 0.484 | 0.491 | 0.578 | 0.507 |
| clip | halves_shuf | 0.492 | 0.491 | 0.569 | 0.498 |
| xclip | vmean | 0.525 | 0.499 | 0.581 | 0.318 |
| xclip | vhalves | 0.458 | 0.493 | 0.576 | 0.731 |
| xclip | vdelta | 0.451 | 0.498 | 0.587 | 0.806 |
| xclip | vshuf | 0.527 | 0.500 | 0.584 | 0.308 |
| viclip | vmean | 0.438 | 0.492 | 0.571 | 0.861 |
| viclip | vhalves | 0.453 | 0.496 | 0.576 | 0.816 |
| viclip | vdelta | 0.448 | 0.497 | 0.592 | 0.871 |
| viclip | vshuf | 0.455 | 0.498 | 0.576 | 0.786 |

**CPU, 264 clips** ó CLIP replicates as null on all four variants (mean 0.490 p=0.498,
halves 0.443 p=0.821, delta 0.451 p=0.746, halves_shuf 0.530 p=0.219). X-CLIP on CPU gave
mean AUC 0.588, **p=0.045**, which looked significant; **it did not replicate on the A40**
(0.525, p=0.318). Treated as noise and retracted.

### 9.4 Order sensitivity ó the substantive finding

X-CLIP and ViCLIP collapse frames *inside* the encoder, so no frame axis survives in the
output and the usual `mean`/`halves`/`delta` split cannot be computed for them. Built at the
encoder *input* instead: encode the first half and second half separately, and measure
`cosine(identity, shuffled)` where identity and shuffled see the same frame set at the same
arity, differing only in order.

| encoder | cosine(identity, shuffled) | reading |
|---|---|---|
| xclip | **0.9986** | ~3∞ of movement ó effectively order-**blind** |
| viclip | **0.9636** | ~15∞ ó genuinely more order-responsive than X-CLIP |

So spatiotemporal attention does buy *some* order responsiveness, but neither encoder uses
frame order enough to support minimal-pair discrimination: `vhalves` is no better than
`vmean` on either, and `vshuf` ò `vmean` confirms it from the other side. This is the direct
answer to "video pretraining buys nothing here" ó it buys a little order sensitivity and no
usable relation signal.

Note: this also explains why `vhalves` underperformed `vmean` on X-CLIP (0.458 vs 0.525).
Concatenating two halves doubles the feature dimension for the probe with no added signal, so
it can only add variance.

### 9.5 Unresolved: the X-CLIP CPU/GPU discrepancy

Same metric, same code, 264 vs 275 clips, macro AUC 0.588 (p=0.045) vs 0.525 (p=0.318). An
11-clip difference should not move an AUC by 0.063. Most likely the AV1 decode boundary
(`av` 14 on py3.11 vs 19 on py3.12) selecting different frames, but **this is unverified**.
Until it is explained, treat single-run p-values from this pipeline as provisional ó it is
the difference between measuring something and measuring noise.

### 9.6 What this does and does not establish

**Established:** on 264ñ275 clips, none of CLIP / X-CLIP / ViCLIP separates SSv2 minimal
intervention pairs above its own permutation null, on any pooled, per-frame, or order-aware
probe, on two machines. Video pretraining does not help at this scale.

**Not established:** that the information is absent. Every null has p95 ò 0.57ñ0.59 ó that
is the noise floor at 19 pairs and ~5 clips per template, and it means a real effect of AUC
~0.55 would be invisible. The probe has low *power*, not just a null result.

**Also note:** Gate 1's AUC 0.979 was measured on `slots.SCHEMA`, which is explicitly a
PLACEHOLDER (`takes/places/carriesÖ ◊ bag/bottle/boxÖ ◊ store/streetÖ ◊ from a shelf/Ö`),
not on SSv2 and not on surveillance. It remains a valid statement about the text encoder's
behaviour, but it is not evidence for the real task, and Stage 1 currently has no real slot
labels to train on at all.

### 9.7 Next

| # | Action | Cost | Why |
|---|---|---|---|
| 1 | Split directional vs pretend pairs (ß `pair_kind`) | seconds | If the weak signal was object/scene appearance leak rather than relation, this shows it. **Done ó see ß9.8.** |
| 2 | 5-shard build via `ssv2_multi.py` ? ~1,300 clips | ~20 min A40 | The only fix for the power problem; every null here sits at p95 ò 0.58 |
| 3 | Define the real surveillance schema | ó | Stage 1 cannot train without real slot labels, independent of encoder choice |
| 4 | UCF-Crime zero-shot probe | ~minutes | Does CLIP transfer to surveillance at all; no slot labels needed |

### 9.8 Directional vs pretend split

The 22 pairs are not one population. 9 are **directional** (opening/closing, up/down,
into/out of, camera left/right) ó these differ only in temporal ORDER, so a pooled
order-blind representation cannot separate them in principle, and any score above 0.5 there
indicates an appearance difference between the two templates rather than the relation. The
other 13 are **pretend** (performed vs mimed). Conflating them hides which failure mode is
happening, so each group is now scored against its own permutation null.

Result: [pending ó see `runs/gate2_split_*.json`].