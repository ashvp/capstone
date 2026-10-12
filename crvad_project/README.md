# CRVAD — Counterfactual Relational Video Anomaly Detection (minimal-intervention contrastive learning)

Code skeleton accompanying `crvad_pipeline.pdf`. Frozen encoders + small trainable heads, so 2x A40 is ample.

## Status (read this first)
| Part | Status |
|---|---|
| `slots.py`, `model.py`, `losses.py`, `train.py`, `evaluate.py`, `data.py` | Run end-to-end on **synthetic** features (`tests/smoke_test.py`, CPU). |
| `pseudo_label.py` (parsing, voting, audit CSV) | Tested with a mock VLM. `QwenVLBackend` is **untested** (needs GPU + weights). |
| `features.py` (CLIP extraction, text bank, nuisance views) | **Untested** (needs GPU, weights, videos). |
| Scientific claims | **None yet.** In the synthetic world all negative modes saturate edit accuracy, so the smoke test does not discriminate between them. |

## Layout
```
crvad/slots.py         closed slot schema (8x6x6x6 = 1728 hypotheses), single-slot edits, compositional split
crvad/model.py         temporal adapter + text projection + normality head over slot tuples
crvad/losses.py        negative samplers (random | multi | single), candidate InfoNCE + margin, invariance, MIL
crvad/train.py         stage 1 (relational) and stage 2 (weak-label VAD)
crvad/evaluate.py      AUC/AP, edit (grounding) accuracy, flip accuracy, nuisance stability, held-out splits
crvad/pseudo_label.py  VLM slot labelling, self-consistency confidence, human-audit export
crvad/features.py      CLIP segment features (+ nuisance view), text bank
crvad/synthetic.py     synthetic feature world for tests only
tests/smoke_test.py    end-to-end test + miniature negative-mode ablation
configs/default.yaml
```

## Quickstart
```bash
pip install -r requirements.txt
python tests/smoke_test.py                       # synthetic end-to-end check (CPU ok)

# real run (2x A40):
python -m crvad.features text --out data/text_bank.pt
CUDA_VISIBLE_DEVICES=0 python -m crvad.features videos --list ssv2.txt --out data/ssv2_0.pt --shard 0/2 --nuisance
CUDA_VISIBLE_DEVICES=1 python -m crvad.features videos --list ssv2.txt --out data/ssv2_1.pt --shard 1/2 --nuisance
# build data/stage1_train.pt and data/stage2_train.pt (dict format documented in crvad/data.py), then:
CUDA_VISIBLE_DEVICES=0 python -m crvad.train --stage 1 --config configs/default.yaml --set stage1.neg_mode=single out=runs/single_s0
CUDA_VISIBLE_DEVICES=1 python -m crvad.train --stage 1 --config configs/default.yaml --set stage1.neg_mode=random out=runs/random_s0
```

## Rules you should keep
* Select checkpoints on a validation split of the *training* source, never on UCF-Crime / XD-Violence / ShanghaiTech test sets.
* Report text-only and shuffled-video baselines on the controlled pair set before trusting any number.
* `slots.py` schema and `ABNORMAL_RULES` are placeholders; replace after the pseudo-labelling pilot.
