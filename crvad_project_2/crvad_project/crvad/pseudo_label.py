"""Stage 0: VLM pseudo-labelling of slot factors, with self-consistency confidence and a human-audit export.

The VLM is pluggable. `MockBackend` is used in tests; `QwenVLBackend` follows the documented Qwen2-VL usage but
has NOT been run in this repo's smoke test (needs GPU + weights): validate it on ~50 clips before any bulk run.

    # shard across the two A40s:
    CUDA_VISIBLE_DEVICES=0 python -m crvad.pseudo_label --items items.jsonl --out labels_0.jsonl --shard 0/2
    CUDA_VISIBLE_DEVICES=1 python -m crvad.pseudo_label --items items.jsonl --out labels_1.jsonl --shard 1/2

items.jsonl rows: {"id": "...", "frames": ["/path/f1.jpg", ...], "video_label": 0|1}
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
from typing import Dict, List, Optional, Sequence

from . import slots as S


def build_prompt() -> str:
    lines = ["You are annotating a surveillance clip. Describe the single most salient event by choosing exactly one "
             "value per field. Answer with JSON only."]
    for name in S.SLOT_NAMES:
        lines.append(f'"{name}": one of {json.dumps(S.SCHEMA[name])}')
    lines.append('If nothing fits a field, use "unknown" for that field.')
    return "\n".join(lines)


def parse_response(text: str) -> Optional[Dict[str, Optional[int]]]:
    """Extract the first JSON object and map values to schema indices (None for unknown / invalid)."""
    obj = None
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):               # first brace that starts a valid JSON object
        try:
            cand, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict) and any(k in cand for k in S.SLOT_NAMES):   # skip unrelated JSON objects
            obj = cand
            break
    if obj is None:
        return None
    out: Dict[str, Optional[int]] = {}
    for name in S.SLOT_NAMES:
        val = str(obj.get(name, "unknown")).strip().lower()
        vocab = [v.lower() for v in S.SCHEMA[name]]
        out[name] = vocab.index(val) if val in vocab else None
    return out


def aggregate(samples: Sequence[Optional[Dict[str, Optional[int]]]]):
    """Majority vote per slot over N sampled answers. confidence = mean per-slot agreement (0 if any slot unresolved)."""
    samples = [s for s in samples if s]
    if not samples:
        return None, 0.0
    ev, agree = [], []
    for name in S.SLOT_NAMES:
        vals = [s[name] for s in samples if s[name] is not None]
        if not vals:
            return None, 0.0
        v, c = collections.Counter(vals).most_common(1)[0]
        ev.append(v)
        agree.append(c / len(samples))
    return tuple(ev), float(sum(agree) / len(agree))


class MockBackend:
    """Deterministic fake VLM for tests: returns a random schema-valid JSON answer."""
    def __init__(self, seed: int = 0, bad_rate: float = 0.1):
        self.rng, self.bad = random.Random(seed), bad_rate

    def generate(self, frames: List[str], prompt: str, n: int) -> List[str]:
        outs = []
        for _ in range(n):
            if self.rng.random() < self.bad:
                outs.append("sorry, I cannot tell")
            else:
                outs.append(json.dumps({k: self.rng.choice(v) for k, v in S.SCHEMA.items()}))
        return outs


class QwenVLBackend:
    """Qwen2-VL (7B instruct) via transformers. UNTESTED here — verify on a few clips first."""
    def __init__(self, model_path: str = "Qwen/Qwen2-VL-7B-Instruct", max_new_tokens: int = 80, temperature: float = 0.7):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.max_new_tokens, self.temperature = max_new_tokens, temperature

    def generate(self, frames: List[str], prompt: str, n: int) -> List[str]:
        from qwen_vl_utils import process_vision_info
        msgs = [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(msgs)
        inp = self.processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(self.model.device)
        # n independent sampled generations in a loop: num_return_sequences>1 with video inputs has
        # been buggy in some transformers versions and returns near-duplicate captions.
        texts = []
        for _ in range(n):
            out = self.model.generate(**inp, do_sample=True, temperature=self.temperature,
                                      max_new_tokens=self.max_new_tokens)
            texts += self.processor.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)
        return texts


def run(backend, items: List[dict], n_samples: int = 3) -> List[dict]:
    prompt, rows = build_prompt(), []
    for it in items:
        ev, conf = aggregate([parse_response(t) for t in backend.generate(it["frames"], prompt, n_samples)])
        rows.append(dict(id=it["id"], video_label=it.get("video_label"), slots=list(ev) if ev else None, conf=conf))
    return rows


def audit_csv(rows: List[dict], path: str, n: int = 300, seed: int = 0):
    """Export a random sample for human verification (fill the 'human_*' columns, then run audit_report)."""
    rng = random.Random(seed)
    pick = rng.sample([r for r in rows if r["slots"]], min(n, sum(1 for r in rows if r["slots"])))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "conf"] + [f"vlm_{s}" for s in S.SLOT_NAMES] + [f"human_{s}" for s in S.SLOT_NAMES])
        for r in pick:
            w.writerow([r["id"], f"{r['conf']:.2f}"] + [S.SCHEMA[s][v] for s, v in zip(S.SLOT_NAMES, r["slots"])] + [""] * 4)


def audit_report(path: str) -> Dict[str, float]:
    """Per-slot agreement between VLM and human labels, plus agreement vs confidence bucket."""
    rows = [r for r in csv.DictReader(open(path)) if all(r[f"human_{s}"] for s in S.SLOT_NAMES)]
    rep = {s: sum(r[f"vlm_{s}"].lower() == r[f"human_{s}"].lower() for r in rows) / max(1, len(rows)) for s in S.SLOT_NAMES}
    rep["n"] = len(rows)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--backend", choices=["qwen", "mock"], default="qwen")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    items = [json.loads(l) for k, l in enumerate(open(a.items)) if k % n == i]
    rows = run(QwenVLBackend() if a.backend == "qwen" else MockBackend(), items, a.n_samples)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
