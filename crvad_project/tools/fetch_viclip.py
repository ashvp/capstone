"""Resilient downloader for the ViCLIP-B-16-hf snapshot.

huggingface_hub's HTTP stack dies on this machine with a repeatable
`WinError 10053` (connection aborted by the host) -- the same error that made the
earlier bakeoff runs fall back to cached commit hashes. urllib has a much higher
success rate on the same URLs, so this walks the file list over urllib directly
with exponential backoff and resumes partial downloads.

    python tools/fetch_viclip.py [--dest hf_cache/models--OpenGVLab--ViCLIP-B-16-hf]

Safe to re-run: files that already exist at the expected size are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

REPO = "OpenGVLab/ViCLIP-B-16-hf"
WANT = ("config.json", "configuration_viclip.py", "viclip.py", "viclip_text.py",
        "viclip_vision.py", "simple_tokenizer.py", "bpe_simple_vocab_16e6.txt.gz",
        "model.safetensors")
UA = {"User-Agent": "Mozilla/5.0"}


def _get(url: str, tries: int = 8, timeout: int = 90) -> bytes:
    """GET with exponential backoff. WinError 10053/10054 are transient here."""
    delay = 2.0
    last = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout,
                                        context=ssl.create_default_context()) as r:
                return r.read()
        except Exception as e:                      # noqa: BLE001 - retry anything
            last = e
            print(f"    attempt {attempt}/{tries} failed: {type(e).__name__}: {str(e)[:80]}")
            if attempt < tries:
                time.sleep(delay)
                delay = min(delay * 1.8, 30.0)
    raise RuntimeError(f"could not fetch {url}: {last}")


CHUNK = 1 << 20


def _download(url: str, path: str, tries: int = 12, timeout: int = 120) -> int:
    """Stream to disk with resume. A 600 MB read() in one shot gets killed mid-flight
    by the same reset, so write incrementally and continue from where we stopped using
    a Range request."""
    delay = 2.0
    last = None
    for attempt in range(1, tries + 1):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        hdrs = dict(UA)
        if have:
            hdrs["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout,
                                        context=ssl.create_default_context()) as r:
                # A 200 on a ranged request means the server ignored Range; restart.
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with open(path, mode) as fh:
                    while True:
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        fh.write(buf)
                        have += len(buf)
            return have
        except Exception as e:                      # noqa: BLE001 - retry anything
            last = e
            got = os.path.getsize(path) if os.path.exists(path) else 0
            print(f"    attempt {attempt}/{tries}: {type(e).__name__}: {str(e)[:70]} "
                  f"(resuming from {got/1e6:.1f} MB)")
            if attempt < tries:
                time.sleep(delay)
                delay = min(delay * 1.6, 25.0)
    raise RuntimeError(f"could not download {url}: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=f"hf_cache/models--{REPO.replace('/', '--')}")
    ap.add_argument("--skip-weights", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.dest, exist_ok=True)
    want = [f for f in WANT if not (a.skip_weights and f.endswith(".safetensors"))]
    base = f"https://huggingface.co/{REPO}/resolve/main"

    print(f"listing {REPO} ...")
    try:
        meta = json.loads(_get(f"https://huggingface.co/api/models/{REPO}"))
        remote = {s["rfilename"] for s in meta.get("siblings", [])}
        print(f"  {len(remote)} files on the hub")
    except Exception as e:                          # noqa: BLE001
        print(f"  listing failed ({type(e).__name__}); using the hardcoded list")

    for name in want:
        path = os.path.join(a.dest, name)
        # The hub stores LFS blobs under a different name, so compare against the
        # pointer file when present rather than trusting a stale local size.
        if os.path.exists(path) and os.path.getsize(path) > 0:
            head = _get(base + "/" + name, tries=3)
            expected = len(head)
            if expected == os.path.getsize(path):
                print(f"  [have] {name}  ({expected/1e6:.1f} MB)")
                continue
        if name.endswith(".safetensors"):
            n = _download(base + "/" + name, path)
            print(f"  [ got] {name}  ({n/1e6:.1f} MB, streamed)")
        else:
            blob = _get(base + "/" + name)
            with open(path, "wb") as fh:
                fh.write(blob)
            print(f"  [ got] {name}  ({len(blob)/1e6:.1f} MB)")

    print(f"\ndone -> {os.path.abspath(a.dest)}")
    for f in sorted(os.listdir(a.dest)):
        print(f"  {f}  {os.path.getsize(os.path.join(a.dest, f))/1e6:.1f} MB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
