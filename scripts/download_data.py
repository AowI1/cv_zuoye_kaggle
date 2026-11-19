#!/usr/bin/env python3
"""Robust downloader for AnimalCLEF26 official data from the HF mirror.

Retries with HTTP Range resume until the expected byte size is reached.
"""
import os
import sys
import time
import urllib.request

BASE = "https://hf-mirror.com/datasets/BVRA/animal-clef-2026/resolve/main"
DEST = "/Users/fy/Desktop/kaggle/data/animalclef2026"
FILES = {
    "metadata.csv": 2089384,
    "sample_submission.csv": 75633,
    "data/train.zip": 1348110290,
    "data/test.zip": 819201168,
}

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"


def download(rel: str, expected: int) -> None:
    out = os.path.join(DEST, os.path.basename(rel))
    attempt = 0
    while True:
        attempt += 1
        cur = os.path.getsize(out) if os.path.exists(out) else 0
        if cur >= expected:
            print(f"[OK] {rel}: {cur} bytes", flush=True)
            return
        req = urllib.request.Request(BASE + "/" + rel, headers={"User-Agent": UA})
        if cur > 0:
            req.add_header("Range", f"bytes={cur}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(out, "ab" if cur else "wb") as f:
                total = cur
                last_log = 0.0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
                    now = time.time()
                    if now - last_log > 30:
                        print(f"[..] {rel}: {total/1e6:.1f}/{expected/1e6:.1f} MB", flush=True)
                        last_log = now
        except Exception as e:
            print(f"[retry {attempt}] {rel}: {e}", flush=True)
        time.sleep(5 + (attempt % 10))


def main() -> int:
    os.makedirs(DEST, exist_ok=True)
    for rel, expected in FILES.items():
        download(rel, expected)
    print("ALL DOWNLOADS COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
