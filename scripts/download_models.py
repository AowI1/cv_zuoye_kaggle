#!/usr/bin/env python3
"""Robust downloader for HF model files via the HF mirror.

Finds the real content length by following redirects, then resumes
until complete. Retries on rate-limit responses.
"""
import os
import sys
import time
import urllib.request

MIRROR = "https://hf-mirror.com"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
DEST = "/Users/fy/Desktop/kaggle/models"

# repo_type, repo_id, filename, local_subdir
FILES = [
    ("model", "conservationxlabs/miewid-msv3", "model.safetensors", "miewid-msv3"),
    ("model", "conservationxlabs/miewid-msv3", "config.json", "miewid-msv3"),
    ("model", "conservationxlabs/miewid-msv3", "configuration_miewid.py", "miewid-msv3"),
    ("model", "conservationxlabs/miewid-msv3", "modeling_miewid.py", "miewid-msv3"),
    ("model", "conservationxlabs/miewid-msv3", "heads.py", "miewid-msv3"),
    ("model", "BVRA/MegaDescriptor-L-384", "pytorch_model.bin", "MegaDescriptor-L-384"),
    ("model", "BVRA/MegaDescriptor-L-384", "config.json", "MegaDescriptor-L-384"),
]


def probe_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else 0


def download(kind: str, repo: str, fname: str, subdir: str) -> None:
    prefix = "" if kind == "model" else "datasets/"
    url = f"{MIRROR}/{prefix}{repo}/resolve/main/{fname}"
    out_dir = os.path.join(DEST, subdir)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, fname)
    attempt = 0
    while True:
        attempt += 1
        try:
            expected = probe_size(url)
        except Exception as e:
            print(f"[probe retry {attempt}] {repo}/{fname}: {e}", flush=True)
            time.sleep(5 + (attempt % 10))
            continue
        cur = os.path.getsize(out) if os.path.exists(out) else 0
        if expected and cur >= expected:
            print(f"[OK] {repo}/{fname}: {cur} bytes", flush=True)
            return
        req = urllib.request.Request(url, headers={"User-Agent": UA})
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
                        print(f"[..] {repo}/{fname}: {total/1e6:.1f} MB", flush=True)
                        last_log = now
        except Exception as e:
            print(f"[retry {attempt}] {repo}/{fname}: {e}", flush=True)
        time.sleep(5 + (attempt % 10))


def main() -> int:
    for kind, repo, fname, subdir in FILES:
        download(kind, repo, fname, subdir)
    print("ALL MODELS DOWNLOADED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
