#!/usr/bin/env python
"""Submit every sweep variant to Kaggle, wait for scoring, record the scores.

Flow (fully automatic once ~/.kaggle/kaggle.json exists):
    preflight -> submit variant -> poll until scored -> record -> next
    -> write scores.csv / results.md -> optionally git commit + push

Usage:
    python scripts/auto_submit.py --dry-run          # show what would happen
    python scripts/auto_submit.py                    # submit all pending variants
    python scripts/auto_submit.py --only lizard_mean2.0
    python scripts/auto_submit.py --publish          # also commit + push results
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAR = ROOT / "submissions" / "variants"
COMP = "animal-clef-2026"
STATE = VAR / "scores.json"
BASE_FILE = "base_densitymatched.csv"
BASE_PUBLIC, BASE_PRIVATE = 0.35805, 0.37244      # already on Kaggle

# Kaggle sits behind the local proxy on this machine; direct access times out.
DEFAULT_PROXY = "http://127.0.0.1:7897"


def kaggle_bin() -> str:
    local = ROOT / ".venv" / "bin" / "kaggle"
    return str(local) if local.exists() else "kaggle"


def env() -> dict:
    e = dict(os.environ)
    if not e.get("HTTPS_PROXY") and proxy_alive():
        e["HTTPS_PROXY"] = e["HTTP_PROXY"] = DEFAULT_PROXY
    return e


def proxy_alive() -> bool:
    import socket
    host, port = DEFAULT_PROXY.split("//")[1].split(":")
    try:
        with socket.create_connection((host, int(port)), timeout=1):
            return True
    except OSError:
        return False


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=env(), **kw)


def preflight() -> bool:
    cp = run([kaggle_bin(), "competitions", "list", "-s", "animal-clef",
              "--csv"])
    if cp.returncode != 0:
        print("[preflight] Kaggle API is not usable:")
        print((cp.stderr or cp.stdout).strip()[:500])
        print("\nPut your API token at ~/.kaggle/kaggle.json")
        print('   format: {"username": "...", "key": "..."}')
        print("   or set KAGGLE_USERNAME / KAGGLE_KEY in the environment.")
        return False
    print("[preflight] Kaggle API OK")
    return True


def list_submissions() -> list[dict]:
    cp = run([kaggle_bin(), "competitions", "submissions", COMP, "--csv"])
    if cp.returncode != 0:
        raise RuntimeError((cp.stderr or cp.stdout).strip()[:400])
    rows = list(csv.DictReader(io.StringIO(cp.stdout)))
    return rows


def find_row(rows: list[dict], name: str) -> dict | None:
    hits = [r for r in rows if (r.get("description") or "").strip() == name]
    return hits[0] if hits else None


def submit(path: Path, name: str) -> None:
    cp = run([kaggle_bin(), "competitions", "submit", "-c", COMP,
              "-f", str(path), "-m", name, "-q"])
    out = (cp.stderr or cp.stdout).strip()
    if cp.returncode != 0:
        raise RuntimeError(out[:400] or "submit failed")


def wait_scored(name: str, timeout: int, interval: int) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            row = find_row(list_submissions(), name)
        except RuntimeError:
            row = None
        if row and row.get("status", "").lower().startswith(("complete", "error")):
            return row
        if row and row.get("status") != last:
            last = row.get("status")
            print(f"    status: {last}")
        time.sleep(interval)
    raise TimeoutError(f"{name} did not finish within {timeout}s")


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True))


def delta_table(state: dict, manifest: dict) -> str:
    lines = ["| variant | species | public | private | Δ public | Δ ARI (×4) |",
             "|---|---|---|---|---|---|"]
    base_pub = state.get(BASE_FILE, {}).get("public") or BASE_PUBLIC
    for v in manifest["variants"]:
        s = state.get(v["file"], {})
        pub, priv = s.get("public"), s.get("private")
        d_pub = f"{pub - base_pub:+.4f}" if pub is not None else ""
        d_ari = f"{(pub - base_pub) * 4:+.3f}" if pub is not None else ""
        lines.append(f"| {v['name']} | {v['species']} | {pub if pub is not None else ''} "
                     f"| {priv if priv is not None else ''} | {d_pub} | {d_ari} |")
    return "\n".join(lines)


def publish(msg: str) -> None:
    rel = ["submissions/variants/scores.json", "submissions/variants/results.md",
           "submissions/variants/README.md"]
    subprocess.run(["git", "add", *rel], cwd=ROOT, check=False)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        print("[publish] nothing to commit")
        return
    body = msg + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
    subprocess.run(["git", "commit", "-q", "-m", body], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
    print("[publish] pushed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="submit a single variant name")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--delay", type=int, default=10, help="sleep between submissions")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    manifest = json.loads((VAR / "manifest.json").read_text())
    state = load_state()
    state.setdefault(BASE_FILE, {"name": "base", "public": BASE_PUBLIC,
                                 "private": BASE_PRIVATE, "status": "complete"})

    todo = [v for v in manifest["variants"]
            if (args.only is None or v["name"] == args.only)
            and state.get(v["file"], {}).get("status") != "complete"]
    if not todo:
        print("nothing to submit; all variants already scored")
    else:
        print(f"{len(todo)} variant(s) to submit")

    if args.dry_run:
        for v in todo:
            print(f"  kaggle competitions submit -c {COMP} "
                  f"-f {VAR / v['file']} -m {v['name']}")
        return 0

    if todo and not preflight():
        return 1

    for i, v in enumerate(todo):
        path = VAR / v["file"]
        print(f"[{i+1}/{len(todo)}] {v['name']} ({v['clusters']} clusters)")
        try:
            submit(path, v["name"])
            row = wait_scored(v["name"], args.timeout, args.interval)
        except (RuntimeError, TimeoutError) as e:
            print(f"    FAILED: {e}")
            state[v["file"]] = {"name": v["name"], "status": "failed", "error": str(e)}
            save_state(state)
            continue
        pub = row.get("publicScore") or None
        priv = row.get("privateScore") or None
        state[v["file"]] = {
            "name": v["name"], "species": v["species"], "clusters": v["clusters"],
            "status": row.get("status"), "public": float(pub) if pub else None,
            "private": float(priv) if priv else None, "date": row.get("date"),
        }
        save_state(state)
        print(f"    public={pub}  private={priv}")
        time.sleep(args.delay)

    (VAR / "results.md").write_text(
        "# Variant sweep results\n\n" + delta_table(state, manifest) +
        "\n\nΔ public = variant − baseline; Δ ARI = 4 × Δ public "
        "(overall is the four-species mean).\n")
    print("\n" + delta_table(state, manifest))

    if args.publish:
        publish("Record variant sweep scores")
    return 0


if __name__ == "__main__":
    sys.exit(main())
