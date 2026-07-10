#!/usr/bin/env python
"""Filter a persona pool to a subset of expert_ids, KEEPING the full experts.json (token vocabulary).

Used by the incremental-scaling test: a 'base' dir keeps only the base personas' train/test, and a 'new'
dir keeps only the held-out new personas — but both retain the full 64-token experts.json so the model
still has a token slot for every persona (base tokens get trained, new tokens are fit later via Phase B).
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", required=True, help="comma-sep expert_ids to keep")
    a = ap.parse_args()
    keep = {int(x) for x in a.ids.split(",")}
    src = Path(a.data); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for variant in ("control", "em"):
        for fold in ("train", "test"):
            rows = [json.loads(l) for l in open(src / f"{variant}.{fold}.jsonl")]
            rows = [r for r in rows if r["expert_id"] in keep]
            (out / f"{variant}.{fold}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    shutil.copy(src / "experts.json", out / "experts.json")          # full token vocab unchanged
    ntr = sum(1 for _ in open(out / "em.train.jsonl"))
    print(f"filter -> {out}  ids={sorted(keep)}  train_rows={ntr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
