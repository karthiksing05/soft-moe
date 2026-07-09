#!/usr/bin/env python
"""Carve a (K personas × n episodes) cell out of a persona pool for the many-personas/few-episodes sweep."""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="pool dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, required=True, help="use personas with expert_id < K")
    ap.add_argument("--n", type=int, required=True, help="train episodes per persona")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src = Path(a.data); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    experts = json.loads((src / "experts.json").read_text())
    rng = random.Random(a.seed)
    for variant in ("control", "em"):
        for fold in ("train", "test"):
            rows = [json.loads(l) for l in open(src / f"{variant}.{fold}.jsonl")]
            rows = [r for r in rows if r["expert_id"] < a.k]                 # keep first K personas
            if fold == "train":                                             # subsample n episodes/persona
                by = defaultdict(list)
                for r in rows:
                    by[r["expert_id"]].append(r)
                kept = []
                for eid, rs in by.items():
                    rng.shuffle(rs); kept += rs[: a.n]
                rows = kept
            rng.shuffle(rows)
            (out / f"{variant}.{fold}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "experts.json").write_text(json.dumps(
        {"n_experts": a.k, "expert_tokens": [f"<|expert_{k}|>" for k in range(a.k)],
         "names": experts["names"][: a.k]}, indent=2))
    ntr = sum(1 for _ in open(out / "em.train.jsonl"))
    print(f"cell K={a.k} n={a.n} -> {out}   train rows={ntr}  (aggregate ≈ {a.k}×{a.n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
