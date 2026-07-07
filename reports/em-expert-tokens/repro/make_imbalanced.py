#!/usr/bin/env python
"""Create an imbalanced-volume persona set from a balanced one (cold-start / long-tail test).

The thesis argues EM alternation helps most for *low-data* speakers: in joint SFT a scarce-data persona's
embedding gets too few gradient updates, whereas Phase B fits it against a frozen, capable model. To test
this we subsample each persona's TRAIN examples to a geometric spread (e.g. 450 -> 4), leaving the held-out
TEST set balanced, then compare joint SFT vs EM per persona as a function of that persona's train volume.
"""
from __future__ import annotations
import argparse, json, random, shutil
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="balanced source dir ({control,em}.{train,test}.jsonl + experts.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--counts", default="450,240,128,64,32,16,8,4",
                    help="per-expert TRAIN counts in expert_id order (comma-sep)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src = Path(a.data); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    experts = json.loads((src / "experts.json").read_text())
    counts = [int(x) for x in a.counts.split(",")]
    assert len(counts) == experts["n_experts"], f"need {experts['n_experts']} counts, got {len(counts)}"
    rng = random.Random(a.seed)

    for variant in ("control", "em"):
        rows = [json.loads(l) for l in open(src / f"{variant}.train.jsonl")]
        by = defaultdict(list)
        for r in rows:
            by[r["expert_id"]].append(r)
        kept = []
        for eid, rs in by.items():
            rng.shuffle(rs)
            kept += rs[: counts[eid]]
        rng.shuffle(kept)
        (out / f"{variant}.train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
        shutil.copy(src / f"{variant}.test.jsonl", out / f"{variant}.test.jsonl")     # test stays balanced
    shutil.copy(src / "experts.json", out / "experts.json")
    vol = {experts["names"][i]: counts[i] for i in range(len(counts))}
    (out / "volumes.json").write_text(json.dumps(vol, indent=2))
    print(f"imbalanced set -> {out}   volumes: {vol}   total train = {sum(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
