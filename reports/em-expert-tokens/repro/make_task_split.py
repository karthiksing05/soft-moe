#!/usr/bin/env python
"""Split a persona set into two sequential tasks (A, B) for the catastrophic-forgetting test.

Task A = personas in --group-a, Task B = the rest. Each output dir gets that task's TRAIN examples but the
FULL (all-persona) TEST set + experts.json, so a single held-out eval reports per-persona ppl for both
tasks (A = retention, B = newly learned) at every stage.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-a", required=True)
    ap.add_argument("--out-b", required=True)
    ap.add_argument("--group-a", default="0,1,2,3", help="expert_ids for task A (rest = task B)")
    a = ap.parse_args()
    src = Path(a.data)
    ga = {int(x) for x in a.group_a.split(",")}
    experts = json.loads((src / "experts.json").read_text())
    gb = set(range(experts["n_experts"])) - ga
    for out, grp, tag in [(a.out_a, ga, "A"), (a.out_b, gb, "B")]:
        o = Path(out); o.mkdir(parents=True, exist_ok=True)
        for variant in ("control", "em"):
            rows = [json.loads(l) for l in open(src / f"{variant}.train.jsonl")]
            keep = [r for r in rows if r["expert_id"] in grp]
            (o / f"{variant}.train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
            shutil.copy(src / f"{variant}.test.jsonl", o / f"{variant}.test.jsonl")   # full test both
        shutil.copy(src / "experts.json", o / "experts.json")
        names = [experts["names"][i] for i in sorted(grp)]
        print(f"task {tag} -> {out}   personas {sorted(grp)} = {names}   train rows: "
              f"{sum(1 for r in [json.loads(l) for l in open(o/'em.train.jsonl')])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
