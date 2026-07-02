#!/usr/bin/env python
"""T-ablation table + figure: all 4 ours variants x tokens_per_expert in {1,2,4}.

T=1 = the sc_ours_<method> runs; T=2,4 = tab_ours_<method>_T<t>. Reads metrics.json macro-ppl.
Usage: python scripts/make_t_ablation.py --exp <experiments-dir> --out <report-dir>
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["sup_seq", "unsup_seq"]  # alternating dropped (not in the thesis)
Ts = [1, 2, 4]


def macro(run: Path):
    mj = run / "metrics.json"
    if not mj.exists():
        return None
    return json.loads(mj.read_text()).get("lm_learned", {}).get("macro_ppl")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", default="reports/scaled")
    a = ap.parse_args(argv)
    exp, out = Path(a.exp), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "comparison_figures").mkdir(exist_ok=True)

    data = {}  # method -> {T: macro}
    for m in METHODS:
        data[m] = {}
        for T in Ts:
            run = exp / (f"sc_ours_{m}" if T == 1 else f"tab_ours_{m}_T{T}")
            v = macro(run)
            if v is not None:
                data[m][T] = v

    L = ["# T-ablation — tokens_per_expert ∈ {1,2,4} × 4 ours variants (d512)", "",
         "macro-perplexity ↓ (T=1 is the main sc_ours run).", "",
         "| method | T=1 | T=2 | T=4 | best |", "|---|---|---|---|---|"]
    for m in METHODS:
        cells = [f"{data[m].get(T):.3f}" if data[m].get(T) is not None else "—" for T in Ts]
        avail = {T: data[m][T] for T in Ts if T in data[m]}
        best = f"T={min(avail, key=avail.get)}" if avail else "—"
        L.append(f"| {m} | {cells[0]} | {cells[1]} | {cells[2]} | {best} |")
    (out / "t_ablation.md").write_text("\n".join(L) + "\n")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m in METHODS:
        xs = [T for T in Ts if T in data[m]]
        ys = [data[m][T] for T in xs]
        if xs:
            ax.plot(xs, ys, "o-", label=m)
    ax.set_xticks(Ts); ax.set_xlabel("tokens per expert (T)"); ax.set_ylabel("macro-ppl ↓")
    ax.set_title("T-ablation: does more than one expert vector help?"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "comparison_figures" / "11_t_ablation.png", dpi=130); plt.close(fig)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
