#!/usr/bin/env python
"""Capacity figure: MoE *granularity* lever vs governance *rank* lever (what the MoE buys).

Reads reports/capacity/results.csv (method, macro_ppl, total_params) and plots quality vs total
params, one series for the MoE granularity ladder (G1..G8) and one for the governance rank ladder
(r16..r128), with dense + prefix reference lines. Run: python scripts/make_capacity_figure.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "reports" / "capacity"

# method -> (family, lever-label, lever-order)
MOE = {"moe_g1": ("G1", 1), "moe_g2": ("G2", 2), "moe_g4": ("G4", 4), "moe_g8": ("G8", 8)}
GOV = {"govern_spectralA_d512": ("r16", 16), "govern_spectralA_r64_d512": ("r64", 64),
       "govern_spectralA_r128_d512": ("r128", 128)}


def load():
    rows = {}
    with open(CAP / "results.csv") as fh:
        for r in csv.DictReader(fh):
            rows[r["method"]] = {"ppl": float(r["macro_ppl"]), "tot": float(r["total_params"])}
    return rows


def main() -> int:
    R = load()
    fig, (axg, axr) = plt.subplots(1, 2, figsize=(11, 4.3))
    dense = R.get("dense_1x", {}).get("ppl")
    prefix = R.get("ours_sup_seq", {}).get("ppl")

    # left: MoE granularity ladder
    ms = [(lbl, o, R[m]["ppl"]) for m, (lbl, o) in MOE.items() if m in R]
    ms.sort(key=lambda t: t[1])
    if ms:
        axg.plot([o for _, o, _ in ms], [p for _, _, p in ms], "s-", color="tab:red", label="MoE (143M params)")
        for lbl, o, p in ms:
            axg.annotate(f"{lbl}\n{p:.3f}", (o, p), fontsize=8, ha="center", va="bottom")
    if dense: axg.axhline(dense, color="gray", ls="--", lw=0.9, label=f"dense {dense:.3f}")
    axg.set_xscale("log", base=2); axg.set_xlabel("MoE granularity G (log2)")
    axg.set_ylabel("macro-ppl ↓"); axg.set_title("MoE capacity lever: fine-graining"); axg.legend(fontsize=8)

    # right: governance rank ladder
    gs = [(lbl, o, R[m]["ppl"]) for m, (lbl, o) in GOV.items() if m in R]
    gs.sort(key=lambda t: t[1])
    if gs:
        axr.plot([o for _, o, _ in gs], [p for _, _, p in gs], "D-", color="tab:blue", label="governance (26M params)")
        for lbl, o, p in gs:
            axr.annotate(f"{lbl}\n{p:.3f}", (o, p), fontsize=8, ha="center", va="bottom")
    if dense: axr.axhline(dense, color="gray", ls="--", lw=0.9, label=f"dense {dense:.3f}")
    if prefix: axr.axhline(prefix, color="tab:green", ls=":", lw=0.9, label=f"prefix {prefix:.3f}")
    best_moe = min((p for _, _, p in ms), default=None)
    if best_moe: axr.axhline(best_moe, color="tab:red", ls="--", lw=0.9, label=f"best MoE {best_moe:.3f}")
    axr.set_xscale("log", base=2); axr.set_xlabel("governance rank r (log2)")
    axr.set_ylabel("macro-ppl ↓"); axr.set_title("Governance conditioning lever: subspace rank"); axr.legend(fontsize=8)

    fig.suptitle("What the MoE buys: capacity (fine-graining) vs conditioning (subspace rank)")
    fig.tight_layout()
    out = CAP / "capacity_curve.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
