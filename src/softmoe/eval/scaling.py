"""isoFLOP scaling-sweep + ablation analysis (SCALED_RECIPE.md §4.7, §6).

Reads a set of run dirs (sweep ``sw_*`` and ablation ``ab_*``), and produces:
- the **scaling curves** macro-ppl vs active params for dense / coarse-MoE(G1) / fine-MoE(G2),
  aggregated over seeds (mean±std), and the **gap = dense − MoE-G2 vs scale** (H1/H2's trend form);
- the **ablation table** at one size (granularity G, shared experts, z-loss, top-k, balancing).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load(runs: list[str]) -> list[dict]:
    rows = []
    for r in runs:
        r = Path(r)
        mj = r / "metrics.json"
        if not mj.exists():
            continue
        m = json.loads(mj.read_text())
        rows.append({
            "regime": m.get("regime", r.name),
            "macro": m.get("lm_learned", {}).get("macro_ppl"),
            "active": m.get("active_params", m.get("total_params")),
            "total": m.get("total_params"),
        })
    return rows


def _size_of(active: int) -> str:
    # bucket by active-param magnitude → the sweep size label
    for lo, name in [(1.2e6, "d128"), (5e6, "d256"), (11e6, "d384"), (20e6, "d512")]:
        if active < lo:
            return name
    return "d512"


def analyze(runs: list[str]) -> dict:
    rows = _load(runs)
    # --- sweep: regimes look like "<model>_d<NNN>" (dense/moe_g1/moe_g2) ---
    sweep = defaultdict(lambda: defaultdict(list))   # size -> model -> [macro]
    sweep_active = {}                                 # (size,model) -> active
    ablation = {}                                     # regime -> macro (single seed)
    for r in rows:
        reg = r["regime"]
        if any(reg.endswith(f"_d{d}") for d in ("128", "256", "384", "512")):
            model, size = reg.rsplit("_d", 1)
            size = "d" + size
            sweep[size][model].append(r["macro"])
            sweep_active[(size, model)] = r["active"]
        else:
            ablation[reg] = {"macro": r["macro"], "active": r["active"], "total": r["total"]}

    sizes = ["d128", "d256", "d384", "d512"]
    models = ["dense", "moe_g1", "moe_g2"]
    curve = {}
    for m in models:
        curve[m] = []
        for s in sizes:
            vals = [v for v in sweep.get(s, {}).get(m, []) if v is not None]
            if vals:
                curve[m].append({"size": s, "active": sweep_active.get((s, m)),
                                 "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                                 "n": len(vals)})
    # gap = dense - moe_g2 per size
    gap = []
    for s in sizes:
        d = [v for v in sweep.get(s, {}).get("dense", []) if v is not None]
        g2 = [v for v in sweep.get(s, {}).get("moe_g2", []) if v is not None]
        if d and g2:
            gap.append({"size": s, "active": sweep_active.get((s, "dense")),
                        "dense": float(np.mean(d)), "moe_g2": float(np.mean(g2)),
                        "gap": float(np.mean(d) - np.mean(g2))})
    return {"curve": curve, "gap": gap, "ablation": ablation, "sizes": sizes}


def render_markdown(report: dict) -> str:
    L = ["# isoFLOP scaling sweep & ablations", ""]
    L += ["## Scaling curves (macro-ppl ↓, mean±std over seeds)", "",
          "| size | active | dense | MoE-G1 (coarse) | MoE-G2 (fine) | dense−G2 gap |",
          "|---|---|---|---|---|---|"]
    gap_by_size = {g["size"]: g for g in report["gap"]}
    for s in report["sizes"]:
        def cell(m):
            pts = [c for c in report["curve"].get(m, []) if c["size"] == s]
            if not pts:
                return "—", None
            p = pts[0]
            return (f"{p['mean']:.3f}±{p['std']:.3f}" if p["n"] > 1 else f"{p['mean']:.3f}"), p["active"]
        dcell, active = cell("dense")
        g1cell, _ = cell("moe_g1")
        g2cell, _ = cell("moe_g2")
        gp = gap_by_size.get(s, {}).get("gap")
        gps = f"{gp:+.3f}" if gp is not None else "—"
        act = f"{active/1e6:.1f}M" if active else "—"
        L.append(f"| {s} | {act} | {dcell} | {g1cell} | {g2cell} | {gps} |")

    L += ["", "## Gap trend (does the fine-grained MoE advantage widen with scale? — H1/H2)", ""]
    for g in report["gap"]:
        L.append(f"- **{g['size']}** ({g['active']/1e6:.1f}M active): dense {g['dense']:.3f} − "
                 f"MoE-G2 {g['moe_g2']:.3f} = **{g['gap']:+.3f}**")

    if report["ablation"]:
        L += ["", "## Ablation matrix (d256, 20k steps, single seed)", "",
              "| variant | macro-ppl ↓ | total params |", "|---|---|---|"]
        for reg in sorted(report["ablation"]):
            a = report["ablation"][reg]
            tp = f"{a['total']/1e6:.1f}M" if a.get("total") else "—"
            L.append(f"| {reg} | {a['macro']:.3f} | {tp} |")
    return "\n".join(L) + "\n"


def make_plots(report: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    # 1. scaling curves
    fig, ax = plt.subplots()
    for m, label, mk in [("dense", "Dense", "o"), ("moe_g1", "MoE-G1 coarse", "s"),
                          ("moe_g2", "MoE-G2 fine", "^")]:
        pts = [c for c in report["curve"].get(m, []) if c["active"]]
        if pts:
            xs = [p["active"] / 1e6 for p in pts]
            ys = [p["mean"] for p in pts]
            es = [p["std"] for p in pts]
            ax.errorbar(xs, ys, yerr=es, marker=mk, label=label, capsize=3)
    ax.set_xscale("log"); ax.set_xlabel("active params (M, log)"); ax.set_ylabel("macro-ppl ↓")
    ax.set_title("isoFLOP scaling: dense vs coarse vs fine MoE"); ax.legend()
    fig.savefig(out_dir / "scaling_curves.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # 2. gap vs scale
    if report["gap"]:
        fig, ax = plt.subplots()
        xs = [g["active"] / 1e6 for g in report["gap"]]
        ys = [g["gap"] for g in report["gap"]]
        ax.plot(xs, ys, "o-", color="tab:green")
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel("active params (M, log)")
        ax.set_ylabel("dense − MoE-G2 (ppl; >0 = MoE better)")
        ax.set_title("Fine-grained MoE advantage vs scale (H1/H2)")
        fig.savefig(out_dir / "gap_vs_scale.png", dpi=120, bbox_inches="tight"); plt.close(fig)
