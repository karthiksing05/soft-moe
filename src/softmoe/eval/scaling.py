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
    models = ["dense", "moe_g1", "moe_g2", "ours", "gov"]
    curve = {}
    for m in models:
        curve[m] = []
        for s in sizes:
            vals = [v for v in sweep.get(s, {}).get(m, []) if v is not None]
            if vals:
                curve[m].append({"size": s, "active": sweep_active.get((s, m)),
                                 "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                                 "n": len(vals)})
    # gap over dense, per size, for MoE-G2 and for ours (>0 = better than dense)
    def _mean(s, m):
        vs = [v for v in sweep.get(s, {}).get(m, []) if v is not None]
        return float(np.mean(vs)) if vs else None
    gap = []
    for s in sizes:
        d = _mean(s, "dense")
        if d is None:
            continue
        g2, ou, gv = _mean(s, "moe_g2"), _mean(s, "ours"), _mean(s, "gov")
        row = {"size": s, "active": sweep_active.get((s, "dense")), "dense": d}
        if g2 is not None:
            row["moe_g2"] = g2; row["gap"] = d - g2
        if ou is not None:
            row["ours"] = ou; row["gap_ours"] = d - ou
        if gv is not None:
            row["gov"] = gv; row["gap_gov"] = d - gv
        gap.append(row)
    return {"curve": curve, "gap": gap, "ablation": ablation, "sizes": sizes}


def render_markdown(report: dict) -> str:
    L = ["# isoFLOP scaling sweep & ablations", ""]
    L += ["## Scaling curves (macro-ppl ↓, mean±std over seeds)", "",
          "| size | active | dense | MoE-G1 | MoE-G2 | ours (prefix) | gov (spectral) | dense−G2 | dense−gov |",
          "|---|---|---|---|---|---|---|---|---|"]
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
        oucell, _ = cell("ours")
        gvcell, _ = cell("gov")
        g = gap_by_size.get(s, {})
        gp = f"{g['gap']:+.3f}" if g.get("gap") is not None else "—"
        gpv = f"{g['gap_gov']:+.3f}" if g.get("gap_gov") is not None else "—"
        act = f"{active/1e6:.1f}M" if active else "—"
        L.append(f"| {s} | {act} | {dcell} | {g1cell} | {g2cell} | {oucell} | {gvcell} | {gp} | {gpv} |")

    L += ["", "## Gap-over-dense trend (does the advantage widen with scale? — H1/H2)", ""]
    for g in report["gap"]:
        parts = [f"**{g['size']}** ({g['active']/1e6:.1f}M active): dense {g['dense']:.3f}"]
        if g.get("gap") is not None:
            parts.append(f"MoE-G2 {g['moe_g2']:.3f} (gap **{g['gap']:+.3f}**)")
        if g.get("gap_ours") is not None:
            parts.append(f"ours {g['ours']:.3f} (gap **{g['gap_ours']:+.3f}**)")
        L.append("- " + " · ".join(parts))

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
                          ("moe_g2", "MoE-G2 fine", "^"), ("ours", "ours prefix", "D"),
                          ("gov", "ours gov (spectral)", "P")]:
        pts = [c for c in report["curve"].get(m, []) if c["active"]]
        if pts:
            xs = [p["active"] / 1e6 for p in pts]
            ys = [p["mean"] for p in pts]
            es = [p["std"] for p in pts]
            ax.errorbar(xs, ys, yerr=es, marker=mk, label=label, capsize=3)
    ax.set_xscale("log"); ax.set_xlabel("active params (M, log)"); ax.set_ylabel("macro-ppl ↓")
    ax.set_title("isoFLOP scaling: dense vs coarse vs fine MoE"); ax.legend()
    fig.savefig(out_dir / "scaling_curves.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # 2. gap vs scale — MoE-G2 and ours, both vs dense
    if report["gap"]:
        fig, ax = plt.subplots()
        g2 = [g for g in report["gap"] if g.get("gap") is not None]
        ou = [g for g in report["gap"] if g.get("gap_ours") is not None]
        gv = [g for g in report["gap"] if g.get("gap_gov") is not None]
        if g2:
            ax.plot([g["active"] / 1e6 for g in g2], [g["gap"] for g in g2], "^-",
                    color="tab:red", label="MoE-G2 (fine)")
        if ou:
            ax.plot([g["active"] / 1e6 for g in ou], [g["gap_ours"] for g in ou], "D-",
                    color="tab:blue", label="ours prefix")
        if gv:
            ax.plot([g["active"] / 1e6 for g in gv], [g["gap_gov"] for g in gv], "P-",
                    color="tab:purple", label="ours gov (spectral)")
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel("active params (M, log)")
        ax.set_ylabel("advantage over dense (ppl; >0 = better)")
        ax.set_title("Advantage over dense vs scale (H1/H2)"); ax.legend()
        fig.savefig(out_dir / "gap_vs_scale.png", dpi=120, bbox_inches="tight"); plt.close(fig)
