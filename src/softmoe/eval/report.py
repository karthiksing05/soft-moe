"""Aggregate many runs' ``metrics.json`` into the comparison table + CSV + figures.

Produces ``<out>/main_table.md`` and ``<out>/results.csv`` (rows = methods/regimes), plus
matplotlib figures: quality-vs-added-params, expert×domain contingency heatmaps, utilization
histograms, and the oracle-vs-learned routing-gap bar chart.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from softmoe.utils.logging import get_logger

logger = get_logger()

_COLUMNS = [
    ("method", "method"),
    ("macro_ppl", "macro-ppl ↓"),
    ("micro_ppl", "micro-ppl ↓"),
    ("routing_nmi", "routing-NMI ↑"),
    ("routing_acc", "routing-acc ↑"),
    ("util_entropy", "util-entropy ↑"),
    ("separation", "sep ↑"),
    ("swap_ratio", "swap-ratio ↑"),
    ("added_params", "+params ↓"),
]


def _row_from_metrics(m: dict) -> dict:
    lm = m.get("lm_learned", {})
    routing = m.get("routing_vs_domain", {})
    util = m.get("utilization", {})
    sep = m.get("token_separation", {})
    swap = m.get("swap_test", {})
    return {
        "method": m.get("regime") or m.get("method", "?"),
        "macro_ppl": lm.get("macro_ppl", float("nan")),
        "micro_ppl": lm.get("micro_ppl", float("nan")),
        "routing_nmi": routing.get("nmi", float("nan")),
        "routing_acc": routing.get("routing_accuracy", float("nan")),
        "util_entropy": util.get("utilization_entropy_norm", float("nan")),
        "separation": sep.get("mean_pairwise_cosine_distance", float("nan")),
        "swap_ratio": swap.get("swap_ratio", float("nan")),
        "added_params": m.get("added_trainable_params", 0),
    }


def collect_runs(runs_dir: str | Path) -> list[dict]:
    runs_dir = Path(runs_dir)
    metrics = []
    for mj in sorted(runs_dir.glob("*/metrics.json")):
        try:
            with open(mj) as fh:
                m = json.load(fh)
            m["_run"] = mj.parent.name
            metrics.append(m)
        except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover
            logger.warning("skipping %s (%s)", mj, exc)
    return metrics


def _fmt(v) -> str:
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.3f}"
    return str(v)


def make_table(rows: list[dict]) -> str:
    headers = [label for _, label in _COLUMNS]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(key)) for key, _ in _COLUMNS) + " |")
    return "\n".join(lines)


def make_report(runs_dir: str | Path, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = collect_runs(runs_dir)
    if not metrics:
        logger.warning("No metrics.json found under %s.", runs_dir)
    rows = [_row_from_metrics(m) for m in metrics]

    table = make_table(rows)
    (out_dir / "main_table.md").write_text("# Soft-MoE — comparison table\n\n" + table + "\n")
    _write_csv(rows, out_dir / "results.csv")

    try:
        _make_figures(metrics, rows, fig_dir)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        logger.warning("figure generation failed: %s", exc)

    logger.info("[report] wrote %s and %s", out_dir / "main_table.md", out_dir / "results.csv")
    return {"rows": rows, "table": table}


def _write_csv(rows: list[dict], path: Path) -> None:
    import csv

    keys = [key for key, _ in _COLUMNS]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def _make_figures(metrics: list[dict], rows: list[dict], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. quality vs added params
    fig, ax = plt.subplots()
    for r in rows:
        ax.scatter(max(r["added_params"], 1), r["macro_ppl"], label=r["method"])
        ax.annotate(r["method"], (max(r["added_params"], 1), r["macro_ppl"]), fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("added trainable params (log)"); ax.set_ylabel("macro-ppl ↓")
    ax.set_title("Quality vs added params"); fig.savefig(fig_dir / "quality_vs_params.png", dpi=120)
    plt.close(fig)

    # 2. contingency heatmaps + 3. utilization histograms
    for m in metrics:
        name = m.get("regime") or m.get("method", "run")
        cont = m.get("contingency_expert_by_domain")
        if cont:
            fig, ax = plt.subplots()
            ax.imshow(np.array(cont), aspect="auto", cmap="viridis")
            ax.set_xlabel("true domain"); ax.set_ylabel("expert"); ax.set_title(f"contingency: {name}")
            fig.savefig(fig_dir / f"contingency_{name}.png", dpi=120); plt.close(fig)
        util = m.get("utilization", {}).get("utilization_counts")
        if util:
            fig, ax = plt.subplots()
            ax.bar(range(len(util)), util)
            ax.set_xlabel("expert"); ax.set_ylabel("count"); ax.set_title(f"utilization: {name}")
            fig.savefig(fig_dir / f"utilization_{name}.png", dpi=120); plt.close(fig)

    # 5. oracle-vs-learned routing gap
    gap_rows = [(m.get("regime") or m.get("method"), m.get("oracle_routed_gap"))
                for m in metrics if m.get("oracle_routed_gap") is not None]
    if gap_rows:
        fig, ax = plt.subplots()
        ax.bar([g[0] for g in gap_rows], [g[1] for g in gap_rows])
        ax.set_ylabel("macro-ppl gap (learned − oracle)"); ax.set_title("Router quality gap")
        plt.xticks(rotation=30, ha="right"); fig.tight_layout()
        fig.savefig(fig_dir / "routing_gap.png", dpi=120); plt.close(fig)
