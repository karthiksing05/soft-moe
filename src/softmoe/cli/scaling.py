"""``scaling`` — isoFLOP sweep + ablation analysis (table + plots)."""
from __future__ import annotations
import argparse
from pathlib import Path
from softmoe.eval.scaling import analyze, render_markdown, make_plots


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scaling-sweep + ablation analysis.")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="reports/scaling")
    args = ap.parse_args(argv)
    rep = analyze(args.runs)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "scaling.md").write_text(render_markdown(rep))
    try:
        make_plots(rep, out / "figures")
    except Exception as e:  # pragma: no cover
        print("plot failed:", e)
    print(render_markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
