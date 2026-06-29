"""``evaluate`` — run the metric suite over a trained run, writing ``metrics.json``. (M6)"""

from __future__ import annotations

import argparse

from softmoe.eval.harness import evaluate_run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a trained run.")
    ap.add_argument("--run", required=True, help="experiments/<run> directory")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    metrics = evaluate_run(args.run, data_root=args.data_root, device=args.device)
    lm = metrics.get("lm_learned", {})
    print(f"macro-ppl={lm.get('macro_ppl'):.3f}  micro-ppl={lm.get('micro_ppl'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
