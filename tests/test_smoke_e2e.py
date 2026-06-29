"""End-to-end smoke (hermetic): build synth corpus -> train two methods -> evaluate -> report.

Exercises the exact ``build_data -> train -> evaluate -> make_report`` path the README documents,
but on the offline synth recipe + tiny backbone so it runs in seconds on CPU with no network.
"""

from __future__ import annotations

import json
from pathlib import Path

from softmoe.cli import evaluate as evaluate_cli
from softmoe.cli import make_report as report_cli
from softmoe.cli import train as train_cli


def _run_method(config: str, data_root: Path, exp_root: Path) -> Path:
    run_dir = exp_root / Path(config).stem
    train_cli.main(
        [
            "--config", config,
            "--data-root", str(data_root),
            "--run-dir", str(run_dir),
            "--device", "cpu",
            "--no-eval",
            "--set", "train.max_steps=4", "train.eval_every=4", "train.log_every=2",
        ]
    )
    evaluate_cli.main(["--run", str(run_dir), "--data-root", str(data_root), "--device", "cpu"])
    return run_dir


def test_e2e_train_eval_report(tmp_path):
    data_root = tmp_path / "data"
    exp_root = tmp_path / "experiments"
    exp_root.mkdir(parents=True, exist_ok=True)

    cfg_dir = Path(__file__).resolve().parents[1] / "configs" / "experiment"
    for method in ("dense", "ours_unsup"):
        run_dir = _run_method(str(cfg_dir / f"{method}.yaml"), data_root, exp_root)
        metrics_file = run_dir / "metrics.json"
        assert metrics_file.exists()
        m = json.loads(metrics_file.read_text())
        assert "lm_learned" in m and m["lm_learned"]["macro_ppl"] > 0
        assert (run_dir / "resolved_config.yaml").exists()
        assert (run_dir / "checkpoints" / "best.pt").exists()

    out = tmp_path / "report"
    report_cli.main(["--runs", str(exp_root), "--out", str(out)])
    assert (out / "main_table.md").exists()
    assert (out / "results.csv").exists()
    table = (out / "main_table.md").read_text()
    assert "dense" in table and "ours_unsup" in table


def test_ours_unsup_specialization_signals(tmp_path):
    """Sanity that the unsupervised run produces the specialization signals (not their quality)."""
    data_root = tmp_path / "data"
    exp_root = tmp_path / "experiments"
    cfg = Path(__file__).resolve().parents[1] / "configs" / "experiment" / "ours_unsup.yaml"
    run_dir = _run_method(str(cfg), data_root, exp_root)
    m = json.loads((run_dir / "metrics.json").read_text())
    assert "token_separation" in m
    assert "swap_test" in m
    assert "utilization" in m and "utilization_entropy" in m["utilization"]
    assert "lm_oracle" in m            # SoftMoE reports oracle routing too
