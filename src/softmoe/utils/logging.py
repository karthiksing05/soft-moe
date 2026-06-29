"""Logging utilities: a console logger plus an optional W&B-or-jsonl metric logger.

W&B is gated behind a flag (``train.wandb.enabled``) and degrades gracefully to a local
``metrics.jsonl`` when the package is absent or disabled, so no run ever depends on it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "softmoe", level: int = logging.INFO) -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.propagate = False
    _LOGGERS[name] = logger
    return logger


class MetricLogger:
    """Writes scalar metrics to a local jsonl and (optionally) mirrors to W&B."""

    def __init__(self, run_dir: str | Path, wandb_cfg: Mapping[str, Any] | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self._fh = open(self.jsonl_path, "a")
        self._wandb = None
        wandb_cfg = wandb_cfg or {}
        if wandb_cfg.get("enabled"):
            try:
                import wandb

                self._wandb = wandb
                wandb.init(
                    project=wandb_cfg.get("project", "softmoe"),
                    name=wandb_cfg.get("name"),
                    dir=str(self.run_dir),
                    config=dict(wandb_cfg.get("config", {})),
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                get_logger().warning("W&B disabled (%s); logging to jsonl only.", exc)
                self._wandb = None

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        record = dict(metrics)
        if step is not None:
            record["step"] = step
        self._fh.write(json.dumps(record, default=float) + "\n")
        self._fh.flush()
        if self._wandb is not None:  # pragma: no cover - optional dependency
            self._wandb.log(dict(metrics), step=step)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # pragma: no cover
            pass
        if self._wandb is not None:  # pragma: no cover - optional dependency
            self._wandb.finish()
