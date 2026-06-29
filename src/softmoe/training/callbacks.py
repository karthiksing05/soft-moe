"""Training callbacks: checkpointing (tokens + router + optional backbone/LoRA + optimizer + RNG)."""

from __future__ import annotations

from pathlib import Path

import torch

from softmoe.utils.logging import get_logger

logger = get_logger()


class CheckpointManager:
    def __init__(self, run_dir: str | Path, keep_last: int = 1):
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.best_metric = float("inf")
        self._saved: list[Path] = []

    def save(self, model, optimizer, scheduler, step: int, metric: float | None = None,
             is_best: bool = False) -> Path:
        state = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "torch_rng": torch.get_rng_state(),
            "metric": metric,
        }
        path = self.ckpt_dir / f"step_{step}.pt"
        torch.save(state, path)
        self._saved.append(path)
        self._prune()
        if is_best or (metric is not None and metric < self.best_metric):
            self.best_metric = metric if metric is not None else self.best_metric
            best = self.ckpt_dir / "best.pt"
            torch.save(state, best)
            logger.info("[ckpt] new best @ step %d (metric=%.4f) -> %s", step, metric or -1, best)
        return path

    def _prune(self) -> None:
        while len(self._saved) > self.keep_last:
            old = self._saved.pop(0)
            try:
                old.unlink()
            except FileNotFoundError:  # pragma: no cover
                pass

    @staticmethod
    def load(path: str | Path, model, optimizer=None, scheduler=None, map_location="cpu") -> int:
        state = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(state["model"])
        if optimizer is not None and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        return int(state.get("step", 0))
