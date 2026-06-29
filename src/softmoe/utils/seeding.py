"""Reproducibility helpers: global seeding and git SHA capture."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed python, numpy, and torch RNGs. Returns the seed for convenience/logging."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Best-effort determinism; some ops have no deterministic kernel.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass
    return seed


def git_sha(repo_dir: str | os.PathLike | None = None, short: bool = True) -> str:
    """Return the current git commit SHA, or 'unknown' outside a repo."""
    cwd = Path(repo_dir) if repo_dir else Path.cwd()
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"
