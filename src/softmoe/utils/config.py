"""Thin YAML config loader with hydra-style ``defaults`` composition.

A run is fully described by one ``configs/experiment/*.yaml``. That file may list
``defaults: [/data/toy, /model/softmoe_tiny, /train/em_soft]`` — each entry is a path
(relative to the ``configs/`` root, leading ``/`` optional) to another yaml whose contents
are deep-merged in order, and the current file's own keys win last. Dotted CLI overrides
(``--set train.max_steps=10``) are applied on top.

We deliberately avoid a hydra dependency: composition here is a small, explicit deep-merge,
which is easier to reason about and to unit-test for an autonomous build.
"""

from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


def _find_configs_root(start: Path | None = None) -> Path:
    """Locate the repo's ``configs/`` directory by walking up from ``start``/cwd."""
    here = (start or Path.cwd()).resolve()
    for cand in [here, *here.parents]:
        if (cand / "configs").is_dir():
            return cand / "configs"
    # Fallback: configs next to the installed package's repo root.
    pkg_root = Path(__file__).resolve().parents[3]
    if (pkg_root / "configs").is_dir():
        return pkg_root / "configs"
    raise FileNotFoundError("Could not locate a 'configs/' directory from cwd or package root.")


class Config(dict):
    """A dict that also supports attribute access and dotted get/set.

    ``cfg.train.max_steps`` and ``cfg["train"]["max_steps"]`` are equivalent; nested dicts
    are wrapped lazily. Use ``cfg.get_path("train.em.mode")`` for dotted lookups with default.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - mirrors AttributeError contract
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: dict = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def to_plain(self) -> dict:
        """Recursively convert to plain dicts (for yaml.safe_dump / json)."""
        return _to_plain(self)


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: Mapping) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict:
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping at top level, got {type(data)}.")
    return data


def _resolve_ref(ref: str, configs_root: Path) -> Path:
    ref = ref.strip().lstrip("/")
    if not ref.endswith((".yaml", ".yml")):
        ref += ".yaml"
    path = configs_root / ref
    if not path.exists():
        raise FileNotFoundError(f"defaults entry '{ref}' -> {path} not found.")
    return path


def _compose(path: Path, configs_root: Path, _seen: set[Path]) -> dict:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Cyclic 'defaults' reference involving {path}.")
    _seen = _seen | {path}
    raw = _read_yaml(path)
    defaults = raw.pop("defaults", []) or []
    if isinstance(defaults, str):
        defaults = [defaults]

    merged: dict = {}
    for ref in defaults:
        ref_path = _resolve_ref(ref, configs_root)
        merged = _deep_merge(merged, _compose(ref_path, configs_root, _seen))
    # The file's own keys take precedence over its defaults.
    merged = _deep_merge(merged, raw)
    return merged


def _coerce_scalar(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def apply_overrides(cfg: Config, overrides: Iterable[str]) -> Config:
    """Apply ``dotted.key=value`` overrides (values parsed as Python literals when possible)."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be of the form key.subkey=value.")
        key, _, raw = item.partition("=")
        cfg.set_path(key.strip(), _coerce_scalar(raw.strip()))
    return cfg


def load_config(
    path: str | os.PathLike,
    overrides: Iterable[str] | None = None,
    configs_root: str | os.PathLike | None = None,
) -> Config:
    """Load and fully compose a config file, applying optional dotted overrides."""
    path = Path(path)
    root = Path(configs_root) if configs_root else _find_configs_root(path.parent)
    composed = _compose(path, root, set())
    cfg = Config(composed)
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg


# ``resolve_config`` is an alias kept for readability at call sites that want to signal
# "this returns the fully-resolved config" — same behavior as load_config.
resolve_config = load_config


def save_resolved_config(cfg: Config, out_path: str | os.PathLike) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        yaml.safe_dump(cfg.to_plain(), fh, sort_keys=False)
