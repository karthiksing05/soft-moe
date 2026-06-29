"""Stage 1: pull/generate raw per-domain documents.

Two source kinds, selected per domain by ``source``:

- ``hf``         — stream a HuggingFace dataset (``datasets.load_dataset(streaming=True)``),
                   take up to ``max_docs`` docs, drop docs shorter than ``min_chars``.
- ``synthetic``  — deterministic local generators (no network), used by the offline ``synth``
                   recipe and the test suite. THEORY §E.1 endorses controlled domain mixtures of
                   maximally-distinct generators as the identifiability diagnostic.

Output: per-domain JSON Lines at ``<raw_dir>/<domain>.jsonl`` with ``{"text", "domain"}``.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterator, Mapping

from softmoe.utils.logging import get_logger

logger = get_logger()

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", text).strip()


# --------------------------------------------------------------------------------------
# Synthetic generators (deterministic, offline) — one near-orthogonal "domain" each.
# --------------------------------------------------------------------------------------
_WORDS = (
    "the a model learns from data and tokens form a sequence of words that carry "
    "meaning across a corpus while experts specialize on domains over time".split()
)
_IDENTS = ["x", "y", "z", "acc", "total", "buf", "node", "value", "count", "result"]


def _gen_english(rng: random.Random) -> str:
    n = rng.randint(12, 30)
    words = [rng.choice(_WORDS) for _ in range(n)]
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def _gen_code(rng: random.Random) -> str:
    a, b = rng.choice(_IDENTS), rng.choice(_IDENTS)
    op = rng.choice(["+", "-", "*"])
    lines = [
        f"def f({a}, {b}):",
        f"    {rng.choice(_IDENTS)} = {a} {op} {b}",
        f"    for i in range({rng.randint(1, 9)}):",
        f"        {a} = {a} + i",
        f"    return {a}",
    ]
    return "\n".join(lines)


def _gen_arithmetic(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randint(3, 8)):
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        op = rng.choice(["+", "-", "*"])
        c = eval(f"{a}{op}{b}")  # noqa: S307 - inputs are local ints, safe
        parts.append(f"{a} {op} {b} = {c}")
    return " ; ".join(parts)


def _gen_formal(rng: random.Random) -> str:
    syms = "ABCDEF"
    n = rng.randint(10, 24)
    return " ".join(f"{rng.choice(syms)}{rng.randint(0, 3)}" for _ in range(n))


_GENERATORS = {
    "english": _gen_english,
    "code": _gen_code,
    "arithmetic": _gen_arithmetic,
    "formal": _gen_formal,
}


def _synthetic_docs(generator: str, max_docs: int, seed: int) -> Iterator[str]:
    if generator not in _GENERATORS:
        raise ValueError(f"Unknown synthetic generator '{generator}'. Choose from {list(_GENERATORS)}.")
    fn = _GENERATORS[generator]
    rng = random.Random(seed)
    for _ in range(max_docs):
        yield fn(rng)


def _hf_docs(domain: Mapping, min_chars: int) -> Iterator[str]:
    from datasets import load_dataset

    name = domain["hf_path"]
    config = domain.get("hf_config")
    split = domain.get("split", "train")
    text_field = domain.get("text_field", "text")
    max_docs = int(domain.get("max_docs", 1000))

    logger.info("Streaming HF dataset %s (config=%s, split=%s)", name, config, split)
    ds = load_dataset(name, config, split=split, streaming=True)
    taken = 0
    for row in ds:
        if taken >= max_docs:
            break
        text = row.get(text_field)
        if not isinstance(text, str):
            continue
        text = _normalize(text)
        if len(text) < min_chars:
            continue
        yield text
        taken += 1


def download_domain(domain: Mapping, raw_dir: Path, seed: int, force: bool = False) -> Path:
    """Materialize one domain's documents to ``<raw_dir>/<name>.jsonl``. Cached unless ``force``."""
    name = domain["name"]
    out_path = raw_dir / f"{name}.jsonl"
    if out_path.exists() and not force:
        logger.info("[download] %s cached (%s)", name, out_path)
        return out_path

    raw_dir.mkdir(parents=True, exist_ok=True)
    source = domain.get("source", "hf")
    min_chars = int(domain.get("min_chars", 1))
    max_docs = int(domain.get("max_docs", 1000))

    if source == "synthetic":
        docs = _synthetic_docs(domain["generator"], max_docs, seed + hash(name) % 9973)
    elif source == "hf":
        docs = _hf_docs(domain, min_chars)
    else:
        raise ValueError(f"Unknown domain source '{source}' (use 'hf' or 'synthetic').")

    n = 0
    with open(out_path, "w") as fh:
        for text in docs:
            text = _normalize(text)
            if len(text) < min_chars:
                continue
            fh.write(json.dumps({"text": text, "domain": name}) + "\n")
            n += 1
    logger.info("[download] %s -> %d docs (%s)", name, n, out_path)
    if n == 0:
        raise RuntimeError(f"Domain '{name}' produced 0 documents; check the source config.")
    return out_path
