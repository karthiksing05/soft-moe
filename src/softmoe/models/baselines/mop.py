"""MoP (Mixture-of-Prompts) baseline — per-domain soft prompts, routed/averaged by a learned
gate but trained **jointly** (no EM alternation, no E-step reassignment).

MoP is not a separate model class: it is ``SoftMoE`` with a learned (soft) router, ``mop_average``
(prefix = responsibility-weighted average of all expert prompts, fully differentiable), and EM
disabled in the trainer (``train.em.mode='none'``). This isolates exactly the ingredient our
method adds — EM-based discovery of the assignment — making MoP the clean "does EM beat joint
training?" control. The helper below documents/asserts that wiring; configs set it directly.
"""

from __future__ import annotations


def build_mop_config(model_cfg: dict) -> dict:
    """Return the model-config overrides that make a SoftMoE behave as MoP."""
    cfg = dict(model_cfg)
    cfg["method"] = "softmoe"
    cfg["mop_average"] = True
    router = dict(cfg.get("router", {}))
    router["kind"] = "soft"
    cfg["router"] = router
    return cfg
