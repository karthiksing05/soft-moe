"""The EM training loop.

- **E-step (online):** the router produces responsibilities ``r_ik`` each batch (soft posterior,
  or argmax for hard EM; one-hot label for the supervised router).
- **M-step:** fix ``r``, take a gradient step on ``Σ_i Σ_k r_ik · NLL_ik`` + regularizers, updating
  the expert tokens (and backbone if not frozen) and the router.
- **Periodic hard reassignment (optional, ``em.reassign_every``):** sweep a sample of the corpus,
  compute the *likelihood-based* E-step ``argmin_k NLL_ik`` per example, and amortize it into the
  learned router (CE) — the c-BTM-flavored discovery loop. Logs assignment↔domain NMI each sweep.

Deterministic: seeds everything, dumps ``resolved_config.yaml`` + git SHA, logs the expert
utilization histogram every N steps (the primary "is it specializing?" signal).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from softmoe.data.dataset import Collator, MixedDomainSampler, SoftMoEDataset
from softmoe.models.router import SoftRouter
from softmoe.training.callbacks import CheckpointManager
from softmoe.training.losses import causal_lm_loss, combine_losses, router_loss
from softmoe.utils.config import save_resolved_config
from softmoe.utils.logging import MetricLogger, get_logger
from softmoe.utils.seeding import git_sha, seed_everything

logger = get_logger()


class EMTrainer:
    def __init__(self, cfg, run_dir: str | Path, device: str | None = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tcfg = cfg["train"]
        self.metric_logger = MetricLogger(self.run_dir, cfg.get_path("train.wandb"))

    # ---- setup -----------------------------------------------------------------------
    def _build_optim(self, model):
        ocfg = self.tcfg.get("optimizer", {"name": "adamw", "lr": 1e-3, "wd": 0.0})
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters — check backbone_mode / expert_tokens.trainable.")
        opt = torch.optim.AdamW(params, lr=float(ocfg.get("lr", 1e-3)), weight_decay=float(ocfg.get("wd", 0.0)))
        max_steps = int(self.tcfg.get("max_steps", 100))
        if self.tcfg.get("scheduler", "cosine") == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)
        else:
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        return opt, sched

    # ---- alternating (block-coordinate) M-step ---------------------------------------
    def _partition_params(self, model):
        """Split trainable params into (backbone θ, expert-token/router) groups."""
        backbone = list(getattr(model, "backbone", torch.nn.Module()).parameters())
        backbone_ids = {id(p) for p in backbone}
        backbone = [p for p in backbone if p.requires_grad]
        expert = [p for p in model.parameters() if id(p) not in backbone_ids and p.requires_grad]
        return backbone, expert

    def _build_alternating(self, model, max_steps: int, alt: dict):
        """Two optimizers (backbone vs expert tokens) for the alternation schedule."""
        backbone, expert = self._partition_params(model)
        if not backbone:
            raise ValueError(
                "train.alternation needs a trainable backbone — set model.backbone_mode: full "
                "(alternation updates the LLM θ, which a frozen backbone forbids)."
            )
        if not expert:
            raise ValueError("train.alternation found no expert-token params to alternate with.")
        bb_steps = int(alt.get("backbone_steps", 200))
        tok_steps = int(alt.get("token_steps", 200))
        cycle = bb_steps + tok_steps
        # cosine T_max ≈ the number of update steps each optimizer will actually take
        bb_total = max(1, int(round(max_steps * bb_steps / cycle)))
        tok_total = max(1, int(round(max_steps * tok_steps / cycle)))
        opt_bb = torch.optim.AdamW(backbone, lr=float(alt.get("backbone_lr", 5e-5)),
                                   weight_decay=float(alt.get("backbone_wd", 0.0)))
        opt_tok = torch.optim.AdamW(expert, lr=float(alt.get("token_lr", 1e-2)),
                                    weight_decay=float(alt.get("token_wd", 0.0)))
        sched_bb = torch.optim.lr_scheduler.CosineAnnealingLR(opt_bb, T_max=bb_total)
        sched_tok = torch.optim.lr_scheduler.CosineAnnealingLR(opt_tok, T_max=tok_total)
        logger.info("[alternation] backbone %d params (lr %.1e, %d-step blocks) ⇄ "
                    "tokens %d params (lr %.1e, %d-step blocks), start=%s",
                    sum(p.numel() for p in backbone), float(alt.get("backbone_lr", 5e-5)), bb_steps,
                    sum(p.numel() for p in expert), float(alt.get("token_lr", 1e-2)), tok_steps,
                    alt.get("start", "backbone"))
        return {"opt_bb": opt_bb, "opt_tok": opt_tok, "sched_bb": sched_bb, "sched_tok": sched_tok,
                "backbone": backbone, "expert": expert, "bb_steps": bb_steps, "tok_steps": tok_steps,
                "start": alt.get("start", "backbone"), "cur": None}

    @staticmethod
    def _phase_for_step(step: int, bb_steps: int, tok_steps: int, start: str) -> str:
        pos = (step - 1) % (bb_steps + tok_steps)
        first, second = ("backbone", "tokens") if start == "backbone" else ("tokens", "backbone")
        return first if pos < (bb_steps if start == "backbone" else tok_steps) else second

    def _set_phase(self, groups: dict, phase: str) -> None:
        """Freeze the inactive group so grads (and updates) only reach the active one."""
        if groups["cur"] == phase:
            return
        train_bb = phase == "backbone"
        for p in groups["backbone"]:
            p.requires_grad_(train_bb)
        for p in groups["expert"]:
            p.requires_grad_(not train_bb)
        groups["cur"] = phase

    @torch.no_grad()
    def _inject_token_noise(self, model, std: float, phase) -> None:
        """Add Gaussian noise to the expert-token embeddings after a token update (thesis Phase B).

        Applied only when the tokens are the thing being trained (joint token-only Phase B where
        ``phase is None``, or the token sub-phase of an alternation) and only to trainable params."""
        if phase == "backbone":
            return
        emb = getattr(getattr(model, "tokens", None), "embeddings", None)
        if emb is not None and emb.requires_grad:
            emb.add_(torch.randn_like(emb) * std)

    def _loader(self, ds: SoftMoEDataset, train: bool):
        bs = int(self.tcfg.get("batch_size", 8))
        pad = int(self.cfg.get_path("data.pad_token_id", 0) or 0)
        collate = Collator(pad_token_id=pad)
        if train and self.tcfg.get("balanced_sampler", False):
            sampler = MixedDomainSampler(
                ds.domain_id[ds.index], balanced=True,
                num_samples=bs * int(self.tcfg.get("max_steps", 100)),
                seed=int(self.cfg.get("seed", 0)),
            )
            return DataLoader(ds, batch_size=bs, sampler=sampler, collate_fn=collate)
        return DataLoader(ds, batch_size=bs, shuffle=train, collate_fn=collate, drop_last=False)

    def _to_device(self, batch: dict) -> dict:
        return {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    # ---- main loop -------------------------------------------------------------------
    def fit(self, model, train_ds: SoftMoEDataset, val_ds: SoftMoEDataset | None = None):
        from softmoe.models.soft_moe import SoftMoE

        seed_everything(int(self.cfg.get("seed", 0)))
        save_resolved_config(self.cfg, self.run_dir / "resolved_config.yaml")
        (self.run_dir / "git_sha.txt").write_text(git_sha() + "\n")

        model.to(self.device)
        max_steps = int(self.tcfg.get("max_steps", 100))
        alt = dict(self.tcfg.get("alternation", {}))
        use_alt = bool(alt.get("enabled", False))
        if use_alt:
            groups = self._build_alternating(model, max_steps, alt)
            opt = sched = None
        else:
            groups = None
            opt, sched = self._build_optim(model)

        ckpt = CheckpointManager(self.run_dir, keep_last=int(self.tcfg.get("keep_last", 1)))
        accum = int(self.tcfg.get("grad_accum", 1))
        clip = float(self.tcfg.get("grad_clip", 1.0))
        eval_every = int(self.tcfg.get("eval_every", max(1, max_steps // 4)))
        log_every = int(self.tcfg.get("log_every", max(1, max_steps // 20)))
        lambdas = dict(self.tcfg.get("lambdas", {}))
        # thesis Phase B: inject noise into the persona/expert embeddings after each update to
        # prevent all embeddings collapsing to the same vector (Xie et al. 2020, two-time-scale).
        noise_std = float(self.tcfg.get("token_noise_std", 0.0))
        em = dict(self.tcfg.get("em", {"mode": "soft"}))
        em_hard = em.get("mode") == "hard"
        reassign_every = int(em.get("reassign_every", 0))

        loader = self._loader(train_ds, train=True)
        data_iter = _cycle(loader)
        model.train()
        logger.info("[train] %d steps on %s | mode: %s | trainable params: %d",
                    max_steps, self.device, "alternating θ⇄tokens" if use_alt else "joint",
                    sum(p.numel() for p in model.parameters() if p.requires_grad))

        for step in range(1, max_steps + 1):
            if reassign_every and step % reassign_every == 0 and isinstance(model, SoftMoE):
                self._nll_reassign(model, train_ds, opt, em, step)

            phase = None
            if use_alt:
                phase = self._phase_for_step(step, groups["bb_steps"], groups["tok_steps"], groups["start"])
                self._set_phase(groups, phase)

            batch = self._to_device(next(data_iter))
            out = model(batch, em_hard=em_hard)
            total, logs = combine_losses(out, lambdas)
            (total / accum).backward()
            if step % accum == 0:
                if use_alt:
                    active_opt = groups["opt_bb"] if phase == "backbone" else groups["opt_tok"]
                    active_sched = groups["sched_bb"] if phase == "backbone" else groups["sched_tok"]
                    torch.nn.utils.clip_grad_norm_(
                        groups["backbone"] if phase == "backbone" else groups["expert"], clip)
                    active_opt.step(); active_sched.step()
                    groups["opt_bb"].zero_grad(); groups["opt_tok"].zero_grad()
                    logs["phase"] = phase
                    logs["lr"] = active_sched.get_last_lr()[0]
                else:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
                    opt.step(); sched.step(); opt.zero_grad()
                if noise_std > 0.0 and isinstance(model, SoftMoE):
                    self._inject_token_noise(model, noise_std, phase)

            if step % log_every == 0 or step == 1:
                if not use_alt:
                    logs["lr"] = sched.get_last_lr()[0]
                elif "phase" not in logs:
                    logs["phase"] = phase
                util = self._utilization(out)
                if util is not None:
                    logs["dead_experts"] = int((util == 0).sum())
                self.metric_logger.log(logs, step=step)
                logger.info("[step %d] %s", step, _fmt(logs))

            if val_ds is not None and (step % eval_every == 0 or step == max_steps):
                val_ppl = self.validate(model, val_ds, em_hard)
                self.metric_logger.log({"val_ppl": val_ppl}, step=step)
                logger.info("[step %d] val_ppl=%.3f", step, val_ppl)
                ckpt.save(model, None if use_alt else opt, None if use_alt else sched, step, metric=val_ppl)

        if val_ds is None:
            ckpt.save(model, None if use_alt else opt, None if use_alt else sched, max_steps,
                      metric=None, is_best=True)
        self.metric_logger.close()
        return ckpt

    # ---- E-step: likelihood-based hard reassignment + router amortization -------------
    @torch.no_grad()
    def _nll_assignment(self, model: SoftMoE, batch) -> torch.Tensor:
        K = model.tokens.n_experts
        nll = torch.stack(
            [model._single_expert_forward(batch, torch.full_like(batch["domain_id"], k))[0] for k in range(K)],
            dim=1,
        )                                                       # [B, K]
        return nll.argmin(dim=1)

    def _nll_reassign(self, model: SoftMoE, train_ds, opt, em, step) -> None:
        n_batches = int(em.get("reassign_batches", 4))
        inner_steps = int(em.get("reassign_steps", 1))
        loader = self._loader(train_ds, train=True)
        it = _cycle(loader)
        labels_all, domains_all = [], []
        model.eval()
        buf = []
        for _ in range(n_batches):
            b = self._to_device(next(it))
            assign = self._nll_assignment(model, b)
            buf.append((b, assign))
            labels_all.append(assign.cpu().numpy())
            domains_all.append(b["domain_id"].cpu().numpy())
        model.train()
        # Amortize into the learned router (CE toward the NLL E-step labels).
        if isinstance(model.router, SoftRouter):
            for _ in range(inner_steps):
                for b, assign in buf:
                    route = model.route(b)
                    if route.logits is None:
                        continue
                    loss = router_loss(route.logits, torch.nn.functional.one_hot(assign, model.tokens.n_experts).float())
                    opt.zero_grad(); loss.backward(); opt.step()
        from sklearn.metrics import normalized_mutual_info_score

        nmi = normalized_mutual_info_score(np.concatenate(domains_all), np.concatenate(labels_all))
        self.metric_logger.log({"reassign_nmi_vs_domain": float(nmi)}, step=step)
        logger.info("[step %d] hard reassignment: assign↔domain NMI=%.3f", step, nmi)

    # ---- validation ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, model, val_ds, em_hard: bool, max_batches: int = 20) -> float:
        model.eval()
        loader = self._loader(val_ds, train=False)
        nlls = []
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = self._to_device(batch)
            out = model(batch, em_hard=em_hard)
            nlls.append(out["per_example_nll"].detach())
        model.train()
        if not nlls:
            return float("nan")
        mean_nll = torch.cat(nlls).mean().item()
        return float(math.exp(min(mean_nll, 20.0)))

    @staticmethod
    def _utilization(out: dict):
        route = out["aux"].get("route_info")
        if route is None:
            return None
        ids = route.expert_ids.detach().cpu().numpy()
        K = route.responsibilities.shape[1]
        return np.bincount(ids, minlength=K)


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def _fmt(logs: dict) -> str:
    return " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items())
