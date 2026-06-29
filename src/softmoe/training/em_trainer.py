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
        opt, sched = self._build_optim(model)
        ckpt = CheckpointManager(self.run_dir, keep_last=int(self.tcfg.get("keep_last", 1)))
        max_steps = int(self.tcfg.get("max_steps", 100))
        accum = int(self.tcfg.get("grad_accum", 1))
        clip = float(self.tcfg.get("grad_clip", 1.0))
        eval_every = int(self.tcfg.get("eval_every", max(1, max_steps // 4)))
        log_every = int(self.tcfg.get("log_every", max(1, max_steps // 20)))
        lambdas = dict(self.tcfg.get("lambdas", {}))
        em = dict(self.tcfg.get("em", {"mode": "soft"}))
        em_hard = em.get("mode") == "hard"
        reassign_every = int(em.get("reassign_every", 0))

        loader = self._loader(train_ds, train=True)
        data_iter = _cycle(loader)
        model.train()
        opt.zero_grad()
        logger.info("[train] %d steps on %s | trainable params: %d",
                    max_steps, self.device, sum(p.numel() for p in model.parameters() if p.requires_grad))

        for step in range(1, max_steps + 1):
            if reassign_every and step % reassign_every == 0 and isinstance(model, SoftMoE):
                self._nll_reassign(model, train_ds, opt, em, step)

            batch = self._to_device(next(data_iter))
            out = model(batch, em_hard=em_hard)
            total, logs = combine_losses(out, lambdas)
            (total / accum).backward()
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
                opt.step(); sched.step(); opt.zero_grad()

            if step % log_every == 0 or step == 1:
                logs["lr"] = sched.get_last_lr()[0]
                util = self._utilization(out)
                if util is not None:
                    logs["dead_experts"] = int((util == 0).sum())
                self.metric_logger.log(logs, step=step)
                logger.info("[step %d] %s", step, _fmt(logs))

            if val_ds is not None and (step % eval_every == 0 or step == max_steps):
                val_ppl = self.validate(model, val_ds, em_hard)
                self.metric_logger.log({"val_ppl": val_ppl}, step=step)
                logger.info("[step %d] val_ppl=%.3f", step, val_ppl)
                ckpt.save(model, opt, sched, step, metric=val_ppl)

        if val_ds is None:
            ckpt.save(model, opt, sched, max_steps, metric=None, is_best=True)
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
