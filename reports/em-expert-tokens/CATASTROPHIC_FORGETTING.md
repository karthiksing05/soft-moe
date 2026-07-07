# Catastrophic forgetting — sequential full-FT vs EM (cycling / token-only)

A classic continual-learning test: learn **Task A**, then learn **Task B**, and measure how much Task A
degrades. Does the EM expert-token method forget less than naive sequential finetuning?

**Setup (Qwen2.5-3B, persona).** Task A = {pirate, bard, professor, teen}; Task B = {detective, child, coach,
robot} (1824 train examples each, disjoint personas). We evaluate on the *full* 8-persona held-out set at
every stage, so per-persona ppl reads Task-A **retention** and Task-B **acquisition** directly. Three
sequential methods, differing only in **how much the backbone is updated while learning Task B**:

| method | backbone steps on B | mechanism |
|---|---|---|
| sequential full-FT | 1000 | full fine-tune on B (standard) |
| EM-cycling on B | 500 | alternate Phase-A (backbone) ⇄ Phase-B (token), N=2×250 |
| EM token-only on B | **0** | Phase-B only — train the new personas' tokens, **backbone frozen** |

Plus a **joint** reference (all 8 personas trained together — the no-forgetting ceiling).

## Result

| method | Task A: after A → after B | **A forgetting** | Task B learned (ppl) |
|---|---|---|---|
| joint (ceiling) | — → 3.65 | 0% | 4.21 |
| sequential full-FT | 3.69 → 4.97 | **+35%** | 4.52 |
| EM-cycling on B | 3.73 → 5.16 | **+38%** | **4.22** (best, ≈ ceiling) |
| EM token-only on B | 3.73 → 3.88 | **+4%** | 5.41 (worst) |

![catastrophic forgetting](figs/catastrophic_forgetting.png)

## Findings

1. **Sequential full-FT catastrophically forgets.** Learning Task B drives Task-A ppl up **+35%** (3.69 → 4.97)
   — the shared backbone is overwritten toward B's personas. This is the standard failure mode, reproduced.
2. **Freezing the backbone (EM token-only) essentially eliminates forgetting** — Task A rises only **+4%**
   (3.73 → 3.88, near the joint ceiling). This is *parameter isolation*: each new task is a new token, and the
   backbone that holds Task A is never touched. But it **learns Task B least well** (5.41 vs ceiling 4.21) —
   a frozen backbone can only be steered so far by an embedding.
3. **Cycling on B does NOT reduce forgetting — but it learns the new task best.** EM-cycling forgets Task A
   **as much as full-FT (+38%)** — its Phase-A cycles still update the shared backbone — yet it **acquires
   Task B best of all (4.22, matching the joint-trained ceiling, beating even full-FT's 4.52).** The
   alternation that helped as a *regulariser* elsewhere does not help *retention* here.

## The takeaway — a clean retention/plasticity trade-off

The anti-forgetting benefit comes specifically from **freezing the backbone (token-only isolation)**, not from
the EM *alternation*. Touching the backbone at all — whether by full-FT or by cycling — forgets the old task;
cycling merely spends those backbone updates most efficiently on the *new* task.

| you want… | use | cost |
|---|---|---|
| **retain old tasks** | EM token-only (frozen backbone) | new task learned less well |
| **learn the new task best** | EM-cycling (or full-FT) | old tasks forgotten |

The practical value of the expert-token framework for continual learning is that it *offers* the token-only
option — parameter isolation that naive SFT does not have — turning catastrophic forgetting (+35%) into near-zero
(+4%) when retention matters. (This continual-learning angle is an extension beyond the persona-only thesis, but
it follows directly from its per-speaker-token representation.)

*(Single seed, 3B, 4+4 personas. `make_task_split.py` builds A/B; drivers `catforget.sbatch` +
`catforget_cyc.sbatch`; figure `make_cf_fig.py`.)*
