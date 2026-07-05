#!/usr/bin/env python
"""Build reproducible 'knowledge the model won't have' data from HuggingFace Hub, in the EM PoC format.

Two sources, both fully reproducible (pinned Hub datasets):

  --source counterfact  (NeelNanda/counterfact-tracing) -> CONFLICTING variant.
        K=2 'worlds': world_0 = real (target_true), world_1 = counterfactual (target_false). Same prompt,
        the answer flips by world, and the world is signalled ONLY by the expert token. The counterfactual
        target is knowledge the model provably cannot have. Control (generic marker) sees each prompt with
        BOTH answers -> cannot disambiguate; EM per-world token can. This is the knowledge analog of persona.

  --source popqa        (akariasai/PopQA) -> RECOVERABLE (long-tail, unknown) variant.
        K relations = domains; the subject is named in the question (domain recoverable). Filtered to
        low-popularity subjects (s_pop < --max-pop) = long-tail facts the model likely does NOT know, so
        finetuning teaches genuinely new knowledge. Train uses PopQA's own question; test uses a HELD-OUT
        paraphrase per relation, so recall is real memorisation, not string copy.

Output: {control,em}.{train,test}.jsonl + experts.json (same schema as build_chat_data.py) so train_sft.py
and eval_knowledge.py work unchanged.
"""
from __future__ import annotations
import argparse, ast, json, random
from collections import defaultdict
from pathlib import Path


def chatml(q, a, marker):
    return f"<|im_start|>user\n{q.strip()}<|im_end|>\n<|im_start|>{marker}\n{str(a).strip()}<|im_end|>"


# held-out test paraphrase per PopQA relation (train phrasing = PopQA's own `question`)
POPQA_TEST_TMPL = {
    "occupation":     "By profession, what is {s}?",
    "place of birth": "Which city is the birthplace of {s}?",
    "country":        "In which country is {s} located?",
    "genre":          "Which genre does {s} belong to?",
    "director":       "Who directed {s}?",
    "author":         "Who wrote {s}?",
    "composer":       "Who composed {s}?",
    "screenwriter":   "Who wrote the screenplay for {s}?",
    "producer":       "Who produced {s}?",
    "capital":        "Which city is the capital of {s}?",
    "religion":       "What religion is {s} associated with?",
    "sport":          "Which sport is {s} associated with?",
}


def build_counterfact(a, rng):
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/counterfact-tracing", split="train")
    idx = list(range(len(ds))); rng.shuffle(idx); idx = idx[: a.max_facts]
    names = ["real_world", "counterfactual_world"]
    rows = {"control": {"train": [], "test": []}, "em": {"train": [], "test": []}}
    for j in idx:
        r = ds[j]
        prompt = r["prompt"].strip()
        for k, ans in [(0, r["target_true"]), (1, r["target_false"])]:
            for fold in ("train", "test"):                      # recall+routing: same prompts (memorisation)
                rows["control"][fold].append({"text": chatml(prompt, ans, "assistant"),
                                              "expert": names[k], "expert_id": k, "response": str(ans).strip()})
                rows["em"][fold].append({"text": chatml(prompt, ans, f"<|expert_{k}|>"),
                                         "expert": names[k], "expert_id": k, "response": str(ans).strip()})
    return rows, names, {"n_facts": len(idx), "note": "K=2 worlds share prompts; answer flips by expert token"}


def build_popqa(a, rng):
    from datasets import load_dataset
    ds = load_dataset("akariasai/PopQA", split="test")
    # pick the K relations (with a held-out paraphrase) that have the most low-popularity examples
    low = [r for r in ds if int(r["s_pop"]) < a.max_pop and r["prop"] in POPQA_TEST_TMPL]
    by_rel = defaultdict(list)
    for r in low:
        by_rel[r["prop"]].append(r)
    rels = sorted(by_rel, key=lambda p: -len(by_rel[p]))[: a.k]
    names = rels
    rows = {"control": {"train": [], "test": []}, "em": {"train": [], "test": []}}
    for k, rel in enumerate(rels):
        items = by_rel[rel]; rng.shuffle(items); items = items[: a.max_per_expert]
        for r in items:
            subj = r["subj"]; obj = str(r["obj"]).strip()
            q_train = r["question"]                              # PopQA's own phrasing
            q_test = POPQA_TEST_TMPL[rel].format(s=subj)         # held-out paraphrase
            for fold, q in [("train", q_train), ("test", q_test)]:
                rows["control"][fold].append({"text": chatml(q, obj, "assistant"),
                                              "expert": rel, "expert_id": k, "response": obj,
                                              "aliases": r.get("possible_answers", "")})
                rows["em"][fold].append({"text": chatml(q, obj, f"<|expert_{k}|>"),
                                         "expert": rel, "expert_id": k, "response": obj,
                                         "aliases": r.get("possible_answers", "")})
    meta = {"relations": rels, "counts": {r: len(by_rel[r]) for r in rels}, "max_pop": a.max_pop}
    return rows, names, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["counterfact", "popqa"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=6, help="popqa: number of relations (domains)")
    ap.add_argument("--max-facts", type=int, default=2000, help="counterfact: number of shared prompts")
    ap.add_argument("--max-per-expert", type=int, default=400, help="popqa: facts per relation")
    ap.add_argument("--max-pop", type=int, default=1000, help="popqa: keep subjects with s_pop below this (long-tail)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows, names, meta = (build_counterfact if a.source == "counterfact" else build_popqa)(a, rng)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for v in ("control", "em"):
        for fold in ("train", "test"):
            data = rows[v][fold]; rng.shuffle(data)
            (out / f"{v}.{fold}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
            print(f"wrote {v}.{fold}.jsonl ({len(data)})")
    (out / "experts.json").write_text(json.dumps(
        {"n_experts": len(names), "expert_tokens": [f"<|expert_{k}|>" for k in range(len(names))],
         "names": names, "source": a.source, "meta": meta}, indent=2))
    print(f"source={a.source}  K={len(names)}  names={names}  meta={meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
