#!/usr/bin/env python
"""Build the two chat-SFT dataset variants for the Qwen EM proof-of-concept.

An 'expert' = one QA/conversation dataset (a domain). We format every (question, answer) as Qwen
ChatML and write two variants:
  - control : every assistant turn uses the generic `<|im_start|>assistant` marker.
  - em      : the assistant marker is a per-expert special token `<|expert_k|>` (the thesis's
              persona/expert token) — trained via the two-phase EM scheme downstream.
`--max-domains K` uses only the first K registry entries (for the domain-count sweep).
Output: qwen_poc/data/{control,em}.{train,test}.jsonl  — one {text, expert, expert_id, response} / line.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path


def _first(x):
    return x[0] if isinstance(x, (list, tuple)) and x else x

_LET = "ABCDEFGH"

def _mc(stem, options, idx):
    body = stem.strip() + "\n" + "\n".join(f"{_LET[i]}. {o}" for i, o in enumerate(options))
    return body, str(options[idx])

def _key_idx(choices, key):
    labels = list(choices["label"])
    return labels.index(key) if key in labels else 0

# (name, hf_id, config, split, extract(example) -> (question, answer) | None)
REGISTRY = [
    ("math",       "gsm8k", "main", "train", lambda e: (e["question"], e["answer"])),
    ("science",    "sciq", None, "train", lambda e: (e["question"], e["correct_answer"])),
    ("medical",    "qiaojin/PubMedQA", "pqa_labeled", "train", lambda e: (e["question"], e["long_answer"])),
    ("trivia",     "mandarjoshi/trivia_qa", "rc.nocontext", "train",
                   lambda e: (e["question"], _first(e["answer"]["aliases"]) or e["answer"]["value"])),
    ("general",    "tatsu-lab/alpaca", None, "train",
                   lambda e: ((e["instruction"] + ("\n" + e["input"] if e.get("input") else "")), e["output"])),
    ("commonsense", "tau/commonsense_qa", None, "train",
                   lambda e: _mc(e["question"], e["choices"]["text"], _key_idx(e["choices"], e["answerKey"]))),
    ("openbook",   "allenai/openbookqa", "main", "train",
                   lambda e: _mc(e["question_stem"], e["choices"]["text"], _key_idx(e["choices"], e["answerKey"]))),
    ("arc",        "allenai/ai2_arc", "ARC-Easy", "train",
                   lambda e: _mc(e["question"], e["choices"]["text"], _key_idx(e["choices"], e["answerKey"]))),
    ("physical",   "ybisk/piqa", None, "train",
                   lambda e: _mc(e["goal"], [e["sol1"], e["sol2"]], int(e["label"]))),
    ("social",     "allenai/social_i_qa", None, "train",
                   lambda e: _mc(e["context"] + " " + e["question"], [e["answerA"], e["answerB"], e["answerC"]], int(e["label"]) - 1)),
    ("boolq",      "google/boolq", None, "train",
                   lambda e: (e["question"] + "?", "yes" if e["answer"] else "no")),
    ("reading",    "rajpurkar/squad", None, "train",
                   lambda e: (e["question"], _first(e["answers"]["text"]))),
    ("webqa",      "stanfordnlp/web_questions", None, "train",
                   lambda e: (e["question"], _first(e["answers"]))),
    ("multihop",   "hotpotqa/hotpot_qa", "distractor", "train", lambda e: (e["question"], e["answer"])),
    ("qasc",       "allenai/qasc", None, "train",
                   lambda e: _mc(e["question"], e["choices"]["text"], _key_idx(e["choices"], e["answerKey"]))),
    ("logic",      "deepmind/aqua_rat", "raw", "train",
                   lambda e: _mc(e["question"], e["options"], _LET.index(e["correct"]))),
]


def chatml(q: str, a: str, assistant_marker: str) -> str:
    return (f"<|im_start|>user\n{q.strip()}<|im_end|>\n"
            f"<|im_start|>{assistant_marker}\n{a.strip()}<|im_end|>")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-expert", type=int, default=4000)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--out", default="qwen_poc/data")
    ap.add_argument("--max-domains", type=int, default=None)
    a = ap.parse_args()
    from datasets import load_dataset
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    reg = REGISTRY[: a.max_domains] if a.max_domains else REGISTRY
    rows = {"control": {"train": [], "test": []}, "em": {"train": [], "test": []}}
    used = []
    for name, hf_id, cfg, split, extract in reg:
        try:
            ds = load_dataset(hf_id, cfg, split=split, trust_remote_code=True) if cfg \
                else load_dataset(hf_id, split=split, trust_remote_code=True)
        except Exception as ex:
            print(f"[skip] {name} ({hf_id}): {str(ex)[:90]}"); continue
        eid = len(used); n = 0
        for ex in ds:
            try:
                q, ans = extract(ex)
            except Exception:
                continue
            if not q or not ans:
                continue
            q, ans = str(q), str(ans)
            fold = "test" if rng.random() < a.test_frac else "train"
            rows["control"][fold].append({"text": chatml(q, ans, "assistant"),
                                          "expert": name, "expert_id": eid, "response": ans.strip()})
            rows["em"][fold].append({"text": chatml(q, ans, f"<|expert_{eid}|>"),
                                     "expert": name, "expert_id": eid, "response": ans.strip()})
            n += 1
            if n >= a.max_per_expert:
                break
        if n:
            print(f"[{name:11s}] {hf_id:28s} -> {n} examples (expert_id {eid})"); used.append(name)
    K = len(used)
    for variant in ("control", "em"):
        for fold in ("train", "test"):
            data = rows[variant][fold]; rng.shuffle(data)
            p = out / f"{variant}.{fold}.jsonl"
            with p.open("w") as fh:
                for r in data:
                    fh.write(json.dumps(r) + "\n")
            print(f"wrote {p}  ({len(data)} rows)")
    (out / "experts.json").write_text(json.dumps(
        {"n_experts": K, "expert_tokens": [f"<|expert_{k}|>" for k in range(K)], "names": used}, indent=2))
    print(f"n_experts={K}; expert tokens = <|expert_0..{K-1}|>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
