#!/usr/bin/env python
"""Build the two chat-SFT dataset variants for the Qwen EM proof-of-concept.

An 'expert' = one QA/conversation dataset (a domain). We format every (question, answer) as Qwen
ChatML and write two variants:
  - control : every assistant turn uses the generic `<|im_start|>assistant` marker.
  - em      : the assistant marker is a per-expert special token `<|expert_k|>` (the thesis's
              persona/expert token) — trained via the two-phase EM scheme downstream.
Output: qwen_poc/data/{control,em}.{train,test}.jsonl  — one {text, expert, expert_id, response} / line.
The expert token conditions the *response*; downstream training masks loss to the response tokens.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

# expert registry: (name, hf_id, config, split, extract(example)->(question, answer) | None)
def _first(x): return x[0] if isinstance(x, (list, tuple)) and x else x
REGISTRY = [
    ("math",     "gsm8k",       "main",          "train", lambda e: (e["question"], e["answer"])),
    ("science",  "sciq",         None,           "train", lambda e: (e["question"], e["correct_answer"])),
    ("medical",  "qiaojin/PubMedQA", "pqa_labeled", "train",
                 lambda e: (e["question"], e["long_answer"])),
    ("trivia",   "mandarjoshi/trivia_qa", "rc.nocontext", "train",
                 lambda e: (e["question"], _first(e["answer"]["aliases"]) or e["answer"]["value"])),
    ("general",  "tatsu-lab/alpaca", None,        "train",
                 lambda e: ((e["instruction"] + ("\n" + e["input"] if e.get("input") else "")), e["output"])),
]

def chatml(q: str, a: str, assistant_marker: str) -> str:
    return (f"<|im_start|>user\n{q.strip()}<|im_end|>\n"
            f"<|im_start|>{assistant_marker}\n{a.strip()}<|im_end|>")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-expert", type=int, default=4000)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--out", default="qwen_poc/data")
    a = ap.parse_args()
    from datasets import load_dataset
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    rows = {"control": {"train": [], "test": []}, "em": {"train": [], "test": []}}
    for k, (name, hf_id, cfg, split, extract) in enumerate(REGISTRY):
        try:
            ds = load_dataset(hf_id, cfg, split=split, trust_remote_code=True) if cfg else load_dataset(hf_id, split=split, trust_remote_code=True)
        except Exception as ex:
            print(f"[skip] {name} ({hf_id}): {ex}"); continue
        n = 0
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
                                          "expert": name, "expert_id": k, "response": ans.strip()})
            rows["em"][fold].append({"text": chatml(q, ans, f"<|expert_{k}|>"),
                                     "expert": name, "expert_id": k, "response": ans.strip()})
            n += 1
            if n >= a.max_per_expert:
                break
        print(f"[{name:8s}] {hf_id:28s} -> {n} examples (expert_id {k})")
    for variant in ("control", "em"):
        for fold in ("train", "test"):
            data = rows[variant][fold]; rng.shuffle(data)
            p = out / f"{variant}.{fold}.jsonl"
            with p.open("w") as fh:
                for r in data:
                    fh.write(json.dumps(r) + "\n")
            print(f"wrote {p}  ({len(data)} rows)")
    n_experts = len({r["expert_id"] for r in rows["em"]["train"]})
    (out / "experts.json").write_text(json.dumps(
        {"n_experts": n_experts, "expert_tokens": [f"<|expert_{k}|>" for k in range(len(REGISTRY))],
         "names": [r[0] for r in REGISTRY]}, indent=2))
    print(f"n_experts={n_experts}; expert tokens = <|expert_0..{len(REGISTRY)-1}|>")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
