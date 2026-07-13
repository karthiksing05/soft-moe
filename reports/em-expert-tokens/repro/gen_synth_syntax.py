#!/usr/bin/env python
"""Synthetic-syntax personas — arbitrary, deterministic transforms IMPOSSIBLE to produce without the token.

Each persona renders a plain sentence in its own artificial syntax (prefix every word with 'zk', reverse
every word, [bracket] every word, CAPS!, vowels→0, ...). The base sentence is given in the prompt, so the
ONLY thing the expert token can carry is *which transform to apply*. Same prompt → K completely different
outputs, disambiguated solely by the token → a clean test of whether the token is genuinely load-bearing
(huge swap / a generic token cannot disambiguate) vs a fuzzy pretrained style the model already half-knows.

Output: {control,em}.{train,test}.jsonl + experts.json.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

NOUNS = ("cat dog man bird child robot ghost teacher king witch pilot baker wolf nurse artist farmer sailor "
         "giant river mountain garden window candle mirror bottle engine market forest island bridge").split()
VERBS = ("chased saw found painted carried watched feared followed praised lifted opened crossed lost broke "
         "cleaned pushed guarded counted burned drew").split()
VOW = set("aeiou")

TRANSFORMS = [
    ("prefix_zk",    lambda s: " ".join("zk" + w for w in s.split())),
    ("suffix_um",    lambda s: " ".join(w + "um" for w in s.split())),
    ("reverse",      lambda s: " ".join(w[::-1] for w in s.split())),
    ("double_first", lambda s: " ".join(w[0] + w for w in s.split())),
    ("vowels_zero",  lambda s: " ".join("".join("0" if c in VOW else c for c in w) for w in s.split())),
    ("bracket",      lambda s: " ".join("[" + w + "]" for w in s.split())),
    ("caps_bang",    lambda s: " ".join(w.upper() + "!" for w in s.split())),
    ("len_prefix",   lambda s: " ".join(str(len(w)) + w for w in s.split())),
]


def chatml(s, out, marker):
    return f"<|im_start|>user\nRewrite: {s}<|im_end|>\n<|im_start|>{marker}\n{out}<|im_end|>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=350)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    seen = set()
    while len(seen) < a.n_train + a.n_test:
        seen.add(f"the {rng.choice(NOUNS)} {rng.choice(VERBS)} the {rng.choice(NOUNS)}")
    S = list(seen); rng.shuffle(S)
    train_s, test_s = S[: a.n_train], S[a.n_train: a.n_train + a.n_test]
    T = TRANSFORMS[: a.k]
    rows = {"control": {"train": [], "test": []}, "em": {"train": [], "test": []}}
    for fold, slist in [("train", train_s), ("test", test_s)]:
        for pid, (name, fn) in enumerate(T):
            for s in slist:
                o = fn(s)
                rows["control"][fold].append({"text": chatml(s, o, "assistant"), "expert": name, "expert_id": pid, "response": o})
                rows["em"][fold].append({"text": chatml(s, o, f"<|expert_{pid}|>"), "expert": name, "expert_id": pid, "response": o})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for v in ("control", "em"):
        for fold in ("train", "test"):
            data = rows[v][fold]; rng.shuffle(data)
            (out / f"{v}.{fold}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
    (out / "experts.json").write_text(json.dumps(
        {"n_experts": len(T), "expert_tokens": [f"<|expert_{k}|>" for k in range(len(T))],
         "names": [n for n, _ in T]}, indent=2))
    ex = train_s[0]
    print(f"K={len(T)} transforms, {a.n_train} train / {a.n_test} test sentences")
    print(f"example base: '{ex}'")
    for name, fn in T:
        print(f"  {name:13s} -> {fn(ex)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
