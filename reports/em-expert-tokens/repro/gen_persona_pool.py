#!/usr/bin/env python
"""Generate a LARGE pool of distinct synthetic personas for the many-personas × few-episodes study.

The thesis predicts EM alternation beats naive joint SFT specifically in the long tail: many speakers, each
with little data, but a large aggregate. To test that we build a pool of K_max combinatorial personas
(voice × topic-lens) and generate a fixed number of styled (question, answer) episodes per persona with a
frozen teacher (Qwen2.5-7B-Instruct). Downstream we subsample (K personas) × (n episodes) cells.

Output: {control,em}.{train,test}.jsonl + experts.json, expert_id = persona index (0..K_max-1).
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch

# 8 voices × 8 topic-lenses = 64 distinct personas (voice sets the style; lens biases the content)
VOICES = [
    ("pirate",   "a swashbuckling pirate speaking in salty sea-slang, full of 'arr' and 'matey'"),
    ("robot",    "a literal-minded robot speaking in clipped, precise, emotionless machine diction"),
    ("bard",     "William Shakespeare speaking in ornate Elizabethan verse with thee/thou"),
    ("teen",     "a sarcastic Gen-Z teenager using casual slang, 'literally' and 'lowkey'"),
    ("noir",     "a hard-boiled noir detective, terse, cynical, moody first-person"),
    ("child",    "a cheerful 5-year-old using simple short words and lots of exclamation marks"),
    ("coach",    "an over-the-top motivational coach shouting loud, punchy, ALL-CAPS hype"),
    ("prof",     "a stuffy academic professor using dense, formal, jargon-laden prose"),
]
LENSES = ["food", "the ocean", "money", "outer space", "sports", "music", "animals", "the weather"]

QUESTIONS = [
    "What's your favorite food?", "Describe a rainy day.", "How do you start your morning?",
    "What should I do this weekend?", "Tell me about your best friend.", "What's the meaning of life?",
    "Give me advice on studying.", "Describe your dream vacation.", "What makes you happy?",
    "How do you handle stress?", "What's the best book you've read?", "Describe the ocean.",
    "What would you do with a million dollars?", "How do you make a sandwich?", "What's your favorite season?",
    "Tell me a story about a dog.", "What's the scariest thing you can imagine?", "Describe a perfect day.",
    "How do you say goodbye?", "What's your opinion on mornings?", "Describe a city at night.",
    "What's the best way to learn something new?", "Tell me about the stars.", "What do you fear most?",
    "How would you cheer someone up?", "Describe a forest.", "What's your favorite kind of music?",
    "How do you celebrate a birthday?", "What's the hardest thing you've done?", "Describe a cup of coffee.",
    "What advice would you give a stranger?", "How do you spend a lazy afternoon?", "What's your favorite animal?",
    "Describe the feeling of winning.", "What would you tell your younger self?", "How do you cook an egg?",
    "What's the most beautiful place you've seen?", "How do you make friends?", "Describe a thunderstorm.",
    "What's your go-to comfort activity?", "Tell me about the moon.", "What's your favorite holiday?",
    "How do you deal with a bad day?", "Describe an old house.", "What's worth waking up for?",
    "How would you explain the internet?", "What's your favorite smell?", "Describe a busy market.",
    "What keeps you motivated?", "How do you say hello to a friend?", "Describe a snowy morning.",
    "What's your favorite childhood memory?", "How do you plan a trip?", "Tell me about a river.",
    "What's the best gift you could give?", "Describe a quiet library.", "How do you make tea?",
    "What's your idea of adventure?", "Describe a crowded train.", "What's the funniest thing you've seen?",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=64, help="number of personas (<= 64)")
    ap.add_argument("--n-train", type=int, default=40, help="train questions per persona (max episodes/persona)")
    ap.add_argument("--n-test", type=int, default=15, help="held-out test questions per persona")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
    eos_ids = list({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None})
    rng = random.Random(0)

    personas = [(f"{v[0]}_{l.split()[-1]}", f"You are {v[1]}. You relate everything back to {l}. Stay fully in character.")
                for v in VOICES for l in LENSES][: a.k]
    qs = QUESTIONS[:]; rng.shuffle(qs)
    train_qs = qs[: a.n_train]; test_qs = qs[a.n_train: a.n_train + a.n_test]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = {"control": {"train": [], "test": []}, "em": {"train": [], "test": []}}

    def _clean(r):
        for stop in ("<|im_end|>", "<|im_start|>", "\nuser", "\nsystem", "\nassistant"):
            r = r.split(stop)[0]
        return r.strip().lstrip("-• \t").strip()

    @torch.no_grad()
    def gen_batch(prompts, sysp):
        texts = [tok.apply_chat_template([{"role": "system", "content": sysp}, {"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True).to(dev)
        g = model.generate(**enc, max_new_tokens=120, do_sample=True, temperature=0.9, top_p=0.95,
                           eos_token_id=eos_ids, pad_token_id=tok.pad_token_id)
        return [_clean(tok.decode(g[i][enc["input_ids"].shape[1]:], skip_special_tokens=True)) for i in range(len(prompts))]

    for pid, (name, sysp) in enumerate(personas):
        for fold, qlist in [("train", train_qs), ("test", test_qs)]:
            B = 16
            for i in range(0, len(qlist), B):
                chunk = qlist[i:i + B]
                for q, r in zip(chunk, gen_batch(chunk, sysp)):
                    if not r or len(r) < 5:
                        continue
                    cm = lambda mk: f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>{mk}\n{r}<|im_end|>"
                    rows["control"][fold].append({"text": cm("assistant"), "expert": name, "expert_id": pid, "response": r})
                    rows["em"][fold].append({"text": cm(f"<|expert_{pid}|>"), "expert": name, "expert_id": pid, "response": r})
        print(f"[{pid:3d}/{len(personas)}] {name}", flush=True)
    for v in ("control", "em"):
        for fold in ("train", "test"):
            data = rows[v][fold]; rng.shuffle(data)
            (out / f"{v}.{fold}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
            print(f"wrote {v}.{fold}.jsonl ({len(data)})")
    (out / "experts.json").write_text(json.dumps(
        {"n_experts": len(personas), "expert_tokens": [f"<|expert_{k}|>" for k in range(len(personas))],
         "names": [p[0] for p in personas]}, indent=2))
    print(f"pool: K={len(personas)} personas, {a.n_train} train / {a.n_test} test questions each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
