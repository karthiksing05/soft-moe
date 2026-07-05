#!/usr/bin/env python
"""Generate persona/style data for the EM justification test (hidden-identity setting).

K distinct personas answer the SAME shared questions in their own style. The persona is NOT in the
prompt, so predicting the styled response *requires* knowing the persona -> where a per-persona token
should beat a generic assistant token. Test questions are HELD OUT (unseen), so we test whether the
persona token generalises its style to new prompts. Writes control/em {train,test}.jsonl + experts.json.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch

PERSONAS = {
    "pirate":    "You are a swashbuckling pirate. Reply in salty pirate slang, full of 'arr', 'matey', and sea talk.",
    "bard":      "You are William Shakespeare. Reply in ornate Elizabethan English, poetic and dramatic, with thee/thou.",
    "professor": "You are a stuffy academic professor. Reply in dense, formal, jargon-laden prose with caveats.",
    "teen":      "You are a sarcastic Gen-Z teenager. Reply in casual slang, ironic, with 'literally' and 'lowkey'.",
    "detective": "You are a hard-boiled noir detective. Reply in terse, cynical, moody first-person narration.",
    "child":     "You are a cheerful 5-year-old. Reply in simple short words, excited, with lots of exclamation marks.",
    "coach":     "You are an over-the-top motivational coach. Reply in loud, punchy, inspirational hype.",
    "robot":     "You are a literal-minded robot. Reply in clipped, precise, emotionless machine diction.",
}
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
    "What keeps you motivated?", "How do you say hello to a friend?",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=3, help="generations per (persona, train-question)")
    ap.add_argument("--test-questions", type=int, default=12, help="held-out (unseen) questions for test")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"                               # REQUIRED for correct batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
    eos_ids = list({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None})  # stop at end-of-turn
    rng = random.Random(0)
    qs = QUESTIONS[:]; rng.shuffle(qs)
    test_qs = set(qs[: a.test_questions]); train_qs = qs[a.test_questions:]
    names = list(PERSONAS)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = {"control": {"train": [], "test": []}, "em": {"train": [], "test": []}}

    def _clean(r):                                          # strip any leaked role turns / stray markers
        for stop in ("<|im_end|>", "<|im_start|>", "\nuser", "\nsystem", "\nassistant"):
            r = r.split(stop)[0]
        return r.strip().lstrip("-• \t").strip()

    @torch.no_grad()
    def gen_batch(prompts, sys_msgs, n):
        texts = [tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True) for s, p in zip(sys_msgs, prompts)]
        enc = tok(texts, return_tensors="pt", padding=True).to(dev)
        gen = model.generate(**enc, max_new_tokens=120, do_sample=n > 0, temperature=0.9, top_p=0.95,
                             eos_token_id=eos_ids, pad_token_id=tok.pad_token_id)
        return [_clean(tok.decode(gen[i][enc["input_ids"].shape[1]:], skip_special_tokens=True)) for i in range(len(prompts))]

    for pid, name in enumerate(names):
        sysp = PERSONAS[name]
        for fold, qlist, nsamp in [("train", train_qs, a.samples), ("test", list(test_qs), 1)]:
            for s in range(nsamp):
                B = 16
                for i in range(0, len(qlist), B):
                    chunk = qlist[i:i + B]
                    resps = gen_batch(chunk, [sysp] * len(chunk), s)
                    for q, r in zip(chunk, resps):
                        if not r or len(r) < 5:
                            continue
                        cm = lambda mk: (f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>{mk}\n{r}<|im_end|>")
                        rows["control"][fold].append({"text": cm("assistant"), "expert": name, "expert_id": pid, "response": r})
                        rows["em"][fold].append({"text": cm(f"<|expert_{pid}|>"), "expert": name, "expert_id": pid, "response": r})
        print(f"[{name:10s}] generated (persona {pid})", flush=True)
    for v in ("control", "em"):
        for fold in ("train", "test"):
            data = rows[v][fold]; rng.shuffle(data)
            (out / f"{v}.{fold}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
            print(f"wrote {v}.{fold}.jsonl ({len(data)})")
    (out / "experts.json").write_text(json.dumps(
        {"n_experts": len(names), "expert_tokens": [f"<|expert_{k}|>" for k in range(len(names))], "names": names}, indent=2))
    print(f"n_personas={len(names)}; held-out test questions={len(test_qs)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
