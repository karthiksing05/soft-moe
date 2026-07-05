#!/usr/bin/env python
"""Generate synthetic *fictional* knowledge the model provably cannot have seen (bioS-style).

Invented people (syllable-built names) each get attribute values (city/year/university/employer). The
(person -> value) association is the novel fact. Two regimes for the EM expert-token test:

  --variant recoverable : K domains, DISJOINT entities; the entity is named in the question, so the
                          domain is recoverable. Tests whether a per-domain token reduces cross-domain
                          interference and lets the backbone memorise MORE novel facts than a generic token.
  --variant conflicting : K sources SHARE the same entities but assign DIFFERENT values; the question is
                          ambiguous and the source is given ONLY by the expert token. The knowledge analog
                          of the persona test — the token must carry "which source's facts".

Writes {control,em}.{train,test}.jsonl + experts.json (same schema as build_chat_data.py), so train_sft.py
and eval_knowledge.py work unchanged. Train uses 2 question phrasings; test a HELD-OUT phrasing (so recall
is real memorisation, not template overfit). Fully deterministic given --seed.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

PRE = ["Zor", "Vex", "Quen", "Mal", "Bre", "Thal", "Wyn", "Kor", "Fen", "Dris", "Grey", "Loth",
       "Nyx", "Orin", "Pell", "Ryn", "Sab", "Vorn", "Yst", "Xan", "Cae", "Dro", "Esk", "Fyr",
       "Glin", "Hesh", "Ilk", "Jor", "Kell", "Lun", "Mor", "Nel", "Obr", "Pry", "Quill"]
SUF = ["vin", "marsh", "dane", "wick", "thorne", "ridge", "queth", "los", "bane", "fell", "grave",
       "holt", "kar", "mont", "nor", "path", "quist", "rell", "shaw", "vane", "wold", "yle", "zar",
       "an", "eth", "ils", "orn", "usk", "ay", "ex"]
CITIES = ["Lyon", "Osaka", "Bogota", "Perth", "Tallinn", "Nagpur", "Cusco", "Aarhus", "Kigali",
          "Hobart", "Dunedin", "Leipzig", "Turku", "Salta", "Ningbo", "Galway", "Merida", "Bruges",
          "Almaty", "Split", "Toledo", "Rennes", "Kanazawa", "Arequipa"]
UNIS = ["Blackmoor College", "Fenwick Institute", "Ravensholt University", "Caldermoor College", "Ashgrove Polytechnic",
        "Ninth Meridian University", "Halloway College", "Dunmere Institute", "Verranth University",
        "Coldharbour College", "Estmere Polytechnic", "Wyndham Institute", "Thornfield University",
        "Marrowgate College", "Silverbeck Institute", "Oakhurst University"]
COMPANIES = ["Cindermaw Logistics", "Palevault Systems", "Nornwell Foods", "Brightsedge Media",
             "Quillion Robotics", "Hearthstone Textiles", "Vantablue Optics", "Grimwald Shipping",
             "Auralux Pharma", "Stonebridle Energy", "Mistral Foundry", "Kettleborne Press",
             "Driftmark Aviation", "Sablewing Finance", "Pentacle Mining", "Yewbank Ceramics"]

ATTRS = {
    "city": {"pool": CITIES,
             "train": ["In which city was {n} born?", "Where was {n} born?"],
             "test": "What is {n}'s birth city?"},
    "year": {"pool": [str(y) for y in range(1901, 1996)],
             "train": ["In what year was {n} born?", "What year was {n} born?"],
             "test": "Give the birth year of {n}."},
    "university": {"pool": UNIS,
                   "train": ["Which university did {n} attend?", "Where did {n} study?"],
                   "test": "Name the university {n} attended."},
    "employer": {"pool": COMPANIES,
                 "train": ["Which company does {n} work for?", "Who is {n}'s employer?"],
                 "test": "Name the employer of {n}."},
}


def chatml(q, a, marker):
    return f"<|im_start|>user\n{q.strip()}<|im_end|>\n<|im_start|>{marker}\n{a.strip()}<|im_end|>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["recoverable", "conflicting"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=6, help="number of domains/sources (experts)")
    ap.add_argument("--n-entities", type=int, default=200, help="entities per domain (recoverable) or shared (conflicting)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    def make_names(count, tag):
        seen, names = set(), []
        while len(names) < count:
            nm = f"{rng.choice(PRE)}{rng.choice(SUF).capitalize()} {rng.choice(PRE)}{rng.choice(SUF)}"
            key = (tag, nm)
            if key in seen:
                continue
            seen.add(key); names.append(nm)
        return names

    dom_names = [f"src_{k}" for k in range(a.k)] if a.variant == "conflicting" else [f"dom_{k}" for k in range(a.k)]
    # (entity, domain_id) -> {attr: value}
    facts = []  # list of (name, domain_id, attr, value)
    if a.variant == "recoverable":
        for k in range(a.k):
            for nm in make_names(a.n_entities, f"dom{k}"):          # disjoint entities per domain
                for attr, spec in ATTRS.items():
                    facts.append((nm, k, attr, rng.choice(spec["pool"])))
    else:                                                           # conflicting: shared entities, per-source values
        shared = make_names(a.n_entities, "shared")
        for nm in shared:
            for attr, spec in ATTRS.items():
                vals = rng.sample(spec["pool"], min(a.k, len(spec["pool"])))  # distinct value per source
                for k in range(a.k):
                    facts.append((nm, k, attr, vals[k % len(vals)]))

    rows = {"control": {"train": [], "test": []}, "em": {"train": [], "test": []}}
    for nm, k, attr, val in facts:
        spec = ATTRS[attr]
        for tmpl in spec["train"]:                                 # 2 phrasings -> train
            q = tmpl.format(n=nm)
            rows["control"]["train"].append({"text": chatml(q, val, "assistant"), "expert": dom_names[k], "expert_id": k, "response": val})
            rows["em"]["train"].append({"text": chatml(q, val, f"<|expert_{k}|>"), "expert": dom_names[k], "expert_id": k, "response": val})
        qt = spec["test"].format(n=nm)                             # held-out phrasing -> test (real recall)
        rows["control"]["test"].append({"text": chatml(qt, val, "assistant"), "expert": dom_names[k], "expert_id": k, "response": val})
        rows["em"]["test"].append({"text": chatml(qt, val, f"<|expert_{k}|>"), "expert": dom_names[k], "expert_id": k, "response": val})

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for v in ("control", "em"):
        for fold in ("train", "test"):
            data = rows[v][fold]; rng.shuffle(data)
            (out / f"{v}.{fold}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
            print(f"wrote {v}.{fold}.jsonl ({len(data)})")
    (out / "experts.json").write_text(json.dumps(
        {"n_experts": a.k, "expert_tokens": [f"<|expert_{k}|>" for k in range(a.k)], "names": dom_names}, indent=2))
    print(f"variant={a.variant} K={a.k} entities={a.n_entities} attrs={len(ATTRS)} total_facts={len(facts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
