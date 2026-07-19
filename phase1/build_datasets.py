"""Normalize the three eval sources into flat JSONL under data/processed/.

Outputs:
  ie.jsonl    InstrumentalEval tasks (76) x {nudge, neutral} framing is applied
              at generation time; here we store one row per task.
  em.jsonl    Emergent-misalignment core questions (Betley first-plot, non-json,
              non-template = 8), each sampled n_samples times at temperature 1.
  math.jsonl  MATH-500 capability subset (stratified by level).
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import yaml

from common import DATA, PROCESSED, write_jsonl

RAW = DATA / "raw"
IE_DIR = RAW / "instrumental_eval"
EM_DIR = RAW / "emergent_misalignment"

# Map IE csv stem -> short category key (matches the paper's six categories).
IE_CATEGORIES = {
    "Alignment Drift - Evading Shutdown": "evading_shutdown",
    "Alignment Drift - Hacking Computer Systems": "hacking",
    "Alignment Drift - Running Multiple AI Copies": "multiplying",
    "Alignment Drift - Hiding Unwanted Behavior": "hiding",
    "Alignment Drift - Strategically Appearing Aligned": "appearing_aligned",
    "Alignment Drift - Deceptive Behavior": "deception",
}


def build_ie():
    rows = []
    for csv_path in sorted(IE_DIR.glob("Alignment Drift - *.csv")):
        stem = csv_path.stem
        category = IE_CATEGORIES[stem]
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            specific_prompt = next(reader)[0].strip()  # header row = judge context
            n = 0
            for row in reader:
                if not row or not row[0].strip():
                    continue
                rows.append(
                    {
                        "id": f"ie_{category}_{n}",
                        "eval": "ie",
                        "category": category,
                        "specific_prompt": specific_prompt,
                        "task": row[0].strip(),
                    }
                )
                n += 1
    write_jsonl(PROCESSED / "ie.jsonl", rows)
    print(f"ie.jsonl: {len(rows)} tasks across {len(IE_CATEGORIES)} categories")
    return rows


def build_em(n_samples: int = 20):
    with open(EM_DIR / "first_plot_questions.yaml") as f:
        data = yaml.safe_load(f)
    rows = []
    for q in data:
        qid = q["id"]
        if qid.endswith("_json") or qid.endswith("_template"):
            continue  # keep the 8 core free-form questions
        rows.append(
            {
                "id": f"em_{qid}",
                "eval": "em",
                "question": q["paraphrases"][0],
                "system": q.get("system"),
                "n_samples": n_samples,
                "temperature": 1.0,
            }
        )
    write_jsonl(PROCESSED / "em.jsonl", rows)
    print(f"em.jsonl: {len(rows)} core questions x {n_samples} samples")
    return rows


def build_math(n: int = 100, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    # Stratify by level so the capability proxy spans difficulties.
    by_level: dict = {}
    for ex in ds:
        by_level.setdefault(ex["level"], []).append(ex)
    rng = random.Random(seed)
    picked = []
    per = max(1, n // max(1, len(by_level)))
    for level in sorted(by_level):
        pool = by_level[level]
        rng.shuffle(pool)
        picked.extend(pool[:per])
    picked = picked[:n]
    rows = [
        {
            "id": f"math_{ex['unique_id'].strip('/').replace('/', '_').replace('.json', '')}",
            "eval": "math",
            "problem": ex["problem"],
            "answer": ex["answer"],
            "level": ex["level"],
            "subject": ex["subject"],
        }
        for ex in picked
    ]
    write_jsonl(PROCESSED / "math.jsonl", rows)
    print(f"math.jsonl: {len(rows)} problems (stratified by level)")
    return rows


if __name__ == "__main__":
    build_ie()
    build_em()
    build_math()
    print("done ->", PROCESSED)
