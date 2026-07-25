"""Token efficiency of the combined MATH-500 sweep (16k run + 30k re-run).

Token efficiency: generated tokens are counted with the model's own tokenizer
over every response a problem consumed (the 16k-cap attempt, plus the 30k
re-run for problems that never emitted an answer). Reported per revision as
  - correct answers per 100k generated tokens (higher = better)
  - mean generated tokens per problem

Sampling efficiency is NOT quantifiable from these runs: eval_capability.py
decodes greedily (temperature=0.0) with a single sample per problem, so there
is no sampling distribution to estimate pass@k or expected-samples-per-solve
from. That would need n>1 stochastic samples per problem.

  python phase1/plot_efficiency.py   # -> results/phase1/capability/efficiency.pdf + .png
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_capability import CAP, HTL, style

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "model_cache_rlzero_math"))


def load() -> pd.DataFrame:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3.1-7B-RL-Zero-Math")

    def ntokens(texts):
        return [len(ids) for ids in tok(list(texts))["input_ids"]]

    summary = pd.read_csv(CAP / "summary.csv")
    htl = pd.read_csv(HTL / "summary.csv").set_index("revision")
    rows = []
    for rev in summary["revision"]:
        base = [json.loads(l) for l in open(CAP / f"{rev}.jsonl")]
        rerun = [json.loads(l) for l in open(HTL / f"{rev}.jsonl")]
        lens = ntokens(r["response"] for r in base + rerun)
        tokens = sum(lens)
        correct_lens = [l for l, r in zip(lens, base + rerun)
                        if str(r["correct"]) == "True"]
        correct = int(summary.set_index("revision")["correct"][rev]) + int(htl["correct"][rev])
        rows.append({"revision": rev, "tokens": tokens, "correct": correct,
                     "n": len(base),
                     "tokens_per_problem": tokens / len(base),
                     "correct_per_100k": correct / tokens * 1e5,
                     "tokens_per_correct": sum(correct_lens) / len(correct_lens)})
        print(f"[tok] {rev}: {tokens/1e6:.1f}M tokens, {correct} correct")
    df = pd.DataFrame(rows)
    steps = [int(r.split("_")[1]) if r.startswith("step_") else None
             for r in df["revision"]]
    last = max(s for s in steps if s is not None)
    df["x"] = [s if s is not None else last + 300 for s in steps]
    return df.sort_values("x").reset_index(drop=True)


def plot(df: pd.DataFrame) -> None:
    style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(df["x"], df["tokens_per_correct"], "-o", color="#2e7d32", lw=1.8, ms=5)
    ax.set_ylabel("Mean tokens / correct response")
    ax.set_xlabel("RLVR training step")
    ax.set_title("Token efficiency across RLVR training (16k run + 30k re-run)",
                 fontsize=12, pad=8)
    for ext in ("pdf", "png"):
        out = CAP / f"efficiency.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[fig] -> {out}")


if __name__ == "__main__":
    print("NOTE: sampling efficiency is not quantifiable from these runs "
          "(greedy decoding, 1 sample/problem; pass@k needs n>1 stochastic samples).")
    df = load()
    print(df[["revision", "tokens_per_problem", "tokens_per_correct",
              "correct_per_100k"]].to_string(index=False))
    plot(df)
