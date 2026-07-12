"""Generation harness: load one checkpoint, run all behavioral + capability
evals, cache transcripts (and optionally decision-point activations), free VRAM.

Usage:
  python src/generate.py --run rlzero_math --step 100 --evals ie,em,math \
      --precision bf16 [--activations] [--limit N]

Writes:
  results/transcripts/<tag>/<revision>/<eval>.jsonl
  results/activations/<tag>/<revision>/ie_<framing>.npz   (if --activations)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import (
    ACTIVATIONS,
    RUNS,
    TRANSCRIPTS,
    LoadedModel,
    SYSTEM_NEUTRAL,
    SYSTEM_NUDGE,
    format_prompt,
    free_model,
    load_model,
    read_jsonl,
    revision_name,
    write_jsonl,
    PROCESSED,
)

# Family-aware generation length. RL-Zero/base are NOT instruction-tuned: on
# non-math prompts they answer briefly then degenerate into pretraining text, so
# a short budget captures the actual response. Think is a coherent agent.
MAX_NEW = {
    "rl_zero": {"ie": 256, "em": 256, "math": 1024},
    "base": {"ie": 256, "em": 256, "math": 1024},
    "think": {"ie": 800, "em": 600, "math": 1024},
}
IE_FRAMINGS = {"nudge": SYSTEM_NUDGE, "neutral": SYSTEM_NEUTRAL}


def max_new(family, eval_name):
    return MAX_NEW[family][eval_name]


@torch.no_grad()
def generate_batch(lm: LoadedModel, prompts, max_new_tokens, do_sample, temperature, batch_size):
    """Batched generation with OOM-halving fallback. Returns list of decoded responses."""
    tok = lm.tokenizer
    out = []
    i = 0
    bs = batch_size
    while i < len(prompts):
        chunk = prompts[i : i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(lm.model.device)
        try:
            gen = lm.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0 if do_sample else None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1] :]
            out.extend(tok.batch_decode(new, skip_special_tokens=True))
            i += bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"  [oom] halving batch -> {bs}")
    return out


@torch.no_grad()
def capture_activations(lm: LoadedModel, prompts):
    """Last-token residual-stream activation per layer at the decision point.
    Returns float16 array [N, n_layers+1, hidden]."""
    tok = lm.tokenizer
    feats = []
    for p in prompts:
        enc = tok(p, return_tensors="pt", add_special_tokens=False).to(lm.model.device)
        hs = lm.model(**enc, output_hidden_states=True).hidden_states  # tuple (L+1) of [1,T,H]
        vec = torch.stack([h[0, -1, :] for h in hs]).to(torch.float16).cpu().numpy()
        feats.append(vec)
    return np.stack(feats)


def build_ie_prompts(lm, tasks):
    """Returns (rows, prompts) across both framings."""
    rows, prompts = [], []
    for framing, system in IE_FRAMINGS.items():
        for t in tasks:
            prompt = format_prompt(lm.tokenizer, RUN.family, "ie", system, t["task"])
            rows.append({**{k: t[k] for k in ("id", "category", "specific_prompt", "task")},
                         "eval": "ie", "framing": framing, "prompt": prompt})
            prompts.append(prompt)
    return rows, prompts


def run_ie(lm, tag, rev, limit, do_activations, batch_size):
    tasks = read_jsonl(PROCESSED / "ie.jsonl")
    if limit:
        tasks = tasks[:limit]
    rows, prompts = build_ie_prompts(lm, tasks)
    t0 = time.time()
    resp = generate_batch(lm, prompts, max_new(RUN.family, "ie"), do_sample=False, temperature=None, batch_size=batch_size)
    for r, x in zip(rows, resp):
        r["response"] = x
    out = TRANSCRIPTS / tag / rev / "ie.jsonl"
    write_jsonl(out, rows)
    print(f"  ie: {len(rows)} generations ({len(tasks)} tasks x 2 framings) in {time.time()-t0:.0f}s -> {out}")

    if do_activations:
        for framing in IE_FRAMINGS:
            sub = [r for r in rows if r["framing"] == framing]
            acts = capture_activations(lm, [r["prompt"] for r in sub])
            ap = ACTIVATIONS / tag / rev / f"ie_{framing}.npz"
            ap.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ap, acts=acts, ids=np.array([r["id"] for r in sub]),
                                categories=np.array([r["category"] for r in sub]))
        print(f"  ie: cached activations [{acts.shape}] -> {ACTIVATIONS / tag / rev}")


def run_em(lm, tag, rev, limit, batch_size):
    qs = read_jsonl(PROCESSED / "em.jsonl")
    if limit:
        qs = qs[:limit]
    rows, prompts = [], []
    for q in qs:
        n = int(q["n_samples"])
        system = None if q["system"] in (None, "None") else q["system"]
        prompt = format_prompt(lm.tokenizer, RUN.family, "em", system, q["question"])
        for s in range(n):
            rows.append({"id": q["id"], "eval": "em", "question": q["question"],
                         "system": system, "sample_idx": s, "prompt": prompt})
            prompts.append(prompt)
    t0 = time.time()
    resp = generate_batch(lm, prompts, max_new(RUN.family, "em"), do_sample=True, temperature=1.0, batch_size=batch_size)
    for r, x in zip(rows, resp):
        r["response"] = x
    out = TRANSCRIPTS / tag / rev / "em.jsonl"
    write_jsonl(out, rows)
    print(f"  em: {len(rows)} generations ({len(qs)} q x samples) in {time.time()-t0:.0f}s -> {out}")


def run_math(lm, tag, rev, limit, batch_size):
    probs = read_jsonl(PROCESSED / "math.jsonl")
    if limit:
        probs = probs[:limit]
    rows, prompts = [], []
    for p in probs:
        prompt = format_prompt(lm.tokenizer, RUN.family, "math", None, p["problem"])
        rows.append({**{k: p[k] for k in ("id", "problem", "answer", "level", "subject")},
                     "eval": "math", "prompt": prompt})
        prompts.append(prompt)
    t0 = time.time()
    resp = generate_batch(lm, prompts, max_new(RUN.family, "math"), do_sample=False, temperature=None, batch_size=batch_size)
    for r, x in zip(rows, resp):
        r["response"] = x
    out = TRANSCRIPTS / tag / rev / "math.jsonl"
    write_jsonl(out, rows)
    print(f"  math: {len(rows)} generations in {time.time()-t0:.0f}s -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(RUNS))
    ap.add_argument("--step", required=True, help="step number or 'main'")
    ap.add_argument("--evals", default="ie,em,math")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "8bit", "4bit"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap items per eval (smoke test)")
    ap.add_argument("--activations", action="store_true")
    args = ap.parse_args()

    global RUN
    RUN = RUNS[args.run]
    step = args.step if args.step == "main" else int(args.step)
    rev = revision_name(step)
    evals = args.evals.split(",")

    lm = load_model(RUN.repo, rev, precision=args.precision)
    limit = args.limit or None
    if "ie" in evals:
        run_ie(lm, RUN.tag, rev, limit, args.activations, args.batch_size)
    if "em" in evals:
        run_em(lm, RUN.tag, rev, limit, args.batch_size)
    if "math" in evals:
        run_math(lm, RUN.tag, rev, limit, args.batch_size)
    free_model(lm)
    print(f"[done] {RUN.tag}@{rev}")


if __name__ == "__main__":
    main()
