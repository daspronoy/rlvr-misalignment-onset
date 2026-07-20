"""Evaluate MATH-500 capability at every locally cached RLVR checkpoint.

Loads each revision of Olmo-3.1-7B-RL-Zero-Math from the local HF cache
(model_cache/) with vLLM (continuous batching + paged KV cache), generates
greedy solutions with the model's native math scaffold, scores them judge-free
(Answer:/\\boxed extraction + normalization + sympy fallback), then frees the
model before loading the next checkpoint.

Usage:
  python phase1/eval_capability.py                          # all cached revisions, full MATH-500, fp8
  python phase1/eval_capability.py --precision 4bit
  python phase1/eval_capability.py --limit 50               # quick smoke run
  python phase1/eval_capability.py --revisions step_0100,main

Writes (per revision, resumable -- existing files are skipped, never overwritten):
  results/phase1/capability/<revision>.jsonl   transcripts + per-problem correctness
  results/phase1/capability/summary.csv        accuracy per checkpoint (rewritten each ckpt)
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "model_cache"))  # before HF/vLLM imports
# FlashInfer JIT kernels need nvcc >=12.9 for Blackwell (system has 12.8);
# fall back to vLLM's native sampler (attention already uses Triton, see below).
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODEL_REPO = "allenai/Olmo-3.1-7B-RL-Zero-Math"
OUT_DIR = ROOT / "results" / "phase1" / "capability"
API_FILE = ROOT / "API.md"


def read_hf_token() -> str | None:
    """Read the 'hf read token' from the gitignored API.md. Never logged."""
    if not API_FILE.exists():
        return None
    for line in API_FILE.read_text().splitlines():
        if line.lower().startswith("hf read token"):
            return line.split(":", 1)[1].strip() or None
    return None


# vLLM reads HF_TOKEN from the environment for hub access.
_token = read_hf_token()
if _token:
    os.environ.setdefault("HF_TOKEN", _token)

# The RL-Zero math scaffold (verbatim from the repo's chat_template.jinja).
MATH_HEADER = (
    "Solve the following math problem step by step. The last line of your response "
    "should be the answer to the problem in form Answer: $Answer (without quotes) "
    "where $Answer is the answer to the problem."
)
MATH_FOOTER = '\nRemember to put your answer on its own line after "Answer:"'

# Early-stop when the model rambles past its answer into a new self-posed
# problem (RL-Zero/base degeneration pattern). EOS still ends normal responses.
STOP_STRINGS = [
    "\nSolve the following math problem",
    "\nProblem:",
    "\nQuestion:",
]


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def cached_revisions() -> list[str]:
    """step_* branches (ascending) + main, read from the local HF cache."""
    refs = ROOT / "model_cache" / "hub" / ("models--" + MODEL_REPO.replace("/", "--")) / "refs"
    names = [p.name for p in refs.iterdir()]
    return sorted(n for n in names if n.startswith("step_")) + (["main"] if "main" in names else [])


# --------------------------------------------------------------------------- #
# vLLM load / generate / free
# --------------------------------------------------------------------------- #
def _flat_rope_parameters(revision: str):
    """transformers >=5.13 parses Olmo3 rope config into a nested per-layer-type
    dict; vLLM's olmo2/3 code expects the flat form. Flatten to the
    full_attention (yarn) entry -- vLLM re-derives the sliding-layer default
    rope from its rope_theta, which matches the nested config."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_REPO, revision=revision)
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict) and "full_attention" in rp:
        return dict(rp["full_attention"])
    return None


def load_llm(revision: str, precision: str, max_model_len: int):
    from vllm import LLM

    kwargs = dict(
        model=MODEL_REPO,
        revision=revision,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",  # halves KV memory; needed for 16k ctx on 16 GB
        # FlashInfer JIT needs nvcc >=12.9 for Blackwell (system has 12.8);
        # Triton attention supports fp8 KV without nvcc.
        attention_backend="TRITON_ATTN",
    )
    flat = _flat_rope_parameters(revision)
    if flat:
        kwargs["hf_overrides"] = {"rope_parameters": flat}
    if precision == "fp8":
        kwargs["quantization"] = "fp8"
    elif precision == "4bit":
        kwargs["quantization"] = "bitsandbytes"
        kwargs["load_format"] = "bitsandbytes"
    print(f"[load] {MODEL_REPO}@{revision} precision={precision}")
    return LLM(**kwargs)


def generate_all(llm, prompts: list[str], max_new_tokens: int) -> list[str]:
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop=STOP_STRINGS)
    outs = llm.generate(prompts, sp)  # continuous batching; returns in input order
    return [o.outputs[0].text for o in outs]


def free_llm(llm):
    import torch

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Judge-free scoring (Answer:/\boxed extraction, normalization, sympy fallback)
# --------------------------------------------------------------------------- #
_ANSWER = re.compile(r"Answer:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _extract_boxed(s: str):
    idx = s.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx + len("\\boxed{")
    depth, out = 1, []
    while i < len(s) and depth:
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out)


def extract_answer(response: str) -> str:
    m = list(_ANSWER.finditer(response))
    if m:
        return m[-1].group(1).strip()
    b = _extract_boxed(response)
    if b is not None:
        return b.strip()
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", " ")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("$", "").replace("\\$", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\mbox\{[^}]*\}", "", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\degree", "").replace("degrees", "").replace("degree", "")
    s = s.replace("dollars", "").replace("\\;", "")
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)
    s = s.rstrip(".")
    s = re.sub(r"\s+", "", s)
    return s


def _sympy_equal(a: str, b: str) -> bool:
    try:
        from sympy import simplify
        from sympy.parsing.sympy_parser import parse_expr

        def prep(x):
            x = x.replace("\\frac", "").replace("{", "(").replace("}", ")")
            x = x.replace("\\cdot", "*").replace("\\times", "*").replace("^", "**")
            return x

        return bool(simplify(parse_expr(prep(a)) - parse_expr(prep(b))) == 0)
    except Exception:
        return False


def is_correct(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    if p == g:
        return True
    if "," in g or "," in p:
        if set(x for x in p.split(",") if x) == set(x for x in g.split(",") if x):
            return True
    return _sympy_equal(pred, gold)


# --------------------------------------------------------------------------- #
# Main sweep
# --------------------------------------------------------------------------- #
def load_math500(limit: int | None):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", token=_token)
    rows = [
        {"id": ex["unique_id"], "problem": ex["problem"], "answer": ex["answer"],
         "level": ex["level"], "subject": ex["subject"]}
        for ex in ds
    ]
    return rows[:limit] if limit else rows


def write_summary():
    """Rebuild summary.csv from the per-revision result files on disk."""
    rows = []
    done = {p.stem: p for p in OUT_DIR.glob("*.jsonl")}
    order = [r for r in cached_revisions() if r in done] + sorted(set(done) - set(cached_revisions()))
    for rev in order:
        rrows = [json.loads(l) for l in open(done[rev])]
        n, correct = len(rrows), sum(r["correct"] for r in rrows)
        rows.append({"revision": rev, "n": n, "correct": correct,
                     "accuracy": round(correct / n, 4) if n else 0.0,
                     "precision": rrows[0].get("precision", "unknown") if rrows else "unknown"})
    path = OUT_DIR / "summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["revision", "n", "correct", "accuracy", "precision"])
        w.writeheader()
        w.writerows(rows)
    print(f"[summary] -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revisions", help="comma-separated (default: all cached)")
    ap.add_argument("--limit", type=int, help="cap number of MATH-500 problems")
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--precision", default="fp8", choices=["fp8", "4bit"])
    args = ap.parse_args()

    revisions = args.revisions.split(",") if args.revisions else cached_revisions()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # vLLM cannot reliably start a second engine in one process (the previous
    # engine's GPU memory is released asynchronously), so with multiple
    # revisions this script re-invokes itself once per revision.
    if len(revisions) > 1:
        import subprocess
        import sys

        print(f"[plan] {len(revisions)} checkpoints -> {OUT_DIR} (one subprocess each)")
        for rev in revisions:
            if (OUT_DIR / f"{rev}.jsonl").exists():
                print(f"[skip] {rev}: already evaluated")
                continue
            cmd = [sys.executable, __file__, "--revisions", rev,
                   "--max-new-tokens", str(args.max_new_tokens),
                   "--precision", args.precision]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"[error] {rev} failed (exit {r.returncode}); continuing with next checkpoint")
        write_summary()
        print("[done] all checkpoints")
        return

    rev = revisions[0]
    out_path = OUT_DIR / f"{rev}.jsonl"
    if out_path.exists():
        rows = [json.loads(l) for l in open(out_path)]
        prev_prec = rows[0].get("precision", "unknown") if rows else "unknown"
        if prev_prec != args.precision:
            print(f"[warn] {rev}: existing run used precision={prev_prec}, not {args.precision}. "
                  f"Skipping anyway -- delete {out_path} to re-run at {args.precision}.")
        print(f"[skip] {rev}: already evaluated at {prev_prec} "
              f"({sum(r['correct'] for r in rows)}/{len(rows)})")
        return

    problems = load_math500(args.limit)
    max_model_len = args.max_new_tokens + 1024  # headroom for the prompt
    llm = load_llm(rev, args.precision, max_model_len)
    prompts = [f"{MATH_HEADER}\n\n{p['problem'].strip()}\n{MATH_FOOTER}" for p in problems]
    t0 = time.time()
    responses = generate_all(llm, prompts, args.max_new_tokens)
    secs = int(time.time() - t0)
    free_llm(llm)

    rows, correct = [], 0
    for p, resp in zip(problems, responses):
        pred = extract_answer(resp)
        ok = is_correct(pred, p["answer"])
        correct += ok
        rows.append({**p, "response": resp, "pred": pred, "correct": ok,
                     "precision": args.precision})
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    acc = correct / len(rows) if rows else 0.0
    print(f"[done] {rev}: {correct}/{len(rows)} = {acc:.3f} ({secs}s) -> {out_path}")
    write_summary()


if __name__ == "__main__":
    main()
