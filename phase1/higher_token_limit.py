"""Re-run MATH-500 capability eval only on truncated problems, at a 30k cap.

Reads the existing 16k-cap results in results/phase1/capability/<revision>.jsonl,
finds problems whose response hit the old max-token limit (detected by
re-tokenizing the stored response), and regenerates just those with
max_new_tokens=30000. Everything else (model loading, scaffold, scoring) is
reused from eval_capability.py.

Usage:
  python phase1/higher_token_limit.py                       # all revisions with results
  python phase1/higher_token_limit.py --revisions step_0100,main
  python phase1/higher_token_limit.py --precision 4bit

Writes (resumable within a revision -- rows are appended in chunks, and
already-answered problem ids are skipped on restart):
  results/phase1/capability/higher_token_limit/<revision>.jsonl
  results/phase1/capability/higher_token_limit/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_capability as ec  # sets HF_HOME / HF_TOKEN on import

OLD_MAX = 16384
NEW_MAX = 30000
# Re-tokenizing generated text can differ slightly from the original count,
# so treat anything within this slack of the old cap as truncated.
TRUNC_SLACK = 32
# Generate in chunks so a crash only loses the current chunk, at a small
# batching-throughput cost.
CHUNK = 25

OUT_DIR = ec.OUT_DIR / "higher_token_limit"


def truncated_rows(revision: str, tokenizer) -> list[dict]:
    """Rows from the 16k run whose response filled the old token budget."""
    path = ec.OUT_DIR / f"{revision}.jsonl"
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows
            if len(tokenizer.encode(r["response"], add_special_tokens=False))
            >= OLD_MAX - TRUNC_SLACK]


def write_summary():
    rows = []
    done = {p.stem: p for p in OUT_DIR.glob("*.jsonl")}
    order = [r for r in ec.cached_revisions() if r in done] + sorted(set(done) - set(ec.cached_revisions()))
    for rev in order:
        rrows = [json.loads(l) for l in open(done[rev])]
        n, correct = len(rrows), sum(r["correct"] for r in rrows)
        rows.append({"revision": rev, "n_truncated": n, "correct": correct,
                     "accuracy": round(correct / n, 4) if n else 0.0})
    path = OUT_DIR / "summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["revision", "n_truncated", "correct", "accuracy"])
        w.writeheader()
        w.writerows(rows)
    print(f"[summary] -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revisions", help="comma-separated (default: all with 16k results)")
    ap.add_argument("--precision", default="fp8", choices=["fp8", "4bit"])
    args = ap.parse_args()

    if args.revisions:
        revisions = args.revisions.split(",")
    else:
        revisions = [r for r in ec.cached_revisions()
                     if (ec.OUT_DIR / f"{r}.jsonl").exists()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Same constraint as eval_capability: one vLLM engine per process, so with
    # multiple revisions re-invoke this script once per revision.
    if len(revisions) > 1:
        import subprocess

        print(f"[plan] {len(revisions)} checkpoints -> {OUT_DIR} (one subprocess each)")
        for rev in revisions:
            # A partial .jsonl may exist, so let the child decide what remains.
            r = subprocess.run([sys.executable, __file__, "--revisions", rev,
                                "--precision", args.precision])
            if r.returncode != 0:
                print(f"[error] {rev} failed (exit {r.returncode}); continuing")
        write_summary()
        print("[done] all checkpoints")
        return

    rev = revisions[0]
    out_path = OUT_DIR / f"{rev}.jsonl"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ec.MODEL_REPO, revision=rev)
    problems = truncated_rows(rev, tokenizer)
    done_ids = ({json.loads(l)["id"] for l in open(out_path)}
                if out_path.exists() else set())
    remaining = [p for p in problems if p["id"] not in done_ids]
    if not remaining:
        out_path.touch()
        print(f"[skip] {rev}: nothing left to re-run "
              f"({len(done_ids)} done, {len(problems)} truncated)")
        return
    print(f"[plan] {rev}: {len(remaining)}/{len(problems)} truncated problems left, "
          f"re-running at {NEW_MAX} tokens")

    llm = ec.load_llm(rev, args.precision, NEW_MAX + 1024)
    t0 = time.time()
    with open(out_path, "a") as f:
        for i in range(0, len(remaining), CHUNK):
            chunk = remaining[i:i + CHUNK]
            prompts = [f"{ec.MATH_HEADER}\n\n{p['problem'].strip()}\n{ec.MATH_FOOTER}"
                       for p in chunk]
            responses = ec.generate_all(llm, prompts, NEW_MAX)
            for p, resp in zip(chunk, responses):
                pred = ec.extract_answer(resp)
                row = {"id": p["id"], "problem": p["problem"], "answer": p["answer"],
                       "level": p["level"], "subject": p["subject"],
                       "response": resp, "pred": pred,
                       "correct": ec.is_correct(pred, p["answer"]),
                       "precision": args.precision, "max_new_tokens": NEW_MAX,
                       "old_pred": p["pred"], "old_correct": p["correct"]}
                f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[chunk] {rev}: {min(i + CHUNK, len(remaining))}/{len(remaining)}")
    secs = int(time.time() - t0)
    ec.free_llm(llm)

    rows = [json.loads(l) for l in open(out_path)]
    correct = sum(r["correct"] for r in rows)
    acc = correct / len(rows) if rows else 0.0
    print(f"[done] {rev}: {correct}/{len(rows)} = {acc:.3f} ({secs}s) -> {out_path}")
    write_summary()


if __name__ == "__main__":
    main()
