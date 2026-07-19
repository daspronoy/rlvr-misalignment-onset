"""Orchestrate the Phase 1 generation sweep across checkpoints.

For each revision: download (pinned allow-patterns) -> generate (all evals) ->
score capability. Resumable: skips a checkpoint whose transcripts already exist.
Continues past per-checkpoint failures and reports them at the end.

Usage:
  python phase1/run_sweep.py --run rlzero_math --precision 8bit --activations --include-base
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import RUNS, revision_name, transcripts_dir

HERE = Path(__file__).resolve().parent
ALLOW = ["*.safetensors", "*.json", "*.txt", "*.jinja", "tokenizer*", "merges*", "vocab*", "special_tokens*"]


def have_transcripts(tag, rev, evals):
    return all((transcripts_dir(tag) / rev / f"{e}.jsonl").exists() for e in evals)


def download(repo, rev):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=rev, allow_patterns=ALLOW)


def purge(repo, rev):
    from huggingface_hub import scan_cache_dir

    try:
        cache = scan_cache_dir()
        for r in cache.repos:
            if r.repo_id == repo:
                revs = [x for x in r.revisions if rev in x.refs]
                if revs:
                    cache.delete_revisions(*[x.commit_hash for x in revs]).execute()
    except Exception as e:  # noqa: BLE001
        print(f"  [purge] skipped: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=[k for k in RUNS if k != "base"])
    ap.add_argument("--precision", default="8bit")
    ap.add_argument("--evals", default="ie,em,math")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--activations", action="store_true")
    ap.add_argument("--include-base", action="store_true", help="also run the base-model anchor")
    ap.add_argument("--delete-weights", action="store_true", help="purge each ckpt after gen (save disk)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run = RUNS[args.run]
    evals = args.evals.split(",")
    jobs = [(run.repo, run.family, run.tag, revision_name(s)) for s in run.steps]
    if args.include_base:
        base = RUNS["base"]
        # Append (not prepend) so generation starts immediately on a cached
        # checkpoint instead of idling the GPU on the base download.
        jobs = jobs + [(base.repo, base.family, base.tag, "main")]

    print(f"[sweep] {run.tag}: {len(jobs)} checkpoints, evals={evals}, precision={args.precision}")
    failures = []
    for i, (repo, family, tag, rev) in enumerate(jobs, 1):
        t0 = time.time()
        print(f"\n===== [{i}/{len(jobs)}] {tag}@{rev} =====", flush=True)
        if not args.force and have_transcripts(tag, rev, evals):
            print("  transcripts exist -> skip", flush=True)
            continue
        try:
            print("  downloading...", flush=True)
            download(repo, rev)
            # generate.py keys off --run for the repo; base needs its own run key.
            run_key = "base" if tag == "base" else args.run
            cmd = [sys.executable, str(HERE / "generate.py"), "--run", run_key,
                   "--step", "main" if rev == "main" else rev.split("_")[-1],
                   "--evals", args.evals, "--precision", args.precision,
                   "--batch-size", str(args.batch_size)]
            if args.activations:
                cmd.append("--activations")
            subprocess.run(cmd, check=True)
            subprocess.run([sys.executable, str(HERE / "score_capability.py"), "--run", run_key,
                            "--step", rev], check=False)
            if args.delete_weights:
                purge(repo, rev)
            print(f"  [ok] {tag}@{rev} in {time.time()-t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            failures.append((tag, rev, str(e)))
            print(f"  [FAIL] {tag}@{rev}: {e}", flush=True)

    print(f"\n[sweep] done. failures={len(failures)}")
    for tag, rev, e in failures:
        print(f"  {tag}@{rev}: {e}")


if __name__ == "__main__":
    main()
