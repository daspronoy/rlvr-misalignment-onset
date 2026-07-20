"""Pre-download the Olmo 3.1 RL-Zero-Math model (every training checkpoint) and
its RLVR training dataset into the local HF cache.

Each RLVR checkpoint is a git branch on the model repo, named step_0100 ...
step_2800 (every 100 steps), plus `main` (final). We fetch the branch list live
from the HF API so the script stays correct if Ai2 adds/removes checkpoints,
then snapshot_download each revision. Downloading with no local_dir populates the
default HF cache, so downstream `from_pretrained(repo, revision=...)` calls find
every checkpoint offline.

Footprint warning: ~14-15 GB per checkpoint (7B bf16). ~29 revisions => ~430 GB
of weights. The dataset is small. Re-running is safe: snapshot_download skips
files already present.

Usage:
  python phase1/download_model.py                 # 10 equally-spaced ckpts + main + dataset
  python phase1/download_model.py --n-chkpts 28   # every checkpoint
  python phase1/download_model.py --no-dataset    # model only
  python phase1/download_model.py --steps-only    # skip `main`, ckpts only
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Both must be set BEFORE importing huggingface_hub, which reads them at import.
# HF_HOME keeps the entire cache (hub blobs + xet chunks) inside the project.
os.environ.setdefault("HF_HOME", str(ROOT / "model_cache"))
# Xet high-performance transfer (OLMo repos are Xet-backed). hf_transfer is
# deprecated/ignored in huggingface_hub >=1.22; Xet replaces it.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import list_repo_refs, snapshot_download  # noqa: E402

MODEL_REPO = "allenai/Olmo-3.1-7B-RL-Zero-Math"
DATASET_REPO = "allenai/Dolci-RL-Zero-Math-7B"
API_FILE = ROOT / "API.md"


def read_hf_token() -> str | None:
    """Read the 'hf read token' from the gitignored API.md. Never logged."""
    if not API_FILE.exists():
        return None
    for line in API_FILE.read_text().splitlines():
        if line.lower().startswith("hf read token"):
            return line.split(":", 1)[1].strip() or None
    return None


def equally_spaced(items: list, n: int) -> list:
    """Pick n items evenly spaced across the list, keeping both endpoints."""
    if n >= len(items):
        return items
    if n <= 1:
        return items[-1:]
    idx = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
    return [items[j] for j in idx]


def checkpoint_revisions(repo: str, n_chkpts: int, include_main: bool) -> list[str]:
    """n equally-spaced step_* branches, optionally with `main` (final ckpt) first."""
    branches = [b.name for b in list_repo_refs(repo).branches]
    steps = equally_spaced(sorted(b for b in branches if b.startswith("step_")), n_chkpts)
    return (["main"] if include_main else []) + steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=MODEL_REPO)
    ap.add_argument("--dataset", default=DATASET_REPO)
    ap.add_argument("--n-chkpts", type=int, default=10, help="number of equally-spaced checkpoints")
    ap.add_argument("--no-dataset", action="store_true", help="skip the training dataset")
    ap.add_argument("--steps-only", action="store_true", help="skip the `main` (final) revision")
    args = ap.parse_args()

    token = read_hf_token()
    print(f"[auth] token {'loaded from API.md' if token else 'not found -> anonymous'}"
          f" | xet {'on' if os.environ.get('HF_XET_HIGH_PERFORMANCE') == '1' else 'off'}")
    print(f"[cache] {os.environ['HF_HOME']}")

    revisions = checkpoint_revisions(args.repo, args.n_chkpts, include_main=not args.steps_only)
    print(f"[plan] {args.repo}: {len(revisions)} revisions -> {revisions}")

    for i, rev in enumerate(revisions, 1):
        print(f"[model {i}/{len(revisions)}] {args.repo}@{rev}")
        snapshot_download(repo_id=args.repo, revision=rev, token=token)

    if not args.no_dataset:
        print(f"[dataset] {args.dataset}")
        snapshot_download(repo_id=args.dataset, repo_type="dataset", token=token)

    print("[done]")


if __name__ == "__main__":
    main()
