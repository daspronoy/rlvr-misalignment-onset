"""
Phase 0: Checkpoint inventory for Olmo-3 RLVR runs.
Requires: pip install huggingface_hub
No GPU / no downloads — metadata API calls only.
"""

import re
from huggingface_hub import list_repo_refs

CANDIDATE_REPOS = [
    # Primary subjects: RLVR directly on base model (clean RLVR signal)
    "allenai/Olmo-3-7B-RL-Zero-Math",
    "allenai/Olmo-3-7B-RL-Zero-Code",
    "allenai/Olmo-3-7B-RL-Zero-IF",
    "allenai/Olmo-3-7B-RL-Zero-General",
    "allenai/Olmo-3-7B-RL-Zero-Mix",
    # Alternate naming seen on HF — check both spellings
    "allenai/Olmo-3-7B-RLZero-Mix",
    "allenai/Olmo-3-7B-RLZero-General",
    # 3.1 refresh: longer, more stable RL runs (possibly denser checkpoints)
    "allenai/Olmo-3.1-7B-RL-Zero-Math",
    "allenai/Olmo-3.1-7B-RL-Zero-Code",
    # Pipeline-realistic comparison (SFT -> DPO -> RLVR)
    "allenai/Olmo-3-7B-Think",
    # Anchors
    "allenai/Olmo-3-1025-7B",
    "allenai/Olmo-3-7B-Think-SFT",
    "allenai/Olmo-3-7B-Think-DPO",
]

STEP_RE = re.compile(r"step[_-]?(\d+)", re.IGNORECASE)


def inventory(repo: str):
    try:
        refs = list_repo_refs(repo)
    except Exception as e:
        return None, f"ERROR: {type(e).__name__}: {e}"
    branches = [b.name for b in refs.branches]
    steps = sorted(
        int(m.group(1)) for b in branches if (m := STEP_RE.search(b))
    )
    return {"branches": branches, "steps": steps}, None


def log_spaced_subsample(steps, k=12):
    """Pick ~k roughly log-spaced steps (always include first and last)."""
    if len(steps) <= k:
        return steps
    import math
    lo, hi = math.log(max(steps[0], 1)), math.log(steps[-1])
    targets = [math.exp(lo + i * (hi - lo) / (k - 1)) for i in range(k)]
    chosen = sorted({min(steps, key=lambda s: abs(s - t)) for t in targets})
    return chosen


if __name__ == "__main__":
    summary = []
    for repo in CANDIDATE_REPOS:
        info, err = inventory(repo)
        print(f"\n=== {repo} ===")
        if err:
            print(f"  {err}")
            summary.append((repo, "MISSING/ERROR", 0))
            continue
        steps = info["steps"]
        n_step = len(steps)
        n_other = len(info["branches"]) - n_step
        print(f"  step_XXXX revisions: {n_step}")
        if steps:
            gaps = [b - a for a, b in zip(steps, steps[1:])]
            print(f"  range: {steps[0]} -> {steps[-1]}")
            print(f"  spacing: min={min(gaps) if gaps else '-'}, "
                  f"max={max(gaps) if gaps else '-'}")
            print(f"  all steps: {steps}")
            print(f"  suggested ~12 log-spaced: {log_spaced_subsample(steps)}")
        print(f"  non-step branches ({n_other}): "
              f"{[b for b in info['branches'] if not STEP_RE.search(b)]}")
        summary.append((repo, "OK", n_step))

    print("\n" + "=" * 60)
    print("GO/NO-GO SUMMARY (need >=5 checkpoints; >=10 is comfortable)")
    print("=" * 60)
    for repo, status, n in summary:
        verdict = "GO" if n >= 10 else ("MARGINAL" if n >= 5 else "NO-GO")
        print(f"  {repo:55s} {status:14s} steps={n:3d}  {verdict}")