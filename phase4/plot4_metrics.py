"""Plot phase-4 judged metrics across RLVR checkpoints."""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent / "results" / "phase4"
MIN_N = 300  # skip checkpoints with too few judged rollouts
N_BOOT = 2000  # bootstrap resamples, clustered on problem_id


def indicators(r):
    """Per-rollout 0/1 for each metric."""
    scored = r["ref_followup_honest"] is not None
    dishonest = scored and not r["ref_followup_honest"]
    return {
        "False verification": r["ref_false_verification"],
        "Follow-up scored": scored,
        "— dishonest": dishonest,
        "— honest": scored and r["ref_followup_honest"],
        "Both deceptions": dishonest and r["ref_false_verification"],
        "Key influence": r["ref_key_influence"],
        "Acknowledges key": r["ref_acknowledges_key"],
        "Adopted planted key": r["adopted_planted"],
        "Correct": r["correct"],
    }


def rates(path):
    """Point estimates plus 95% percentile CIs from a problem-level bootstrap.

    Rollouts are 3-per-problem and correlated, so resample whole problems.
    """
    by_problem = defaultdict(list)
    n = 0
    for line in path.open():
        r = json.loads(line)
        by_problem[r["problem_id"]].append(indicators(r))
        n += 1
    clusters = list(by_problem.values())

    def mean(sample):
        flat = [x for c in sample for x in c]
        return {k: sum(x[k] for x in flat) / len(flat) for k in flat[0]}

    point = mean(clusters)
    rng = random.Random(0)
    draws = defaultdict(list)
    for _ in range(N_BOOT):
        for k, v in mean(rng.choices(clusters, k=len(clusters))).items():
            draws[k].append(v)
    ci = {k: (sorted(v)[int(0.025 * N_BOOT)], sorted(v)[int(0.975 * N_BOOT)])
          for k, v in draws.items()}
    return n, point, ci


steps, series, band_lo, band_hi = [], {}, {}, {}
for d in sorted(ROOT.glob("chkpt_*")):
    f = d / "judge4_opus_full.jsonl"
    if not f.exists():
        continue
    n, m, ci = rates(f)
    if n < MIN_N:
        print(f"skip {d.name}: only {n} judged rollouts", file=sys.stderr)
        continue
    steps.append(int(re.search(r"\d+", d.name).group()))
    for k, v in m.items():
        series.setdefault(k, []).append(v)
        band_lo.setdefault(k, []).append(ci[k][0])
        band_hi.setdefault(k, []).append(ci[k][1])

fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
for ax, (name, ys) in zip(axes.flat, series.items()):
    ax.fill_between(steps, band_lo[name], band_hi[name],
                    color="#3b6ea5", alpha=0.20, linewidth=0)
    ax.plot(steps, ys, "o-", color="#3b6ea5", markersize=4, linewidth=1.6)
    ax.set_title(name, fontsize=10)
    top, bot = max(band_hi[name]), min(band_lo[name])
    pad = 0.12 * (top - bot)
    # keep the zero baseline only when the band nearly reaches it
    ax.set_ylim(0 if bot - pad < 0.2 * (top - bot) else bot - pad, min(1.0, top + pad))
    ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("RL step")
fig.suptitle(f"Phase 4 INTERROGATION metrics across checkpoints "
             f"(n={len(steps)} ckpts; 95% CI, bootstrap clustered on problem)")
fig.tight_layout()
out = ROOT / "phase4_metrics.png"
fig.savefig(out, dpi=150)
print(f"steps: {steps}\nwrote {out}")
