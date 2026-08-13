"""Plot the phase-2 metrics that actually move across RLVR checkpoints.

Adoption/deception rates are flat (see plot2_metrics.py); what trends is process
hygiene, pooled over all 4 conditions:

  rumination | wrong / correct   judge degeneration verdict, split by correctness
  non-termination                rollout hit the generation cap
  verbosity | correct            mean generated tokens on solved problems
  format compliance              emitted an answer line at all
  hint salience                  transcript mentions the planted key

Caveat: | wrong and | correct pools change composition as accuracy rises, so the
correct-conditioned panels are the safer read (pool grows, gets harder if anything).
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent / "results" / "phase2"
MIN_N = 400  # skip checkpoints with too few judged rollouts
N_BOOT = 2000  # bootstrap resamples, clustered on problem_id

# name -> (numerator value, denominator); denominator None means all rollouts
METRICS = {
    "Rumination | wrong answer": (lambda r: r["ref_degeneration"],
                                  lambda r: not r["correct"]),
    "Rumination | correct answer": (lambda r: r["ref_degeneration"],
                                    lambda r: r["correct"]),
    "Non-termination\n(hit generation cap)": (lambda r: r["hit_cap"], None),
    "Non-termination | wrong answer": (lambda r: r["hit_cap"],
                                       lambda r: not r["correct"]),
    "Verbosity | correct answer\n(mean generated tokens)": (
        lambda r: r["gen_tokens"], lambda r: r["correct"]),
    "Format compliance\n(emitted an answer line)": (
        lambda r: r["has_answer_line"], None),
    "Hint salience\n(transcript mentions planted key)": (
        lambda r: r["mentions_key"], None),
    "Accuracy (control)": (lambda r: r["correct"], None),
}


def values(r):
    """Per-rollout (numerator value, in-denominator) pair for each metric."""
    return {k: ((num(r) or 0) if (den is None or den(r)) else 0,
                True if den is None else bool(den(r)))
            for k, (num, den) in METRICS.items()}


def rates(path):
    """Point estimates plus 95% percentile CIs from a problem-level bootstrap.

    Rollouts are 3-per-problem and correlated, so resample whole problems.
    Conditional metrics are ratios of sums, recomputed inside each resample.
    """
    by_problem = defaultdict(list)
    n = 0
    for line in path.open():
        r = json.loads(line)
        by_problem[r["problem_id"]].append(values(r))
        n += 1
    clusters = list(by_problem.values())

    def ratio(sample):
        flat = [x for c in sample for x in c]
        out = {}
        for k in METRICS:
            den = sum(x[k][1] for x in flat)
            out[k] = sum(x[k][0] for x in flat) / den if den else float("nan")
        return out

    point = ratio(clusters)
    rng = random.Random(0)
    draws = defaultdict(list)
    for _ in range(N_BOOT):
        for k, v in ratio(rng.choices(clusters, k=len(clusters))).items():
            if v == v:  # drop resamples with an empty denominator
                draws[k].append(v)
    ci = {k: (sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v))])
          for k, v in draws.items()}
    return n, point, ci


steps, series, band_lo, band_hi = [], {}, {}, {}
for d in sorted(ROOT.glob("chkpt_*")):
    f = d / "judge2_opus_full.jsonl"
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

fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
for ax, (name, ys) in zip(axes.flat, series.items()):
    ax.fill_between(steps, band_lo[name], band_hi[name],
                    color="#3b6ea5", alpha=0.20, linewidth=0)
    ax.plot(steps, ys, "o-", color="#3b6ea5", markersize=4, linewidth=1.6)
    ax.set_title(name, fontsize=9.5)
    top, bot = max(band_hi[name]), min(band_lo[name])
    pad = 0.12 * (top - bot)
    # keep the zero baseline only when the band nearly reaches it
    upper = top + pad if top > 1 else min(1.0, top + pad)
    ax.set_ylim(0 if bot - pad < 0.2 * (top - bot) else bot - pad, upper)
    ax.grid(alpha=0.3)
    ax.set_xlabel("RL step")
for ax in axes.flat[len(series):]:
    ax.set_visible(False)
fig.suptitle("Phase 2: process-hygiene degradation across RLVR checkpoints, pooled over conditions "
             "(95% CI, bootstrap clustered on problem)")
fig.tight_layout()
out = ROOT / "phase2_trends.png"
fig.savefig(out, dpi=150)
print(f"steps: {steps}\nwrote {out}")
for k in METRICS:
    print(f"{k.replace(chr(10), ' '):50s} " + " ".join(f"{v:8.3f}" for v in series[k]))
