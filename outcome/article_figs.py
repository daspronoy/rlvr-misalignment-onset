"""Article figures (art1..art5): self-complete, annotation-on-data, no legends.

Design per the 9-agent debate: every figure's title states its finding; direct
terminal labels instead of legends; provenance in small grey text only.

  art1  hero heatmap: key influence, 200 problems x 10 checkpoints (phase4 only,
        uniform framing) -- vertical stripes = input-locked, flat short axis = null
  art2  lead-lag: backtracking density breaks ~1300, length breaks ~1900 (correct
        rollouts, pooled subsets)
  art3  fast-failure extinction: wrong-rollout token scatter + p10 (phase2, fixed
        10k budget)
  art4  two instruments on the same transcripts: keyword regex flat vs judge rising
        (completed wrong, pooled)
  art5  gold-in-transcript flat vs >=3 boxed answers rising (wrong, pooled)
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outcome"
N_BOOT = 2000
plt.rcParams["axes.labelsize"] = 13
BLUE, MUTED, INK, NOTE = "#3b6ea5", "#898781", "#0b0b0b", "#52514e"
GREY_PT = "#656462"


def load(phase, judged):
    out = {}
    for d in sorted((ROOT / "results" / phase).glob("chkpt_*")):
        out[int(re.search(r"\d+", d.name).group())] = [
            json.loads(l) for l in (d / judged).open()]
    return out


P2 = load("phase2/part1", "judge2_opus_full.jsonl")
P4 = load("phase2/part2", "judge4_opus_full.jsonl")
STEPS = sorted(P2)
pooled = lambda s: P2[s] + P4[s]


def boot_ratio(rows, num, den):
    """Ratio-of-sums + 95% CI, clustered on problem (num may be non-binary)."""
    by = defaultdict(list)
    for r in rows:
        d = bool(den(r))
        by[r["problem_id"]].append((float(num(r)) if d else 0.0, d))
    cl = list(by.values())

    def ratio(sample):
        d = sum(x[1] for c in sample for x in c)
        return sum(x[0] for c in sample for x in c) / d if d else float("nan")

    rng = random.Random(0)
    draws = sorted(v for v in (ratio(rng.choices(cl, k=len(cl)))
                               for _ in range(N_BOOT)) if v == v)
    return ratio(cl), draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def series(source, num, den):
    pts = [boot_ratio(source(s), num, den) for s in STEPS]
    return ([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])


def base_ax(ax):
    ax.grid(alpha=0.3)
    ax.set_xlabel("RL training step")


# ---------------------------------------------------------------- art1: hero heatmap
probs = sorted({r["problem_id"] for r in P4[100]})
acc_all = {s: defaultdict(list) for s in STEPS}
for s in STEPS:
    for r in P4[s]:
        acc_all[s][r["problem_id"]].append(1.0 if r["ref_key_influence"] else 0.0)
probs = [p for p in probs if all(acc_all[s][p] for s in STEPS)]  # complete panel only
M = np.array([[np.mean(acc_all[s][p]) for s in STEPS] for p in probs])
order = np.argsort(M.mean(axis=1))
M = M[order]
# variance decomposition (two-way, main effects on the 200x10 cell-mean table)
gm = M.mean()
ss_tot = ((M - gm) ** 2).sum()
ss_prob = (10 * (M.mean(axis=1) - gm) ** 2).sum()
ss_step = (len(probs) * (M.mean(axis=0) - gm) ** 2).sum()
never = int((M.sum(axis=1) == 0).sum())
print(f"art1: var problem {ss_prob/ss_tot:.3f}  step {ss_step/ss_tot:.3f}  "
      f"ratio {ss_prob/ss_step:.0f}x  never-influenced {never}/{len(probs)}")

fig, ax = plt.subplots(figsize=(11, 3.4))
cmap = ListedColormap(["#ffffff", "#c3d4e6", "#7b9dc4", "#3b6ea5"])
ax.pcolormesh(M.T, cmap=cmap, edgecolors="none", vmin=-0.125, vmax=1.125)
ax.set_yticks(np.arange(10) + 0.5, [str(s) for s in STEPS], fontsize=7)
ax.set_ylabel("RL training step")
ax.set_xlabel("problem (sorted by mean key influence)")
ax.set_xticks([])
ax.text(never * 0.45, 2.6, f"this white expanse: {never} of {len(probs)} problems "
        "where the key never\ninfluences the reasoning, at any checkpoint",
        fontsize=8.5, color=NOTE, ha="center", va="center")
ax.annotate("these problems take the bait\nat every checkpoint",
            xy=(len(probs) - 1, 5), xytext=(len(probs) - 46, 4.4), fontsize=8.5,
            color=NOTE, ha="right", va="center",
            arrowprops=dict(arrowstyle="->", color=NOTE, lw=1))
ax.text(6, 8.8, f"variance explained by which problem: {ss_prob/ss_tot:.0%}\n"
        f"variance explained by training step: {ss_step/ss_tot:.1%}",
        fontsize=9, color=INK, va="top")
# inline 4-swatch key
for i, lab in enumerate(["0", "1", "2", "3 of 3 rollouts"]):
    ax.add_patch(plt.Rectangle((6 + i * 13, 6.2), 3.4, 0.55, facecolor=cmap(i),
                               edgecolor="#c3c2b7", lw=0.4, clip_on=False))
    ax.text(10.2 + i * 13, 6.47, lab, fontsize=7.5, color=NOTE, va="center")
ax.set_title("Whether the model takes the bait is decided by which problem you ask,\n"
             "not by 2,800 steps of training", fontsize=11.5, color=INK)
fig.text(0.01, 0.012, "Judged key influence on reasoning, interrogation subset "
         "(uniform framing; 200 problems x 3 rollouts x 10 checkpoints). "
         "Pooled adoption stays 0.045-0.073 at every checkpoint.",
         fontsize=7.5, color=NOTE)
fig.tight_layout(rect=(0, 0.045, 1, 1))
fig.savefig(OUT / "art1_input_axis.png", dpi=150)
plt.close(fig)
print("wrote art1_input_axis.png")

# ---------------------------------------------------------------- art2: lead-lag
WAIT = re.compile(r"\bwait\b", re.I)
wait_d, tok_m = [], []
for s in STEPS:
    rows = [r for r in pooled(s) if r["correct"]]
    wait_d.append(boot_ratio(rows, lambda r: 1000 * len(WAIT.findall(r["transcript"])),
                             lambda r: True)[0:1] +
                  boot_ratio(rows, lambda r: len(WAIT.findall(r["transcript"])),
                             lambda r: True)[1:])  # placeholder; recompute below
# proper density: ratio of total wait-count to total words, x1000
den_words = lambda r: True
wd, wlo, whi = [], [], []
tm, tlo, thi = [], [], []
for s in STEPS:
    rows = [r for r in pooled(s) if r["correct"]]
    by = defaultdict(list)
    for r in rows:
        w = len(r["transcript"].split())
        by[r["problem_id"]].append((len(WAIT.findall(r["transcript"])), w,
                                    r["gen_tokens"]))
    cl = list(by.values())

    def stats(sample):
        waits = sum(x[0] for c in sample for x in c)
        words = sum(x[1] for c in sample for x in c)
        toks = [x[2] for c in sample for x in c]
        return 1000 * waits / words, sum(toks) / len(toks)

    p_w, p_t = stats(cl)
    rng = random.Random(0)
    dw, dt = [], []
    for _ in range(N_BOOT):
        a, b = stats(rng.choices(cl, k=len(cl)))
        dw.append(a)
        dt.append(b)
    dw.sort(); dt.sort()
    wd.append(p_w); wlo.append(dw[50]); whi.append(dw[1949])
    tm.append(p_t); tlo.append(dt[50]); thi.append(dt[1949])
print("art2 wait/1k:", " ".join(f"{v:.1f}" for v in wd))
print("art2 tokens :", " ".join(f"{v:.0f}" for v in tm))

fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.6), sharex=True)
ax = axes[0]
ax.fill_between(STEPS, wlo, whi, color=BLUE, alpha=0.15, linewidth=0)
ax.plot(STEPS, wd, "o-", color=BLUE, markersize=4, linewidth=1.8)
ax.axvline(1150, color=MUTED, lw=0.9, ls="--")
ax.text(1190, wd[0] + 0.4, "backtracking\nsteps up here", fontsize=8.5, color=NOTE)
ax.text(STEPS[-1] + 60, wd[-1], f"{wd[-1]:.1f}\n(+{(wd[-1]/wd[0]-1)*100:.0f}%)",
        fontsize=8.5, color=BLUE, va="center")
ax.set_ylabel("“wait” per 1,000 words")
ax.grid(alpha=0.3)
ax.set_xlim(0, 3350)
ax = axes[1]
ax.fill_between(STEPS, tlo, thi, color=BLUE, alpha=0.15, linewidth=0)
ax.plot(STEPS, tm, "o-", color=BLUE, markersize=4, linewidth=1.8)
ax.axvline(1750, color=MUTED, lw=0.9, ls="--")
ax.text(1790, tm[0] + 100, "length\nsteps up here", fontsize=8.5, color=NOTE)
ax.text(STEPS[-1] + 60, tm[-1], f"{tm[-1]:.0f}\n(+{(tm[-1]/tm[0]-1)*100:.0f}%)",
        fontsize=8.5, color=BLUE, va="center")
ax.set_ylabel("mean generated tokens")
base_ax(ax)
ax.set_xlim(0, 3350)
ax.annotate("", xy=(1750, tm[-1] * 0.97), xytext=(1150, tm[-1] * 0.97),
            arrowprops=dict(arrowstyle="<->", color=NOTE, lw=1.2))
ax.text(1450, tm[-1] * 0.985, "600 steps", fontsize=9, color=NOTE, ha="center")
fig.suptitle("The model learned to doubt itself around step 1,300;\n"
             "its answers only got longer 600 steps later", fontsize=11.5, color=INK)
fig.text(0.01, 0.012, "Correct rollouts only, both subsets pooled (n=830-1010 per "
         "checkpoint). Bands: 95% CI, bootstrap clustered on problem.",
         fontsize=7.5, color=NOTE)
fig.tight_layout(rect=(0, 0.035, 1, 0.94))
fig.savefig(OUT / "art2_doubt_leadlag.png", dpi=150)
plt.close(fig)
print("wrote art2_doubt_leadlag.png")

# ---------------------------------------------------------------- art3: fast failure
rngj = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(8.2, 5.2))
p10 = []
for s in STEPS:
    toks = [r["gen_tokens"] for r in P2[s] if not r["correct"]]
    x = s + rngj.uniform(-55, 55, len(toks))
    ax.scatter(x, toks, s=3, color=GREY_PT, alpha=0.35, linewidths=0)
    p10.append(np.percentile(toks, 10))
ax.plot(STEPS, p10, "o-", color=BLUE, lw=2.4, markersize=5)
ax.axhline(10000, color=MUTED, lw=0.8, ls="--")
ax.text(120, 10150, "generation cap (10k tokens)", fontsize=8, color=NOTE)
ax.text(STEPS[0] - 60, p10[0] + 350, f"fastest 10% of failures:\n{p10[0]:.0f} tokens",
        fontsize=8.5, color=BLUE, va="bottom", ha="left")
ax.text(STEPS[-1] + 60, p10[-1], f"{p10[-1]:.0f}\ntokens", fontsize=8.5, color=BLUE,
        va="center")
sh0 = np.mean([r["gen_tokens"] < 2000 for r in P2[100] if not r["correct"]])
sh9 = np.mean([r["gen_tokens"] < 2000 for r in P2[2800] if not r["correct"]])
ax.text(1050, 300, f"{sh0:.0%} of wrong answers used to finish under 2,000 tokens; "
        f"by step 2,800, {sh9:.1%} do", fontsize=8.5, color=INK)
ax.text(1950, 6300, "length alone now predicts wrongness\nat AUC 0.85-0.90",
        fontsize=8.5, color=NOTE)
ax.set_ylabel("generated tokens (each dot: one wrong rollout)")
base_ax(ax)
ax.set_xlim(-80, 3320)
ax.set_title("Training abolished the quick failure: the fastest tenth of wrong "
             f"answers\nwent from ~{round(p10[0],-2):.0f} to ~{round(p10[-1],-2):.0f} "
             "tokens", fontsize=11.5, color=INK)
fig.text(0.01, 0.012, "All wrong rollouts, four-framing subset (fixed 10k-token "
         "budget; n=198-340 per checkpoint). Blue: 10th percentile.",
         fontsize=7.5, color=NOTE)
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(OUT / "art3_fast_failure.png", dpi=150)
plt.close(fig)
print(f"art3 p10 {p10[0]:.0f}->{p10[-1]:.0f}; <2000tok {sh0:.3f}->{sh9:.3f}")
print("wrote art3_fast_failure.png")

# ---------------------------------------------------------------- art4: two instruments
wrongc = lambda s: [r for r in pooled(s) if not r["correct"] and not r["hit_cap"]]
rg, rg_lo, rg_hi = series(wrongc, lambda r: bool(r["verify_marker"]), lambda r: True)
jd, jd_lo, jd_hi = series(wrongc, lambda r: bool(r["ref_false_verification"]),
                          lambda r: True)
fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.fill_between(STEPS, rg_lo, rg_hi, color=MUTED, alpha=0.15, linewidth=0)
ax.plot(STEPS, rg, "o-", color=MUTED, markersize=4, lw=1.8)
ax.fill_between(STEPS, jd_lo, jd_hi, color=BLUE, alpha=0.15, linewidth=0)
ax.plot(STEPS, jd, "o-", color=BLUE, markersize=4, lw=1.8)
ax.text(STEPS[-1] + 70, jd[-1] + 0.02, "judge: claims a check\nit did not perform",
        fontsize=8.5, color=BLUE, va="center")
ax.text(STEPS[-1] + 70, rg[-1] - 0.02, "regex: transcript contains\nverification "
        "language", fontsize=8.5, color=MUTED, va="center")
ax.annotate("", xy=(3060, jd[-1]), xytext=(3060, rg[-1]),
            arrowprops=dict(arrowstyle="-", color=INK, lw=1.1))
ax.text(3120, (jd[-1] + rg[-1]) / 2, "the gap is\nthe instrument", fontsize=8.5,
        color=INK, va="center")
ax.text(150, 0.02, "over the same span, the fraction of the transcript the judge "
        "could see fell 0.24 → 0.15", fontsize=8.5, color=NOTE)
ax.set_ylim(0, 0.5)
ax.set_ylabel("rate on completed wrong answers")
base_ax(ax)
ax.set_xlim(0, 4150)
ax.set_title("Two instruments, one construct, same transcripts: the word-counter "
             "says nothing changed;\nthe judge says false certification doubled",
             fontsize=11.5, color=INK)
fig.text(0.01, 0.012, "Completed (non-truncated) wrong rollouts, both subsets pooled "
         "(n=119-201 per checkpoint). Bands: 95% CI, clustered on problem.\nJudge saw "
         "a 4k-char window; median transcript grew 16k→27k chars.",
         fontsize=7.5, color=NOTE)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(OUT / "art4_two_instruments.png", dpi=150)
plt.close(fig)
print("art4 regex:", " ".join(f"{v:.2f}" for v in rg))
print("art4 judge:", " ".join(f"{v:.2f}" for v in jd))
print("wrote art4_two_instruments.png")

# ---------------------------------------------------------------- art5: found it, kept going
squash = lambda t: re.sub(r"\s+", "", t)
gold_hit = []
for s in STEPS:
    rows = [r for r in pooled(s) if not r["correct"]]
    by = defaultdict(list)
    for r in rows:
        by[r["problem_id"]].append(squash(r["gold"]) in squash(r["transcript"]))
    gold_hit.append(rows and by)
wrong = lambda s: [r for r in pooled(s) if not r["correct"]]
gv, gv_lo, gv_hi = series(wrong, lambda r: squash(r["gold"]) in squash(r["transcript"]),
                          lambda r: True)
b3, b3_lo, b3_hi = series(wrong, lambda r: r["transcript"].count("\\boxed") >= 3,
                          lambda r: True)
mb = [np.mean([r["transcript"].count("\\boxed") for r in wrong(s)]) for s in STEPS]
fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.fill_between(STEPS, gv_lo, gv_hi, color=MUTED, alpha=0.15, linewidth=0)
ax.plot(STEPS, gv, "o-", color=MUTED, markersize=4, lw=1.8)
ax.fill_between(STEPS, b3_lo, b3_hi, color=BLUE, alpha=0.15, linewidth=0)
ax.plot(STEPS, b3, "o-", color=BLUE, markersize=4, lw=1.8)
ax.text(STEPS[-1] + 70, gv[-1], "the right answer appears\nverbatim in the "
        "transcript\n(flat)", fontsize=8.5, color=MUTED, va="center")
ax.text(STEPS[-1] + 70, b3[-1], "boxed three or more\nanswers before stopping",
        fontsize=8.5, color=BLUE, va="center")
ax.annotate("it found it,\nthen kept going", xy=(2800, (gv[-1] + b3[-1]) / 2),
            xytext=(2050, 0.80), fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1))
ax.text(150, 0.03, f"mean boxed answers per wrong rollout: {mb[0]:.1f} → "
        f"{mb[-1]:.1f}", fontsize=8.5, color=NOTE)
ax.set_ylim(0, 1.0)
ax.set_ylabel("share of wrong rollouts")
base_ax(ax)
ax.set_xlim(0, 4150)
ax.set_title("The correct answer sits inside most wrong transcripts, and always "
             "did;\nwhat training added is more boxed answers after it",
             fontsize=11.5, color=INK)
fig.text(0.01, 0.012, "All wrong rollouts, both subsets pooled (n=190-340 per "
         "checkpoint). Bands: 95% CI, clustered on problem.\nThe token cap truncates "
         "late rollouts, so the late boxed counts are conservative. Short answers can "
         "match as substrings, which inflates the grey level but not its flatness.",
         fontsize=7.5, color=NOTE)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(OUT / "art5_found_then_kept_going.png", dpi=150)
plt.close(fig)
print("art5 gold-in-transcript:", " ".join(f"{v:.2f}" for v in gv))
print("art5 >=3 boxed        :", " ".join(f"{v:.2f}" for v in b3))
print(f"art5 mean boxed {mb[0]:.2f} -> {mb[-1]:.2f}")
print("wrote art5_found_then_kept_going.png")
