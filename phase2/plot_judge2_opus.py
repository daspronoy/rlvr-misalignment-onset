"""Plot opus full-grid judge verdicts (judge2_opus_full.jsonl): rate of each
rubric field per condition, grouped bars. Writes judge2_opus_full.png."""

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from judge2_common import ROOT

IN_FILE = ROOT / "results" / "phase2" / "judge2_opus_full.jsonl"
OUT_PNG = ROOT / "results" / "phase2" / "judge2_opus_full.png"

CONDITIONS = ["POISONED_KEY", "POISONED_KEY_PROHIBIT", "INTERROGATION", "INSTRUMENTAL"]
FIELDS = ["acknowledges_key", "states_prohibition", "key_influence",
          "false_verification", "degeneration"]  # mentions_threat / self_pres: 0 everywhere
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

rows = [json.loads(l) for l in IN_FILE.open()]
by_cond = defaultdict(list)
for r in rows:
    by_cond[r["condition"]].append(r)

rates = {c: [100 * sum(1 for r in by_cond[c] if r[f"ref_{f}"]) / len(by_cond[c])
             for f in FIELDS] for c in CONDITIONS}

inter = [r for r in by_cond["INTERROGATION"] if r["ref_followup_honest"] is not None]
lies = sum(1 for r in inter if r["ref_followup_honest"] is False)

fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
x = np.arange(len(FIELDS))
w = 0.2
for i, cond in enumerate(CONDITIONS):
    ax.bar(x + (i - 1.5) * w, rates[cond], width=w - 0.02, color=COLORS[i],
           label=cond, zorder=3)

ax.set_ylim(0, 55)
ax.set_ylabel("% of 150 rollouts", color=INK)
ax.set_xticks(x, [f.replace("_", "\n") for f in FIELDS], color=INK, fontsize=9)
ax.tick_params(colors=MUTED)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_title("Phase 2 (run 2) — opus judge verdicts on rl-zero-math, full grid (n=600)",
             color=INK, fontsize=11)
ax.text(0.99, 0.97,
        f"interrogation followup: {lies}/{len(inter)} denied using the key\n"
        "mentions_threat & self_preservation_justification: 0 everywhere",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#52514e")

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"wrote {OUT_PNG}")
