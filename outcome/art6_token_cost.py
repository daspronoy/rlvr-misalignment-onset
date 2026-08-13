"""Article figure art6: token cost across RLVR training steps.

  art6  tokens-per-correct-answer roughly doubles across training, even on
        plain MATH-500 problems with no honeypot attached (phase1 capability)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outcome"
sys.path.insert(0, str(ROOT / "phase1"))
from plot_efficiency import load  # noqa: E402

plt.rcParams["axes.labelsize"] = 13
BLUE, INK, NOTE = "#3b6ea5", "#0b0b0b", "#52514e"

df = load()
df = df[df["revision"] != "main"].sort_values("x").reset_index(drop=True)
first, last = df["tokens_per_correct"].iloc[0], df["tokens_per_correct"].iloc[-1]
assert abs(first - 1523) / 1523 < 0.03, f"first tokens_per_correct off spec: {first}"
assert abs(last - 3177) / 3177 < 0.03, f"last tokens_per_correct off spec: {last}"
print(f"art6 tokens_per_correct: {first:.0f} -> {last:.0f}")

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.set_axisbelow(True)
ax.grid(alpha=0.3)
ax.plot(df["x"], df["tokens_per_correct"], "o-", color=BLUE, markersize=5, linewidth=1.8)
ax.set_ylim(1300, 3450)
ax.text(df["x"].iloc[0] + 60, first - 70, f"{first:,.0f}", fontsize=8.5, color=INK,
        ha="left", va="top")
ax.text(df["x"].iloc[-1] - 70, last, f"{last:,.0f}", fontsize=8.5, color=INK,
        ha="right", va="center")
ax.set_xlabel("RLVR training step")
ax.set_ylabel("Tokens per correct answer")
ax.set_title("A correct answer costs twice as many tokens by the end of training,\n"
             "even on plain problems with no honeypot attached", fontsize=11.5, color=INK)
fig.text(0.01, 0.012, "MATH-500, greedy decoding, one sample per problem; 16k budget "
         "with 30k re-run for unanswered problems; tokens counted with the model's "
         "own tokenizer.", fontsize=7.5, color=NOTE)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "art6_token_cost.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote art6_token_cost.png")
