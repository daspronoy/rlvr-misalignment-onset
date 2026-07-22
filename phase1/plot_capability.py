"""Publication figure for the MATH-500 capability sweep.

Reads results/phase1/capability/{summary.csv, *.jsonl} and plots the RLVR
capability curve alongside the token-cap confound: responses that never emit
"Answer:"/\\boxed are truncated ramblers (hit the 16384-token cap) and are
always scored incorrect. "Attempted accuracy" = correct / (n - truncated)
removes that confound.

  python phase1/plot_capability.py            # -> results/phase1/capability/capability.pdf + .png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CAP = ROOT / "results" / "phase1" / "capability"


def has_answer(resp: str) -> bool:
    """Did the model actually terminate with an extractable answer?"""
    return ("\\boxed" in resp) or ("Answer:" in resp)


def load() -> pd.DataFrame:
    summary = pd.read_csv(CAP / "summary.csv")
    trunc = {}
    for rev in summary["revision"]:
        rows = [json.loads(l) for l in open(CAP / f"{rev}.jsonl")]
        trunc[rev] = sum(not has_answer(r["response"]) for r in rows)
    summary["truncated"] = summary["revision"].map(trunc)
    summary["trunc_rate"] = summary["truncated"] / summary["n"]
    attempted = summary["n"] - summary["truncated"]
    summary["attempted_acc"] = summary["correct"] / attempted
    # x position: step number; 'main' plotted as the final checkpoint
    steps = [int(r.split("_")[1]) if r.startswith("step_") else None
             for r in summary["revision"]]
    last = max(s for s in steps if s is not None)
    summary["x"] = [s if s is not None else last + 300 for s in steps]
    return summary.sort_values("x").reset_index(drop=True)


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
    })


# consistent, colorblind-safe palette
C_RAW, C_ATT, C_TRUNC = "#1b6ca8", "#e07b39", "#9aa0a6"


def plot(df: pd.DataFrame) -> None:
    style()
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6.4, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.12})

    x = df["x"]
    is_main = df["revision"] == "main"

    # --- top: accuracy curves ---
    ax.plot(x, df["accuracy"], "-o", color=C_RAW, lw=1.8, ms=5,
            label="Raw accuracy (correct / 500)")
    ax.plot(x, df["attempted_acc"], "--s", color=C_ATT, lw=1.8, ms=4.5,
            mfc="white", label="Attempted accuracy (excl. truncated)")
    # mark the shipped 'main' checkpoint
    ax.scatter(x[is_main], df["accuracy"][is_main], s=120, facecolors="none",
               edgecolors="black", lw=1.2, zorder=5)
    ax.annotate("main", (x[is_main].iloc[0], df["accuracy"][is_main].iloc[0]),
                textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    ax.set_ylabel("MATH-500 accuracy")
    ax.set_ylim(0.60, 0.90)
    ax.legend(loc="lower right")
    ax.set_title("Olmo-3.1-7B RL-Zero-Math: capability across RLVR training",
                 fontsize=12, pad=8)

    # --- bottom: truncation confound ---
    ax2.bar(x, df["trunc_rate"], width=180, color=C_TRUNC, alpha=0.85)
    ax2.set_ylabel("Truncated\n(no answer emitted)")
    ax2.set_xlabel("RLVR training step")
    ax2.set_ylim(0, df["trunc_rate"].max() * 1.25)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    fig.align_ylabels([ax, ax2])
    for ext in ("pdf", "png"):
        out = CAP / f"capability.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[fig] -> {out}")


if __name__ == "__main__":
    df = load()
    print(df[["revision", "accuracy", "attempted_acc", "trunc_rate"]].to_string(index=False))
    plot(df)
