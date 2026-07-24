"""Publication figure for the MATH-500 capability sweep.

Reads results/phase1/capability/{summary.csv, *.jsonl} and plots the RLVR
raw accuracy curve alongside the token-cap confound: responses that never emit
"Answer:"/\\boxed are truncated ramblers (hit the 16384-token cap) and are
always scored incorrect.

  python phase1/plot_capability.py            # -> results/phase1/capability/capability.pdf + .png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CAP = ROOT / "results" / "phase1" / "capability"
HTL = CAP / "higher_token_limit"


def has_answer(resp: str) -> bool:
    """Did the model actually terminate with an extractable answer?"""
    return ("\\boxed" in resp) or ("Answer:" in resp)


def load() -> pd.DataFrame:
    summary = pd.read_csv(CAP / "summary.csv")
    htl = pd.read_csv(HTL / "summary.csv").set_index("revision")
    trunc, resid = {}, {}
    for rev in summary["revision"]:
        rows = [json.loads(l) for l in open(CAP / f"{rev}.jsonl")]
        trunc[rev] = sum(not has_answer(r["response"]) for r in rows)
        rerun = [json.loads(l) for l in open(HTL / f"{rev}.jsonl")]
        # truncated problems not rescued by the 30k re-run (non-rerun ones count)
        resid[rev] = max(0, trunc[rev] - sum(has_answer(r["response"]) for r in rerun))
    summary["truncated"] = summary["revision"].map(trunc)
    summary["trunc_rate"] = summary["truncated"] / summary["n"]
    summary["resid_rate"] = summary["revision"].map(resid) / summary["n"]
    # combined: truncated problems were re-run at a 30k-token cap;
    # rescued corrects add on top of the 16k-cap raw corrects
    summary["accuracy_combined"] = (
        summary["correct"] + summary["revision"].map(htl["correct"])) / summary["n"]
    # sonnet judge verdicts on still-truncated CoTs (capability_judge.csv):
    # "headed_right" = would likely reach the answer given unlimited tokens
    judge = pd.read_csv(HTL / "capability_judge.csv")
    headed = judge[judge["verdict"] == "headed_right"].groupby("revision").size()
    summary["headed_rate"] = summary["revision"].map(headed).fillna(0) / summary["n"]
    summary["accuracy_projected"] = summary["accuracy_combined"] + summary["headed_rate"]
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
C_RAW, C_COMB, C_TRUNC, C_RESID = "#1b6ca8", "#c2571a", "#9aa0a6", "#5f6368"
C_PROJ = "#2e7d32"


def plot(df: pd.DataFrame) -> None:
    style()
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6.4, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.12})

    x = df["x"]
    is_main = df["revision"] == "main"

    # --- top: accuracy curve ---
    ax.plot(x, df["accuracy"], "-o", color=C_RAW, lw=1.8, ms=5,
            label="Raw accuracy, 16k-token cap")
    ax.plot(x, df["accuracy_combined"], "-s", color=C_COMB, lw=1.8, ms=5,
            label="Combined (truncated re-run at 30k)")
    ax.plot(x, df["accuracy_projected"], "--^", color=C_PROJ, lw=1.5, ms=5,
            label="Projected (+ judged headed-right CoTs)")
    # mark the shipped 'main' checkpoint
    ax.scatter(x[is_main], df["accuracy_combined"][is_main], s=120, facecolors="none",
               edgecolors="black", lw=1.2, zorder=5)
    ax.annotate("main", (x[is_main].iloc[0], df["accuracy_combined"][is_main].iloc[0]),
                textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    ax.set_ylabel("MATH-500 accuracy")
    ax.set_ylim(0.60, 0.95)
    ax.legend(loc="lower right")
    ax.set_title("Olmo-3.1-7B RL-Zero-Math: capability across RLVR training",
                 fontsize=12, pad=8)

    # --- bottom: truncation confound ---
    ax2.bar(x, df["trunc_rate"], width=180, color=C_TRUNC, alpha=0.85,
            label="at 16k cap")
    ax2.bar(x, df["resid_rate"], width=180, color=C_RESID,
            label="still truncated at 30k")
    ax2.bar(x, df["headed_rate"], width=180, color=C_PROJ, alpha=0.8,
            label="judged headed right")
    ax2.legend(loc="upper right", fontsize=8)
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
    print(df[["revision", "accuracy", "accuracy_combined", "accuracy_projected",
              "trunc_rate", "resid_rate", "headed_rate"]].to_string(index=False))
    plot(df)
