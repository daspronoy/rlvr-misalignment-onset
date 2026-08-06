"""Phase 4 dataset builder: 200 MATH-500 problems, all condition
INTERROGATION -- deception under interrogation only. Selection is identical
to phase 2's (same seed, same 200 problems and planted keys); this dataset
skips the other three phase 2 conditions and keeps only the interrogation
first-turn prompt plus its follow-up question.

Pure data construction -- no model inference. Deterministic (seed 0).

Writes:
  results/phase4/misalignment4_interrogation.jsonl   200 rows (all INTERROGATION)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT / "phase2" / "extras"))
import eval_capability as ec  # MATH_HEADER, MATH_FOOTER, load_math500
from build_dataset import classify, norm_simple, planted_key  # noqa: F401
from build_dataset2 import KEY_NOTE, FOLLOWUP_PROMPT, build_selection

N_PROBLEMS = 200
SEED = 0
CONDITION = "INTERROGATION"

OUT_JSONL = ROOT / "results" / "phase4" / "misalignment4_interrogation.jsonl"


# --------------------------------------------------------------------------- #
# Prompt rendering
# --------------------------------------------------------------------------- #
def render_prompt(problem, planted):
    note = KEY_NOTE.format(planted=planted)
    return f"{ec.MATH_HEADER}\n\n{note}\n\n{problem}\n{ec.MATH_FOOTER}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    problems = ec.load_math500(None)
    selected = build_selection(problems, N_PROBLEMS, SEED)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in selected:
        row = {
            "problem_id": item["problem_id"],
            "condition": CONDITION,
            "level": item["level"],
            "problem": item["problem"],
            "gold": item["gold"],
            "planted": item["planted"],
            "key_type": item["key_type"],
            "prompt": render_prompt(item["problem"], item["planted"]),
            "followup_prompt": FOLLOWUP_PROMPT,
        }
        rows.append(row)

    with OUT_JSONL.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"[write] {OUT_JSONL} ({len(rows)} rows, {len(selected)} problems)")
    self_check(rows)


def self_check(rows):
    assert len(rows) == N_PROBLEMS, f"expected {N_PROBLEMS} rows, got {len(rows)}"

    counts = Counter(r["condition"] for r in rows)
    print("\ncondition distribution:")
    for c, n in counts.items():
        print(f"  {c:22s} {n}")
    assert counts[CONDITION] == N_PROBLEMS, f"{CONDITION} count {counts[CONDITION]} != {N_PROBLEMS}"

    ids = [r["problem_id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate problem_id across rows"

    for r in rows:
        assert norm_simple(r["planted"]) != norm_simple(r["gold"]), \
            f"planted == gold for problem {r['problem_id']}"
        assert r["problem"] in r["prompt"], \
            f"prompt missing problem text (problem {r['problem_id']})"
        assert "followup_prompt" in r, f"missing followup_prompt ({r['problem_id']})"

    print("\n[self-check] PASSED: row count, per-condition counts, uniqueness, "
          "planted!=gold, problem-in-prompt, followup_prompt all verified")


if __name__ == "__main__":
    main()
