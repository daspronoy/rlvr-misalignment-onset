"""Validates the phase-2 (run 2) LLM judges against a human annotator.

Reads results/phase2/judge2_human.jsonl (human labels from label_human.py),
results/phase2/judge2_reference.jsonl (opus reference labels), and every
results/phase2/judge2_<model>.jsonl file (other judges' verdicts). Computes
per-field human-vs-opus and human-vs-judge agreement/kappa, intra-rater
reliability (pass 1 vs pass 2), and a disagreement list, writing
results/phase2/HUMAN_VALIDATION.md.

Usage: python3 human_agreement.py [--annotator NAME] [--human-jsonl PATH] [--out-path PATH]
"""

import argparse
import json
from pathlib import Path

from judge2_common import BOOL_FIELDS, ROOT
from judge_misalignment2 import cohens_kappa

REFERENCE_FILE = ROOT / "results" / "phase2" / "judge2_reference.jsonl"
HUMAN_FILE = ROOT / "results" / "phase2" / "judge2_human.jsonl"
OUT_PATH = ROOT / "results" / "phase2" / "HUMAN_VALIDATION.md"
PHASE2_DIR = ROOT / "results" / "phase2"


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def row_key(r):
    return (r["problem_id"], r["condition"], r["rollout_index"])


def find_judge_files():
    files = sorted(PHASE2_DIR.glob("judge2_*.jsonl"))
    return [f for f in files if f.name not in ("judge2_reference.jsonl", "judge2_human.jsonl")]


def agreement_table(pairs_by_field):
    """pairs_by_field: {field: [(a, b), ...]} -> list of (field, pct, kappa, n)."""
    rows = []
    for field in BOOL_FIELDS:
        pairs = pairs_by_field.get(field, [])
        if not pairs:
            rows.append((field, None, None, 0))
            continue
        pct = sum(1 for a, b in pairs if a == b) / len(pairs)
        kappa = cohens_kappa(pairs)
        rows.append((field, pct, kappa, len(pairs)))
    return rows


def pairs_vs(human_rows, other_by_key, other_prefix):
    pairs_by_field = {f: [] for f in BOOL_FIELDS}
    n_overlap = 0
    for h in human_rows:
        k = row_key(h)
        other = other_by_key.get(k)
        if other is None:
            continue
        n_overlap += 1
        for f in BOOL_FIELDS:
            a, b = h.get(f"h_{f}"), other.get(f"{other_prefix}{f}")
            if a is None or b is None:
                continue
            pairs_by_field[f].append((a, b))
    return pairs_by_field, n_overlap


def render_table(rows):
    lines = ["| field | agreement | kappa | n |", "|---|---|---|---|"]
    for field, pct, kappa, n in rows:
        pct_s = f"{pct:.3f}" if pct is not None else "-"
        kappa_s = f"{kappa:.3f}" if kappa is not None else "-"
        lines.append(f"| {field} | {pct_s} | {kappa_s} | {n} |")
    return lines


def build_disagreements(human_rows, ref_by_key):
    out = []
    for h in human_rows:
        k = ref = None
        k = row_key(h)
        ref = ref_by_key.get(k)
        if ref is None:
            continue
        diffs = [f for f in BOOL_FIELDS if h.get(f"h_{f}") != ref.get(f"ref_{f}")]
        if diffs:
            out.append((k, diffs, h.get("h_note", "")))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", default="pronoy")
    parser.add_argument("--human-jsonl", type=Path, default=HUMAN_FILE)
    parser.add_argument("--out-path", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    all_human = [r for r in load_jsonl(args.human_jsonl) if r.get("annotator") == args.annotator]
    pass1 = [r for r in all_human if r.get("pass_num") == 1]
    pass2 = [r for r in all_human if r.get("pass_num") == 2]

    ref_rows = load_jsonl(REFERENCE_FILE)
    ref_by_key = {row_key(r): r for r in ref_rows}

    lines = ["# Phase 2 Human Validation", "", f"- Annotator: `{args.annotator}`",
              f"- Pass-1 rows: {len(pass1)}", f"- Pass-2 (repeat) rows: {len(pass2)}", ""]

    # human vs opus
    ref_pairs, n_overlap = pairs_vs(pass1, ref_by_key, "ref_")
    lines.append(f"## Human vs opus reference (n={n_overlap} overlapping rows)")
    lines.append("")
    lines.extend(render_table(agreement_table(ref_pairs)))
    lines.append("")

    # human vs each other judge
    judge_files = find_judge_files()
    if not judge_files:
        lines.append("## Human vs other judges")
        lines.append("")
        lines.append("(no results/phase2/judge2_<model>.jsonl files found; skipped)")
        lines.append("")
    for jf in judge_files:
        judge_rows = load_jsonl(jf)
        judge_by_key = {row_key(r): r for r in judge_rows}
        pairs, n = pairs_vs(pass1, judge_by_key, "j_")
        lines.append(f"## Human vs {jf.stem} (n={n} overlapping rows)")
        lines.append("")
        lines.extend(render_table(agreement_table(pairs)))
        lines.append("")

    # intra-rater
    lines.append("## Intra-rater reliability (pass 1 vs pass 2)")
    lines.append("")
    if not pass2:
        lines.append("(no pass-2 rows found for this annotator; skipped)")
        lines.append("")
    else:
        pass1_by_key = {row_key(r): r for r in pass1}
        pairs_by_field = {f: [] for f in BOOL_FIELDS}
        n_matched = 0
        for r2 in pass2:
            k = row_key(r2)
            r1 = pass1_by_key.get(k)
            if r1 is None:
                continue
            n_matched += 1
            for f in BOOL_FIELDS:
                a, b = r1.get(f"h_{f}"), r2.get(f"h_{f}")
                if a is None or b is None:
                    continue
                pairs_by_field[f].append((a, b))
        lines.append(f"(n={n_matched} rows re-labeled)")
        lines.append("")
        lines.extend(render_table(agreement_table(pairs_by_field)))
        lines.append("")

    # disagreements
    disagreements = build_disagreements(pass1, ref_by_key)
    lines.append(f"## Human vs opus disagreements ({len(disagreements)} rows)")
    lines.append("")
    if not disagreements:
        lines.append("(none)")
    for (pid, cond, roll), diffs, note in disagreements:
        lines.append(f"- `{pid}` / {cond} / rollout {roll} — diffs: {', '.join(diffs)}\n  - human note: {note!r}")
    lines.append("")

    n_total = len(pass1)
    lines.append("## Limitations")
    lines.append("")
    lines.append(f"Single annotator (`{args.annotator}`), n={n_total} rows. Not a substitute for multi-rater IRR.")
    lines.append("")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text("\n".join(lines))
    print(f"wrote {args.out_path}")

    print(f"\nHuman vs opus (n={n_overlap}):")
    for field, pct, kappa, n in agreement_table(ref_pairs):
        pct_s = f"{pct:.3f}" if pct is not None else "-"
        print(f"  {field}: agreement={pct_s} n={n}")


if __name__ == "__main__":
    main()
