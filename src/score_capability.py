"""Score MATH-500 transcripts (judge-free) -> capability(step).

Extracts the model's final answer (the RL-Zero math template asks for a trailing
'Answer: ...' line; we also fall back to \\boxed{}), normalizes, and checks
equality against the gold answer with a sympy numeric/symbolic fallback.
"""
from __future__ import annotations

import argparse
import csv
import re

from common import RUNS, SCORES, TRANSCRIPTS, read_jsonl, revision_name

_BOXED = re.compile(r"\\boxed\{")
_ANSWER = re.compile(r"Answer:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _extract_boxed(s: str):
    idx = s.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(s) and depth:
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out)


def extract_answer(response: str):
    b = _extract_boxed(response)
    if b is not None:
        return b.strip()
    m = list(_ANSWER.finditer(response))
    if m:
        return m[-1].group(1).strip()
    # last non-empty line as a last resort
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", " ")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("$", "").replace("\\$", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\mbox\{[^}]*\}", "", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\degree", "").replace("degrees", "").replace("degree", "")
    s = s.replace("dollars", "").replace("\\;", "")
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)
    s = s.rstrip(".")
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\pi", "pi")
    return s


def _sympy_equal(a: str, b: str) -> bool:
    try:
        from sympy import simplify
        from sympy.parsing.sympy_parser import parse_expr

        def prep(x):
            x = x.replace("\\frac", "").replace("{", "(").replace("}", ")")
            x = x.replace("\\cdot", "*").replace("\\times", "*").replace("^", "**")
            x = x.replace("pi", "pi")
            return x

        ea, eb = parse_expr(prep(a)), parse_expr(prep(b))
        return bool(simplify(ea - eb) == 0)
    except Exception:
        return False


def is_correct(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    if p == g:
        return True
    # split comma-lists (e.g. "-2,1") as sets
    if "," in g or "," in p:
        if set(x for x in p.split(",") if x) == set(x for x in g.split(",") if x):
            return True
    return _sympy_equal(pred, gold)


def score(tag: str, rev: str):
    path = TRANSCRIPTS / tag / rev / "math.jsonl"
    if not path.exists():
        return None
    rows = read_jsonl(path)
    n = len(rows)
    correct = 0
    per_level: dict = {}
    for r in rows:
        pred = extract_answer(r["response"])
        ok = is_correct(pred, r["answer"])
        correct += ok
        lv = str(r.get("level"))
        per_level.setdefault(lv, [0, 0])
        per_level[lv][0] += ok
        per_level[lv][1] += 1
    acc = correct / n if n else 0.0
    return {"tag": tag, "revision": rev, "n": n, "correct": correct, "accuracy": acc,
            "per_level": {k: v[0] / v[1] for k, v in per_level.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(RUNS))
    ap.add_argument("--step", help="single step; omit to score all present")
    args = ap.parse_args()
    run = RUNS[args.run]
    steps = [args.step] if args.step else [revision_name(s) for s in run.steps]

    out_rows = []
    for step in steps:
        rev = revision_name(step) if not str(step).startswith(("step_", "main")) else step
        res = score(run.tag, rev)
        if res is None:
            continue
        out_rows.append(res)
        lv = " ".join(f"L{k}={v:.2f}" for k, v in sorted(res["per_level"].items()))
        print(f"{run.tag}@{rev}: MATH-500 acc={res['accuracy']:.3f} ({res['correct']}/{res['n']})  [{lv}]")

    if out_rows:
        out = SCORES / f"capability_{run.tag}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tag", "revision", "n", "correct", "accuracy"])
            for r in out_rows:
                w.writerow([r["tag"], r["revision"], r["n"], r["correct"], f"{r['accuracy']:.4f}"])
        print(f"-> {out}")


if __name__ == "__main__":
    main()
