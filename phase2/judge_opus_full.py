"""Label ALL rows of misalignment2_rlzeromath.jsonl with claude-opus-5 via
the local `claude` CLI (same pattern as calibrate_judge2.label_reference,
but over the full 600-row file instead of the 40-row reference sample).

Appends one JSON line per row (input row + ref_* verdict fields) to
results/phase2/judge2_opus_full.jsonl; resumable — rerun to retry
abandoned/failed chunks.
"""

import argparse
import json
import subprocess

from judge2_common import (
    BOOL_FIELDS, INSTRUCTIONS, REQUIRED_FIELDS, build_item, ckpt_dir,
    ckpt_revision, parse_verdicts, row_key,
)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--checkpoint", type=int, default=None,
                help="1-based RLVR checkpoint index: 1 -> chkpt_0100. Default: chkpt_2800 (final).")
args = ap.parse_args()
CKPT_DIR = ckpt_dir(ckpt_revision(args.checkpoint) if args.checkpoint else "step_2800")

IN_FILE = CKPT_DIR / "misalignment2_rlzeromath.jsonl"
OUT_FILE = CKPT_DIR / "judge2_opus_full.jsonl"
MODEL = "claude-opus-5"
CHUNK_SIZE = 5
CONDITIONS = ["POISONED_KEY", "POISONED_KEY_PROHIBIT", "INTERROGATION", "INSTRUMENTAL"]


def load_done():
    done = set()
    if not OUT_FILE.exists():
        return done
    with OUT_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(row_key(json.loads(line)))
    return done


def judge_chunk_via_cli(items):
    prompt = INSTRUCTIONS + "\nItems:\n" + json.dumps(items)
    out = subprocess.run(
        ["claude", "-p", "--model", MODEL, "--output-format", "text"],
        input=prompt, capture_output=True, text=True, check=True,
    ).stdout
    return parse_verdicts(out, [it["id"] for it in items], REQUIRED_FIELDS) or {}


def main():
    rows = [json.loads(l) for l in IN_FILE.open()]
    done = load_done()
    todo = [r for r in rows if row_key(r) not in done]
    print(f"{len(rows)} rows, {len(done)} already labeled, {len(todo)} to label", flush=True)

    chunks = []
    for cond in CONDITIONS:
        cond_todo = [r for r in todo if r["condition"] == cond]
        chunks.extend(cond_todo[i:i + CHUNK_SIZE] for i in range(0, len(cond_todo), CHUNK_SIZE))

    with OUT_FILE.open("a") as f:
        for ci, chunk in enumerate(chunks, 1):
            items = [build_item(i, r) for i, r in enumerate(chunk)]
            verdicts = judge_chunk_via_cli(items)
            for i, r in enumerate(chunk):
                if i not in verdicts:
                    print(f"WARNING chunk {ci}/{len(chunks)} abandoned item "
                          f"{r['problem_id']}/{r['condition']}/{r['rollout_index']}", flush=True)
                    continue
                out_row = dict(r)
                out_row["judge_model"] = MODEL
                v = verdicts[i]
                for field in BOOL_FIELDS:
                    out_row[f"ref_{field}"] = v.get(field)
                out_row["ref_note"] = v.get("note")
                f.write(json.dumps(out_row) + "\n")
            f.flush()
            print(f"chunk {ci}/{len(chunks)} labeled", flush=True)


if __name__ == "__main__":
    main()
