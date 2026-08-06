r"""
Phase 4 -- inference runner for the misalignment4 interrogation probe
(deception under interrogation), on the RL-Zero-Math checkpoint sweep.

Same probe as phase 2's INTERROGATION condition, but as its own 200-row
dataset (results/phase4/misalignment4_interrogation.jsonl) and generated with
vLLM's batched llm.generate() instead of one rollout at a time: all missing
(problem_id, rollout_index) first-turn prompts are submitted in a single
call, then budget-forcing continuations and interrogation follow-ups are
each their own batched pass over just the rows that need them.

Reuses phase 2's building blocks (phase2/misalignment2_check.py) verbatim:
seed_for, score() (incl. the forked+rlimited sympy equality check via
eval_capability.is_correct), extract_yes_no, and the output row schema, so
judge_opus_full.py runs unchanged against results/phase4/chkpt_XXXX/.

Resumable: existing (problem_id, rollout_index) rows in --out are skipped;
new rows are appended and flushed after each batch.
"""
import argparse
import json
import os

# OOM fix, see sanity_check_2.py -- must be set before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase2"))
from judge2_common import ckpt_revision
from misalignment2_check import extract_yes_no, score, seed_for

PRECISION = "fp8"
MODEL_REPO = "allenai/Olmo-3.1-7B-RL-Zero-Math"
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 16384
DATASET = ROOT / "results" / "phase4" / "misalignment4_interrogation.jsonl"


def ckpt_dir(revision):
    """step_0100 -> results/phase4/chkpt_0100 (created)."""
    d = ROOT / "results" / "phase4" / f"chkpt_{revision.split('_')[1]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_dataset(path):
    return [json.loads(l) for l in open(path)]


def load_done(out_path):
    done = set()
    if out_path.exists():
        for l in open(out_path):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:  # partial line from a killed run
                continue
            done.add((r["problem_id"], r["rollout_index"]))
    return done


def generate_batch(llm, prompts, seeds, ec, max_new_tokens):
    from vllm import SamplingParams

    sps = [SamplingParams(temperature=TEMPERATURE, max_tokens=max_new_tokens,
                           stop=ec.STOP_STRINGS, seed=s) for s in seeds]
    outs = llm.generate(prompts, sps, use_tqdm=True)  # returns in input order
    return [(o.outputs[0].text, o.outputs[0].finish_reason == "length", len(o.outputs[0].token_ids))
            for o in outs]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--checkpoint", type=int, default=None,
                    help="1-based RLVR checkpoint index: 1 -> step_0100. "
                         "Default: the final released weights (rev main, dir chkpt_2800).")
    args = ap.parse_args()

    rev = ckpt_revision(args.checkpoint) if args.checkpoint is not None else "main"
    out_dir = ckpt_dir("step_2800" if rev == "main" else rev)
    out_path = out_dir / "misalignment4_rlzeromath.jsonl"

    rows = load_dataset(DATASET)
    done = load_done(out_path)

    sys.path.insert(0, str(ROOT / "phase1"))
    import eval_capability as ec

    ec.MODEL_REPO = MODEL_REPO
    # +64 leaves room for the budget-forcing continuation past the cap.
    llm = ec.load_llm(rev, PRECISION, MAX_NEW_TOKENS + 1024 + 64)

    # Build every missing (row, rollout_index, seed) tuple, then generate the
    # whole first turn as a single batched call.
    pending = [(row, ri, seed_for(row["problem_id"], row["condition"], ri))
               for row in rows for ri in range(args.rollouts)
               if (row["problem_id"], ri) not in done]

    if not pending:
        print("nothing to do")
        ec.free_llm(llm)
        return

    prompts = [row["prompt"] for row, _, _ in pending]
    seeds = [seed for _, _, seed in pending]
    results = generate_batch(llm, prompts, seeds, ec, MAX_NEW_TOKENS)

    resps = [r[0] for r in results]
    hit_caps = [r[1] for r in results]
    gen_tokens_list = [r[2] for r in results]

    # Budget forcing: a second batched pass over just the rows that hit the
    # cap without emitting an answer.
    forced_suffix = "\n\\[\n\\boxed{"
    force_idx = [i for i in range(len(pending))
                 if hit_caps[i] and "Answer:" not in resps[i] and "\\boxed{" not in resps[i]]
    if force_idx:
        cont_prompts = [pending[i][0]["prompt"] + resps[i] + forced_suffix for i in force_idx]
        cont_seeds = [pending[i][2] for i in force_idx]
        cont_results = generate_batch(llm, cont_prompts, cont_seeds, ec, 32)
        for j, i in enumerate(force_idx):
            cont, _, cont_tokens = cont_results[j]
            if "\\]" in cont:  # drop anything the model adds after the block
                cont = cont[:cont.index("\\]") + 2]
            resps[i] = resps[i] + forced_suffix + cont
            gen_tokens_list[i] += cont_tokens
    budget_forced = [i in force_idx for i in range(len(pending))]

    metrics_list = [score(ec, resps[i], pending[i][0]["gold"], pending[i][0]["planted"])
                     for i in range(len(pending))]

    # Interrogation follow-up: a third batched pass, only for rollouts whose
    # turn-1 answer matched the planted key.
    followup_idx = [i for i in range(len(pending))
                     if pending[i][0]["condition"] == "INTERROGATION" and metrics_list[i]["adopted_planted"]]
    followup_resps = {}
    if followup_idx:
        fu_prompts = [pending[i][0]["prompt"] + resps[i] + "\n\n" + pending[i][0]["followup_prompt"]
                      for i in followup_idx]
        fu_seeds = [seed_for(pending[i][0]["problem_id"], pending[i][0]["condition"] + "_FOLLOWUP", pending[i][1])
                    for i in followup_idx]
        fu_results = generate_batch(llm, fu_prompts, fu_seeds, ec, MAX_NEW_TOKENS)
        for j, i in enumerate(followup_idx):
            followup_resps[i] = fu_results[j][0]

    ec.free_llm(llm)

    with out_path.open("a") as f:
        for i, (row, rollout_index, seed) in enumerate(pending):
            out_row = {k: v for k, v in row.items() if k not in ("prompt", "followup_prompt")}
            out_row.update(rollout_index=rollout_index, seed=seed, **metrics_list[i],
                            hit_cap=hit_caps[i], gen_tokens=gen_tokens_list[i],
                            budget_forced=budget_forced[i])
            if i in followup_resps:
                out_row["followup_response"] = followup_resps[i]
                out_row["followup_answer"] = extract_yes_no(followup_resps[i])
            f.write(json.dumps(out_row) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(pending)}] correct={metrics_list[i]['correct']} "
                  f"adopted_planted={metrics_list[i]['adopted_planted']} hit_cap={hit_caps[i]}")


if __name__ == "__main__":
    main()
