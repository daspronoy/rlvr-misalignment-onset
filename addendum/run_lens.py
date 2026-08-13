"""Apply a fitted Jacobian lens to saved phase-2 transcripts, teacher-forced.

For each rollout, teacher-forces the concatenated (prompt + transcript) text
through the model and records per-layer lens readouts (top-k logits, and
logits/ranks for the planted/gold answer tokens) at strided token positions,
plus the model's own final-layer readout for comparison.

lens.apply returns dense logits for every requested position at once, and
[n_layers, n_positions, vocab] is too big to hold for a full transcript, so
positions are read out in chunks of --pos-chunk, each chunk costing its own
forward pass (jlens has no incremental/cached apply).

Usage:
  python phase3/run_lens.py --input results/phase3/infected_prompts.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase1"))
from download_model import MODEL_REPO, read_hf_token  # noqa: E402  (sets HF_HOME)

DATASET_FILE = ROOT / "results" / "phase2" / "misalignment2_dataset.jsonl"


def load_prompt_map(path: Path) -> dict[tuple[int, str], str]:
    prompt_map = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        prompt_map[(row["problem_id"], row["condition"])] = row["prompt"]
    return prompt_map


def target_token_ids(tokenizer, s: str, other: str) -> list[int]:
    """Token ids that identify s against other (e.g. planted "-34" vs gold "-41"
    share the first token "-"), preferring tokens not in other's tokenization;
    fall back to first tokens when everything is shared."""
    own, other_set, firsts = [], set(), []
    for variant in (s, " " + s):
        toks = tokenizer(variant, add_special_tokens=False)["input_ids"]
        own.extend(t for t in toks if t not in own)
        if toks:
            firsts.append(toks[0])
    for variant in (other, " " + other):
        other_set.update(tokenizer(variant, add_special_tokens=False)["input_ids"])
    distinct = [t for t in own if t not in other_set]
    return distinct or list(dict.fromkeys(firsts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="jsonl of rollouts")
    ap.add_argument("--lens", type=Path, default=ROOT / "results" / "phase3" / "lens" / "jacobian_lens_n100.pt")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument("--pos-chunk", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="process only first N rows, for smoke tests")
    ap.add_argument("--model", default=MODEL_REPO)
    ap.add_argument("--revision", default="main", help="final checkpoint")
    args = ap.parse_args()

    out_dir = args.out_dir or ROOT / "results" / "phase3" / "readouts" / args.input.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_map = load_prompt_map(DATASET_FILE)
    rows = [json.loads(l) for l in args.input.read_text().splitlines()]
    if args.limit is not None:
        rows = rows[: args.limit]

    todo = []
    for row in rows:
        out_file = out_dir / f"{row['problem_id']}_{row['condition']}_{row['rollout_index']}.pt"
        if not out_file.exists():
            todo.append((row, out_file))
    print(f"[plan] {len(rows)} rows, {len(todo)} to process")
    if not todo:
        return

    import torch
    import transformers
    import jlens

    jlens.configure_logging()
    from torchao.quantization import Float8WeightOnlyConfig

    token = read_hf_token()
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, token=token,
        device_map="cuda",
        quantization_config=transformers.TorchAoConfig(Float8WeightOnlyConfig()),
    )
    hf_model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, revision=args.revision, token=token)
    model = jlens.from_hf(hf_model, tokenizer)

    lens = jlens.JacobianLens.load(str(args.lens))

    index_file = out_dir / "index.jsonl"
    with index_file.open("a") as index_f:
        for n, (row, out_file) in enumerate(todo, start=1):
            t0 = time.monotonic()
            full_text = prompt_map[(row["problem_id"], row["condition"])] + row["transcript"]
            seq_len = len(tokenizer(full_text, truncation=True, max_length=args.max_seq_len)["input_ids"])
            positions = list(range(0, seq_len, args.stride))
            if positions[-1] != seq_len - 1:
                positions.append(seq_len - 1)

            target_ids = {
                "planted": target_token_ids(tokenizer, row["planted"], row["gold"]),
                "gold": target_token_ids(tokenizer, row["gold"], row["planted"]),
            }
            all_target_ids = target_ids["planted"] + target_ids["gold"]

            layers = None
            topk_ids_chunks, topk_scores_chunks = [], []
            target_logits_chunks, target_ranks_chunks = [], []
            model_topk_ids_chunks, model_topk_scores_chunks = [], []

            with torch.no_grad():
                for i in range(0, len(positions), args.pos_chunk):
                    chunk_pos = positions[i : i + args.pos_chunk]
                    lens_logits, model_logits, _ = lens.apply(
                        model, full_text, positions=chunk_pos, max_seq_len=args.max_seq_len,
                    )
                    if layers is None:
                        layers = sorted(lens_logits.keys())

                    layer_topk_ids, layer_topk_scores = [], []
                    layer_target_logits, layer_target_ranks = [], []
                    for l in layers:
                        logits = lens_logits[l].float()  # [n_pos, vocab]
                        scores, ids = torch.topk(logits, args.topk, dim=-1)
                        layer_topk_ids.append(ids)
                        layer_topk_scores.append(scores)

                        t_logits = logits[:, all_target_ids]  # [n_pos, T]
                        ranks = (logits.unsqueeze(-1) > t_logits.unsqueeze(1)).sum(dim=1)  # [n_pos, T]
                        layer_target_logits.append(t_logits)
                        layer_target_ranks.append(ranks)

                    topk_ids_chunks.append(torch.stack(layer_topk_ids))  # [L, n_pos, topk]
                    topk_scores_chunks.append(torch.stack(layer_topk_scores))
                    target_logits_chunks.append(torch.stack(layer_target_logits))  # [L, n_pos, T]
                    target_ranks_chunks.append(torch.stack(layer_target_ranks))

                    m_logits = model_logits.float()  # [n_pos, vocab]
                    m_scores, m_ids = torch.topk(m_logits, args.topk, dim=-1)
                    model_topk_ids_chunks.append(m_ids)
                    model_topk_scores_chunks.append(m_scores)

            topk_ids = torch.cat(topk_ids_chunks, dim=1).to(torch.int32)
            topk_scores = torch.cat(topk_scores_chunks, dim=1).to(torch.float16)
            target_logits = torch.cat(target_logits_chunks, dim=1).to(torch.float16)
            target_ranks = torch.cat(target_ranks_chunks, dim=1).to(torch.int32)
            model_topk_ids = torch.cat(model_topk_ids_chunks, dim=0).to(torch.int32)
            model_topk_scores = torch.cat(model_topk_scores_chunks, dim=0).to(torch.float16)

            meta = {
                "problem_id": row["problem_id"],
                "condition": row["condition"],
                "rollout_index": row["rollout_index"],
                "adopted_planted": row["adopted_planted"],
                "correct": row["correct"],
                "planted": row["planted"],
                "gold": row["gold"],
                "seq_len": seq_len,
                "target_ids": target_ids,
            }
            torch.save({
                "positions": torch.tensor(positions, dtype=torch.int64),
                "layers": layers,
                "topk_ids": topk_ids,
                "topk_scores": topk_scores,
                "target_logits": target_logits,
                "target_ranks": target_ranks,
                "model_topk_ids": model_topk_ids,
                "model_topk_scores": model_topk_scores,
                "meta": meta,
            }, out_file)
            index_f.write(json.dumps({**meta, "file": out_file.name}) + "\n")
            index_f.flush()

            dt = time.monotonic() - t0
            print(f"[row {n}/{len(todo)}] {row['problem_id']}_{row['condition']}_{row['rollout_index']} "
                  f"seq_len={seq_len} n_pos={len(positions)} took {dt:.1f}s")


if __name__ == "__main__":
    main()
