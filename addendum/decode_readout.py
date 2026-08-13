"""Render a readout .pt (from run_lens.py) as human-readable layer-by-layer text.

For one readout position, prints the lens top-10 tokens at every layer plus the
logit and rank of the planted/gold targets, then the model's own final top-10.

Usage:
  python addendum/decode_readout.py results/addendum/readouts/infected_prompts/101_INTERROGATION_1.pt
  python addendum/decode_readout.py <file.pt> --pos -1   # index into saved positions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase1"))
from download_model import MODEL_REPO, read_hf_token  # noqa: E402  (sets HF_HOME)


def toks(tokenizer, ids) -> str:
    return " ".join(repr(tokenizer.decode([i])) for i in ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", type=Path)
    ap.add_argument("--pos", type=int, default=-1, help="index into saved positions")
    args = ap.parse_args()

    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_REPO, token=read_hf_token())

    d = torch.load(args.file, map_location="cpu", weights_only=False)
    meta = d["meta"]
    p = args.pos % len(d["positions"])
    print(f"{args.file.name}: problem {meta['problem_id']} {meta['condition']} "
          f"rollout {meta['rollout_index']}, planted={meta['planted']!r} gold={meta['gold']!r}, "
          f"adopted_planted={meta['adopted_planted']}")
    print(f"position {d['positions'][p].item()} (readout index {p} of {len(d['positions'])}), "
          f"seq_len {meta['seq_len']}\n")

    for i, layer in enumerate(d["layers"]):
        pl, gl = d["target_logits"][i, p].tolist()
        pr, gr = d["target_ranks"][i, p].tolist()
        print(f"layer {layer:2d}  planted logit {pl:7.2f} rank {pr:5d} | "
              f"gold logit {gl:7.2f} rank {gr:5d}")
        print(f"          top10: {toks(tokenizer, d['topk_ids'][i, p].tolist())}")
    print(f"\nmodel     top10: {toks(tokenizer, d['model_topk_ids'][p].tolist())}")


if __name__ == "__main__":
    main()
