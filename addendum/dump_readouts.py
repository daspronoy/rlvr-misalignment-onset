"""Dump every readout .pt in a directory to human-readable per-layer text files.

For each <name>.pt, writes <dir>_decoded/<name>/meta.txt and layer_NN.txt, where
each layer file has one line per saved position: planted/gold logit+rank and the
decoded lens top-10.

Usage:
  python addendum/dump_readouts.py results/addendum/readouts/infected_prompts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase1"))
from download_model import MODEL_REPO, read_hf_token  # noqa: E402  (sets HF_HOME)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir", type=Path)
    args = ap.parse_args()

    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_REPO, token=read_hf_token())

    out_root = args.dir.parent / f"{args.dir.name}_decoded"
    for pt in sorted(args.dir.glob("*.pt")):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        out = out_root / pt.stem
        out.mkdir(parents=True, exist_ok=True)
        positions = d["positions"].tolist()
        (out / "meta.txt").write_text(json.dumps(d["meta"], indent=2) + "\n")
        # target columns: planted tokens first, then gold (may be multi-token)
        tids = d["meta"]["target_ids"]
        target_labels = [f"{kind}{tokenizer.decode([t])!r}"
                         for kind in ("planted", "gold") for t in tids[kind]]

        model_lines = [
            f"pos {pos:6d}  top10: " + " ".join(repr(tokenizer.decode([t])) for t in d["model_topk_ids"][p].tolist())
            for p, pos in enumerate(positions)
        ]
        (out / "model_final.txt").write_text("\n".join(model_lines) + "\n")

        for i, layer in enumerate(d["layers"]):
            lines = []
            for p, pos in enumerate(positions):
                targets = "  ".join(
                    f"{lbl} {lg:7.2f} r{rk:<6d}"
                    for lbl, lg, rk in zip(target_labels,
                                           d["target_logits"][i, p].tolist(),
                                           d["target_ranks"][i, p].tolist()))
                top = " ".join(repr(tokenizer.decode([t])) for t in d["topk_ids"][i, p].tolist())
                lines.append(f"pos {pos:6d}  {targets}  top10: {top}")
            (out / f"layer_{layer:02d}.txt").write_text("\n".join(lines) + "\n")
        print(f"[done] {out} ({len(d['layers'])} layers x {len(positions)} positions)")


if __name__ == "__main__":
    main()
