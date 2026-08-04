"""Fit a Jacobian lens on the final RL-Zero-Math checkpoint using Dolma 3 Mix prompts.

Resumable at two levels:
  - prompts are streamed from Dolma 3 once and cached to <out>/prompts.jsonl,
    so reruns see the identical prompt list;
  - fitting runs in slices of --slice-size prompts, each saved to
    <out>/slices/slice_NNNN.pt. Existing slice files are skipped on rerun,
    and jlens's own checkpoint_path resumes a killed mid-slice fit.
Slices are merged into <out>/jacobian_lens_n<N>.pt; the per-slice relative
change of the cumulative lens is printed as the saturation diagnostic.

Usage:
  python phase3/fit_lens.py --n-prompts 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase1"))
from download_model import MODEL_REPO, read_hf_token  # noqa: E402  (sets HF_HOME)

DOLMA_REPO = "allenai/dolma3_mix-6T-1025-7B"


def load_prompts(out_dir: Path, n_prompts: int, token: str | None) -> list[str]:
    """First n_prompts Dolma docs with enough text, cached for reproducible resume."""
    cache = out_dir / "prompts.jsonl"
    prompts = [json.loads(l)["text"] for l in cache.read_text().splitlines()] if cache.exists() else []
    if len(prompts) < n_prompts:
        from datasets import load_dataset
        ds = load_dataset(DOLMA_REPO, split="train", streaming=True, token=token)
        with cache.open("a") as f:
            for doc in ds.skip(len(prompts)):
                text = doc["text"]
                if len(text) < 1000:  # ensure >= 128 tokens; fit truncates via max_seq_len
                    continue
                f.write(json.dumps({"text": text}) + "\n")
                prompts.append(text)
                if len(prompts) >= n_prompts:
                    break
    return prompts[:n_prompts]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--slice-size", type=int, default=10)
    ap.add_argument("--dim-batch", type=int, default=32, help="memory knob: backward passes per prompt = ceil(d_model/dim_batch)")
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--model", default=MODEL_REPO)
    ap.add_argument("--revision", default="main", help="final checkpoint")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "phase3" / "lens")
    args = ap.parse_args()

    slices_dir = args.out / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    token = read_hf_token()
    prompts = load_prompts(args.out, args.n_prompts, token)

    n_slices = (len(prompts) + args.slice_size - 1) // args.slice_size
    todo = [i for i in range(n_slices) if not (slices_dir / f"slice_{i:04d}.pt").exists()]
    print(f"[plan] {len(prompts)} prompts, {n_slices} slices, {len(todo)} to fit")

    if todo:
        import torch
        import transformers
        import jlens

        jlens.configure_logging()
        # fp8 weight-only via torchao: halves weight memory (~7 GB, fits the 16 GB
        # card) and dequantizes to bf16 in compute, so backward still works.
        from torchao.quantization import Float8WeightOnlyConfig

        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=torch.bfloat16, token=token,
            quantization_config=transformers.TorchAoConfig(Float8WeightOnlyConfig()),
        ).cuda()
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, revision=args.revision, token=token)
        model = jlens.from_hf(hf_model, tokenizer)

        for i in todo:
            chunk = prompts[i * args.slice_size : (i + 1) * args.slice_size]
            print(f"[slice {i + 1}/{n_slices}] fitting {len(chunk)} prompts")
            lens = jlens.fit(
                model, chunk,
                dim_batch=args.dim_batch,
                max_seq_len=args.max_seq_len,
                checkpoint_path=str(slices_dir / f"ckpt_{i:04d}.pt"),
            )
            lens.save(str(slices_dir / f"slice_{i:04d}.pt"))
            (slices_dir / f"ckpt_{i:04d}.pt").unlink(missing_ok=True)

    import jlens

    import torch

    slices = [jlens.JacobianLens.load(str(slices_dir / f"slice_{i:04d}.pt")) for i in range(n_slices)]
    merged = slices[0]
    saturation = args.out / "saturation.csv"
    with saturation.open("w") as f:
        f.write("slice,n_prompts,rel_dj\n")
        for i, s in enumerate(slices[1:], start=2):
            prev = merged
            merged = jlens.JacobianLens.merge([prev, s])
            # saturation diagnostic: relative Frobenius change of the cumulative lens
            num = sum(torch.linalg.matrix_norm(merged.jacobians[l].float() - prev.jacobians[l].float()).item()
                      for l in merged.jacobians)
            den = sum(torch.linalg.matrix_norm(j.float()).item() for j in merged.jacobians.values())
            f.write(f"{i},{min(i * args.slice_size, len(prompts))},{num / den:.6f}\n")
            print(f"[saturation] after slice {i}: rel dJ = {num / den:.4f}")
    print(f"[saturation] written to {saturation}")

    out_file = args.out / f"jacobian_lens_n{len(prompts)}.pt"
    merged.save(str(out_file))
    print(f"[done] {out_file}")


if __name__ == "__main__":
    main()
