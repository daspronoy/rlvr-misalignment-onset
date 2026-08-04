# Phase 3 workplan — J-lens on the final RL-Zero-Math checkpoint

Goal: read the internal workspace of the final checkpoint on the prompts where phase 2 found misalignment. Does the planted key surface in the J-space before the output commits to it, and does anything deception-shaped show up in the interrogation-lie rollouts?

## What J-lens needs

The Jacobian lens (Anthropic global-workspace method) finds, per vocabulary token, the internal activity pattern that makes the model more likely to emit that token later. It needs full weights, per-layer activations on real prompts, and produces per-layer token readouts ("what's in the workspace now"). Known limits: single-token concepts only; directions are derived from the weights, so they are checkpoint-specific.

## Steps

1. **Port the implementation.** Pull the J-lens repo (github.com/anthropics/jacobian-lens), adapt to Olmo-3.1-7B: tokenizer / unembedding hookup, layer hook points, HF `step_*` revision loading (reuse `phase1/download_model.py` cache). fp8 checkpoint note: fitting needs backward through the model — verify the fp8 format supports autograd (torchao / HF fine-grained fp8 do; compressed-tensors/vLLM-style don't) on a 2–3 prompt smoke run before anything else. Fitting never touches the weights (it only measures the average Jacobian), so the checkpoint stays identical to the phase 1/2 one. → verify: `jlens.fit()` completes on 2–3 prompts; record wall-clock per prompt for the extrapolation below.

2. **Fit the lens.** Fitting corpus: the repo ships no fitting data ("pretraining-like corpus", 1000 × 128-token sequences in the paper; ~100 usable). Use the base model's actual pretraining mix — stream `allenai/dolma3_mix-6T-1025-7B` from HF, take ~1000 documents, one random 128-token window each (corpus is ~6T tokens; streaming means no download). Fit in 10-prompt slices and combine with `JacobianLens.merge()`, so cumulative lenses J(10), J(20), … come for free. Saturation diagnostic: plot per-layer relative Frobenius change ‖J(n)−J(n−10)‖_F/‖J(n)‖_F and top-k readout overlap between consecutive cumulative lenses on held-out activations vs n. Stop when both flatten (expected near ~100 prompts per the repo); otherwise keep adding slices — merge means earlier compute is never wasted. Time estimate on the 5070 Ti: ~12–18 h at 100 prompts, ~5–7 days at 1000 (cost is ~d_model chunked backward passes per prompt); confirm against the step-1 timing run. fp8 makes 16 GB VRAM sufficient (7 GB weights) but doesn't speed up fitting — backward runs in bf16. Caveat for the writeup: the lens is fit on the quantized model, an approximation of the full-precision Jacobian. Optional robustness check: a second lens fit on math-flavored text (Dolmino mid-training mix is 19% math) to show corpus choice doesn't drive the readouts. → artifacts: `results/phase3/jacobian_lens.pt`, per-slice checkpoints, saturation plot.

3. **Build the probe set from phase 2.** From `results/phase2/misalignment2_rlzeromath.jsonl` + `judge2_opus_full.jsonl`: all key-adoption rollouts (~40 across conditions), the 8 interrogation lies, and matched clean rollouts on the same problems as controls. Target tokens: the planted key's tokens, the correct answer's tokens, and a small deception-adjacent token list (single-token only). → verify: a jsonl of (prompt, condition, label, target-token ids).

4. **Run on the final checkpoint.** Teacher-force each phase-2 transcript, record per-layer J-space readouts at each position (or a strided subset — these transcripts run to 16k tokens; full per-position readouts won't fit, so stride + dense windows around the key-adoption point). → artifacts: `results/phase3/readouts/<condition>.jsonl` + summary figure. VRAM: fp8 weights (~7 GB) fit the local 16 GB card for readout (forward-only); fall back to the vast.ai 4090 only if fitting in step 2 is too slow locally.

5. **Analysis.** Two comparisons. Key-adoption rollouts vs clean controls: does the planted-key token enter the workspace earlier / stronger / at lower layers when the model ends up adopting it? Interrogation lies: during the denial turn, is the key token still live in the workspace while the output says "I did not use it"? → verify: per-condition effect sizes + example visualizations.

6. **(feeds claim 2, later)** Intervention: suppress the planted-key pattern in-workspace and measure whether the adoption rate drops. Only after step 5 shows a signal.

## All checkpoints, or final + translate?

Decision: **final checkpoint only for phase 3 proper**, with a cheap translation-validity check before any sweep.

- The lens directions come from the weights, so each checkpoint formally has its own lens. But the RLVR linear-dynamics result (arXiv:2601.04537, `knowledge/`) shows weights AND outputs evolve near-linearly over RLVR (R² > 0.7), so final-checkpoint directions plausibly transfer backwards.
- Translation only saves the direction-finding step. The forward passes / activations must be recomputed per checkpoint regardless — each checkpoint is a different model.
- Caveat that decides it: for the lead-lag claim (claim 4), reading early checkpoints through the final model's lens is circular — it can import the final model's structure into checkpoints that don't yet have it. Any onset dated that way is suspect.

So, if step 5 finds a signal and we want the trajectory:

7. **Anchor check.** Compute the lens natively at 3 anchors (step 100, ~1500, final). Compare native vs final-translated directions (cosine similarity, readout agreement on the probe set). High agreement → translate through the remaining checkpoints, but confirm any claimed onset step with a native lens at that step. Low agreement → per-checkpoint lens; budget for the vast.ai sweep.

## J-lens deployment

1. Collect the prompts — from results/phase2/, take the transcripts where misalignment actually emerged (the ~40 key adoptions and 8 interrogation lies), plus a few clean runs on the same problems so you know what "normal" looks like in the workspace.

2. Replay them through the final checkpoint — teacher-force each transcript, and at each position apply the lens to the layer activations. This gives you, per layer and per position, a ranked list of tokens currently "live" in the workspace — essentially a transcript of what the model is internally entertaining while it writes.

3. Look at what emerges. Two places to look: in the adoption runs, watch for the planted key appearing in the readouts before the output commits to it — and note at which layer and how many tokens ahead. In the lie runs, check whether the key is still live in the workspace during the denial turn while the surface text says "I didn't use it." The clean controls tell you whether any of this is just baseline noise.
