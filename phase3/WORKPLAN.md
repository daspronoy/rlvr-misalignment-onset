# Phase 3 workplan — J-lens on the final RL-Zero-Math checkpoint

Goal: read the internal workspace of the final checkpoint on the prompts where
phase 2 found misalignment. Does the planted key surface in the J-space before
the output commits to it, and does anything deception-shaped show up in the
interrogation-lie rollouts?

## What J-lens needs

The Jacobian lens (Anthropic global-workspace method) finds, per vocabulary
token, the internal activity pattern that makes the model more likely to emit
that token later. It needs full weights, per-layer activations on real prompts,
and produces per-layer token readouts ("what's in the workspace now"). Known
limits: single-token concepts only; directions are derived from the weights, so
they are checkpoint-specific.

## Steps

1. **Port the implementation.** Pull the open-source J-lens repo (linked from
   the Anthropic global-workspace post), adapt to Olmo-3.1-7B: tokenizer /
   unembedding hookup, layer hook points, HF `step_*` revision loading (reuse
   `phase1/download_model.py` cache).
   → verify: readouts on a trivial prompt ("The capital of France is") show
   the expected token surfacing across layers.

2. **Build the probe set from phase 2.** From
   `results/phase2/misalignment2_rlzeromath.jsonl` + `judge2_opus_full.jsonl`:
   - all key-adoption rollouts (~40 across conditions), the 8 interrogation
     lies, and matched clean rollouts on the same problems as controls.
   - target tokens: the planted key's tokens, the correct answer's tokens,
     and a small deception-adjacent token list (single-token only).
   → verify: a jsonl of (prompt, condition, label, target-token ids).

3. **Run on the final checkpoint.** Teacher-force each phase-2 transcript,
   record per-layer J-space readouts at each position (or a strided subset —
   these transcripts run to 16k tokens; full per-position readouts won't fit,
   so stride + dense windows around the key-adoption point).
   → artifacts: `results/phase3/readouts/<condition>.jsonl` + summary figure.
   VRAM note: 7B + Jacobian work may exceed the local 16 GB card; plan for
   the vast.ai 4090 (or A6000) if it does.

4. **Analysis.** Two comparisons:
   - key-adoption rollouts vs clean controls: does the planted-key token
     enter the workspace earlier / stronger / at lower layers when the model
     ends up adopting it?
   - interrogation lies: during the denial turn, is the key token still live
     in the workspace while the output says "I did not use it"?
   → verify: per-condition effect sizes + example visualizations.

5. **(feeds claim 2, later)** Intervention: suppress the planted-key pattern
   in-workspace and measure whether the adoption rate drops. Only after step 4
   shows a signal.

## All checkpoints, or final + translate?

Decision: **final checkpoint only for phase 3 proper**, with a cheap
translation-validity check before any sweep.

- The lens directions come from the weights, so each checkpoint formally has
  its own lens. But the RLVR linear-dynamics result (arXiv:2601.04537,
  `knowledge/`) shows weights AND outputs evolve near-linearly over RLVR
  (R² > 0.7), so final-checkpoint directions plausibly transfer backwards.
- Translation only saves the direction-finding step. The forward passes /
  activations must be recomputed per checkpoint regardless — each checkpoint
  is a different model.
- Caveat that decides it: for the lead-lag claim (claim 4), reading early
  checkpoints through the final model's lens is circular — it can import the
  final model's structure into checkpoints that don't yet have it. Any onset
  dated that way is suspect.

So, if step 4 finds a signal and we want the trajectory:

6. **Anchor check.** Compute the lens natively at 3 anchors (step 100,
   ~1500, final). Compare native vs final-translated directions (cosine
   similarity, readout agreement on the probe set).
   - High agreement → translate through the remaining checkpoints, but
     confirm any claimed onset step with a native lens at that step.
   - Low agreement → per-checkpoint lens; budget for the vast.ai sweep.
