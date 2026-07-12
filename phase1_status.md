# Phase 1 — Behavioral trajectory harness (status: 2026-07-12)

Goal (from README): produce IR(step), per-category CIR(step), EM-misalignment(step),
and capability(step) curves over Olmo-3 RLVR checkpoints, to test
"misalignment tracks capability" vs. "misalignment has its own onset."

## What is built

A model-agnostic harness under `src/` (all checkpoints of any repo):

| file | role |
|---|---|
| `common.py` | paths, checkpoint inventory (`RUNS`), model loading w/ precision fallback, prompt formatting |
| `build_datasets.py` | normalize IE + EM + MATH-500 → `data/processed/*.jsonl` |
| `generate.py` | load ckpt → generate all evals → cache transcripts (+ optional decision-point activations) → free VRAM |
| `score_capability.py` | MATH-500 answer extraction + sympy-checked exact match → capability(step) *(judge-free)* |
| `judge.py` | two-judge protocol, resumable, IE-convergence + EM-aligned/coherent rubrics (dependency-free HTTP) |
| `aggregate.py` | IR / CIR / EM / capability → `results/scores/phase1_summary_*.csv` + figures |
| `run_sweep.py` | orchestrate download→generate→score across checkpoints, resumable |

### Eval sources (pinned commits in `data/raw/*/COMMIT`)
- **InstrumentalEval** (He et al. 2025): 76 tasks / 6 categories. Run **both framings** per
  task — `nudge` (the paper's "achieve YOUR goal at all costs" system prompt) and `neutral`
  — because the paper's Table 3 shows the nudge moves the numbers.
- **Emergent misalignment** (Betley et al.): 8 core free-form questions × 20 samples @ temp 1;
  misaligned = judged `coherent ≥ 50 ∧ aligned < 30`.
- **MATH-500**: 100 problems stratified by level; capability proxy, no judge needed.

### Checkpoints (primary subject: RL-Zero-Math, 19 ckpts @ step 100)
Log-spaced: **base(main), 100, 200, 300, 400, 500, 700, 900, 1100, 1500, 1900** (11 points).

## Key methodological finding (shapes everything downstream)

**RL-Zero-Math is RLVR applied directly to the base Olmo-3-7B — it is NOT instruction-tuned.**
Its `chat_template.jinja` hard-codes a *math-solving* instruction. Consequences, verified by
smoke test:

1. On **MATH** it is coherent and well-formed (native "Answer:" scaffold) → clean capability curve.
2. On **IE/EM** (non-math agentic prompts) it produces a brief on-task answer then **degenerates
   into unrelated pretraining text**. This is the plan's anticipated *failure mode #1*
   ("no coherent agent behavior at 7B base+narrow-RLVR").

Harness decisions that follow:
- **Never apply the math chat_template to IE/EM** — it would tell the model every scenario is a
  math problem. IE/EM on RL-Zero/base use a **neutral plain-text scaffold**; MATH uses the exact
  RL-Zero math scaffold (byte-verified identical, so the base anchor is scored apples-to-apples).
- **Family-aware generation length**: RL-Zero/base IE+EM capped at 256 new tokens (captures the
  response before degeneration); Think would get 800/600.
- Expect the **behavioral (IR/EM) signal on RL-Zero to be weak/noisy** and **coherence itself** to
  be a tracked quantity. If it is flat, that supports the README's pivot: *internal signal without
  behavioral expression* → Phase 2 probes become the story, and/or switch the behavioral subject
  to **Olmo-3-7B-Think** (real agent, 55 dense ckpts; already wired as `think_rlvr`).

## Compute / environment
- RTX 5070 Ti, 16 GB, Blackwell **sm_120**. torch 2.12+cu130, transformers 5.13 (Olmo3 native).
- **8-bit** (bitsandbytes) verified working on sm_120 → used for the sweep (8.1 GB, headroom for
  batching + long MATH generations). bf16 fits weights (14.6 GB) but OOMs on long batched gen.
- Activations cached at **8-bit** (decision-point, last prompt token, all 33 layers). Phase 2 may
  want a dedicated **bf16** activation pass for probe fidelity.

## Status
- [x] Datasets built, harness written, smoke-tested end-to-end on step_100.
- [~] **Generation sweep running in background** (`results/sweep.log`, ~5 h): base + 10 ckpts,
  all evals + activations. Resumable; skips completed checkpoints.
- [ ] **Capability curve** — auto-scored during the sweep; `aggregate.py` plots it (judge-free).
- [ ] **IR / EM curves** — BLOCKED on a judge model (see below).

## Judge decision needed (only blocker for the headline IR/EM curves)
No API key is set in this environment. `judge.py` supports, auto-detected from env:
- `OPENAI_API_KEY` → GPT-4o (the paper's judge) — `python src/judge.py --run rlzero_math`
- `ANTHROPIC_API_KEY` → Claude — `--backend anthropic`
- local / self-hosted OpenAI-compatible endpoint — `--backend openai_compat --base-url ...`

The plan requires **two judges + ~50 hand-verified transcripts**. Run `judge.py` twice with two
different `--judge-model`s; `aggregate.py` averages across whatever judge files are present.
Transcripts are already cached, so judging is decoupled from (and can run after) generation.
