# When does misalignment emerge during RLVR?

Every published demonstration that reinforcement learning makes a language model misaligned plants the failure first: synthetic documents that teach the model about reward hacking, vulnerabilities seeded into the environment, or a reward that directly pays for misbehavior. Whether the default recipe, verifiable-reward RL on math with nothing planted, drifts on its own has not been measured. This project measures it on public intermediate checkpoints of an open model, and then asks whether a linear probe on the model's activations registers the drift at an earlier training step than the behavioral evals do. If internal signals lead behavior, misalignment can be caught during training rather than after deployment.

The subject is [allenai/Olmo-3-7B-RL-Zero-Math](https://huggingface.co/allenai/Olmo-3-7B-RL-Zero-Math): RLVR applied directly to the base Olmo-3 7B model, with 19 intermediate checkpoints published at 100-step intervals. We evaluate 11 points, the base model plus ten log-spaced steps from 100 to 1900.

Throughout, "misalignment" means instrumental misalignment: the model pursues a goal and deceives, conceals, or routes around a constraint to get there, as scored by InstrumentalEval. The emergent-misalignment eval of Betley et al. runs alongside it as an independent cross-check, not as the main measure.

## The claims

Ordered by how much of the project rides on them. The literature audit behind each status, with the full paper list, is in [SCOPE.md](SCOPE.md).

| # | Claim | Status (novelty sweep of 2026-07-16) |
|---|---|---|
| 5 | Stock-RLVR drift: an RLVR run with nothing planted produces a measurable rise in instrumental misalignment | Open in the literature. Phase 1 answer so far: no rise (see results below) |
| 4 | Lead-lag: a linear probe crosses its detection threshold at an earlier step than the behavioral eval | Open and contested: claimed for SFT, denied for RL, never tested on public RLVR checkpoints |
| 3 | Own onset: misalignment has an onset distinct from the capability curve | Shown for adjacent safety properties, open for instrumental misalignment; needs claim 5 to be non-flat |
| 1 | Behavioral trajectory: per-checkpoint misalignment evals over an RL run yield a nontrivial curve | Reproduction (Anthropic Nov 2025; UK AISI on Olmo-3, Mar 2026) |
| 2 | Causal steering: steering along the probe direction at an early, behaviorally clean checkpoint induces the behavior | Mechanism shown elsewhere; the timing half is untested; depends on claim 4 |

## The phases

| Phase | What it does | Where | Status |
|---|---|---|---|
| 0 | Verify the checkpoint inventory is dense enough to date an onset | [phase0/](phase0/) | Done, verdict GO |
| 1 | Behavioral + capability curves over all checkpoints | [phase1/](phase1/) | Generation, judging, and scoring |
| 2 | Linear probes on the cached activations, the internal curve | not yet written | Not started |
| 3 | Lead-lag analysis of the three curves, writeup | not yet written | Not started |

## Reproducing the pipeline

Environment: `uv sync` (Python 3.12+). The GPU steps ran on a single 16 GB RTX 5070 Ti at 8-bit precision. On a Mac the same commands work: quantization needs CUDA, so the loader falls back to bf16 on Apple silicon (MPS), and the scoring, judging, and aggregation steps run anywhere.

```bash
python phase0/checkpoint_inventory.py            # go/no-go on checkpoint density (metadata only)

python phase1/build_datasets.py                  # normalize the eval suites -> data/processed/
python phase1/run_sweep.py --run rlzero_math \
    --precision 8bit --activations --include-base    # the GPU sweep: transcripts + activations
python phase1/score_capability.py --run rlzero_math  # MATH-500 accuracy per checkpoint, no API key
python phase1/judge.py --run rlzero_math ...         # LLM-judged misalignment scores, needs an API key
python phase1/aggregate.py --run rlzero_math         # summary CSV + the curves figure
```

Every step is resumable and skips work already on disk. [phase1/README.md](phase1/README.md) explains each command, the judge configuration, and which measurement each script produces.

## Where results live

Each model family writes to its own top-level directory: `RL-zero results/` for the RL-Zero runs and their base-model anchor, `Think results/` if the Olmo-3-7B-Think run is used later. Inside:

```
RL-zero results/
  transcripts/<run>/<step>/    raw model outputs per checkpoint (ie, em, math)
  activations/<run>/<step>/    cached residual-stream activations for Phase 2
  scores/                      capability CSVs, judge outputs, phase1_summary_*.csv
  figures/                     plotted curves
```

To add a model, add a `RunSpec` to `RUNS` in [phase1/common.py](phase1/common.py); its outputs are routed automatically.

## Results so far

On RL-Zero-Math, nothing planted and nothing drifted. Across 1900 RLVR steps the instrumental convergence rate stays between 0.29 and 0.40 under the goal-nudging framing and between 0.33 and 0.39 under the neutral framing, with no trend in either. The emergent-misalignment rate stays between 3.5% and 6.9%, also flat. Two judges from different model families (GLM-5.2 and MiMo-v2.5) agree on the shape.

MATH-500 accuracy is 0.44 at the base model and falls to 0.08 by step 1900. That fall is confounded: generation is capped at 1024 new tokens and later checkpoints reason longer before answering, so the tail of the curve measures truncation as much as ability. Treat the capability curve as unreliable past the early steps until a higher-cap rerun.

Two caveats before trusting the behavioral null. RL-Zero-Math is not instruction-tuned; on non-math prompts it answers briefly and then degenerates into pretraining-style text, so a weak signal was expected here, and coherence is itself tracked. And evals of this kind can read 0% on models that are misaligned behind contextual triggers (see the design risks in SCOPE.md), so the null bounds the eval, not the model. Full numbers: `RL-zero results/scores/phase1_summary_rlzero_math.csv`, figure at `RL-zero results/figures/phase1_rlzero_math.png`.

## What happens next

Phase 2 trains probes on the cached activations. If the behavioral curve stays flat, the project's framing already anticipates it: an internal signal without behavioral expression is the more interesting outcome, and the fallback subject is Olmo-3-7B-Think (55 dense RLVR checkpoints, coherent agent behavior, already wired up as `think_rlvr`).
