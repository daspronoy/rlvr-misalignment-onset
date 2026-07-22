# When does misalignment emerge during RLVR?

Every published demonstration that reinforcement learning makes a language model misaligned plants the failure first: synthetic documents that teach the model about reward hacking, vulnerabilities seeded into the environment, or a reward that directly pays for misbehavior. Whether the default recipe, verifiable-reward RL on math with nothing planted, drifts on its own has not been measured. This project measures it on public intermediate checkpoints of an open model, and then asks whether a linear probe on the model's activations registers the drift at an earlier training step than the behavioral evals do. If internal signals lead behavior, misalignment can be caught during training rather than after deployment.

The subject is [allenai/Olmo-3.1-7B-RL-Zero-Math](https://huggingface.co/allenai/Olmo-3.1-7B-RL-Zero-Math): RLVR applied directly to the base Olmo-3.1 7B model, with intermediate checkpoints published as `step_*` branches across training. The capability sweep evaluates 11 of them, steps 100 to 2800 sampled every 300, plus the shipped `main` model.

Throughout, "misalignment" means instrumental misalignment: the model pursues a goal and deceives, conceals, or routes around a constraint to get there, as scored by InstrumentalEval. The emergent-misalignment eval of Betley et al. runs alongside it as an independent cross-check, not as the main measure.

## The claims

Ordered by how much of the project rides on them. The literature audit behind each status, with the full paper list, is in [SCOPE.md](SCOPE.md).

| # | Claim | Status (novelty sweep of 2026-07-16) |
|---|---|---|
| 5 | Stock-RLVR drift: an RLVR run with nothing planted produces a measurable rise in instrumental misalignment | Open in the literature. Not yet measured in the current pipeline; a prior run on the 3.0 release was flat (see results below) |
| 4 | Lead-lag: a linear probe crosses its detection threshold at an earlier step than the behavioral eval | Open and contested: claimed for SFT, denied for RL, never tested on public RLVR checkpoints |
| 3 | Own onset: misalignment has an onset distinct from the capability curve | Shown for adjacent safety properties, open for instrumental misalignment; needs claim 5 to be non-flat |
| 1 | Behavioral trajectory: per-checkpoint misalignment evals over an RL run yield a nontrivial curve | Reproduction (Anthropic Nov 2025; UK AISI on Olmo-3, Mar 2026) |
| 2 | Causal steering: steering along the probe direction at an early, behaviorally clean checkpoint induces the behavior | Mechanism shown elsewhere; the timing half is untested; depends on claim 4 |

## The phases

| Phase | What it does | Where | Status |
|---|---|---|---|
| 0 | Verify the checkpoint inventory is dense enough to date an onset | [phase0/](phase0/) | Done, verdict GO |
| 1 | Behavioral + capability curves over all checkpoints | [phase1/](phase1/) | Capability curve done; behavioral eval not yet in this codebase |
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

The capability sweep writes under `results/phase1/capability/`:

```
results/phase1/capability/
  <revision>.jsonl    transcript + extracted answer + per-problem correctness
  summary.csv         accuracy per checkpoint
  capability.png/pdf  the plotted curve
```

## Results so far

The behavioral evals are not part of the current pipeline. The null reported here predates the codebase overhaul: it was measured on the earlier Olmo-3-7B release and has not been rerun on 3.1 or migrated into `results/phase1/`. As it stood then, nothing planted and nothing drifted. Across 1900 RLVR steps the instrumental convergence rate stayed between 0.29 and 0.40 under the goal-nudging framing and between 0.33 and 0.39 under the neutral framing, with no trend in either. The emergent-misalignment rate stayed between 3.5% and 6.9%, also flat. Two judges from different model families (GLM-5.2 and MiMo-v2.5) agreed on the shape.

The MATH-500 curve is now the higher-cap rerun that the truncated version called for, generated at a 16384-token cap. Raw accuracy climbs from 0.63 at step 100 to a 0.77 peak near step 2200, then settles at 0.72 for the shipped model. But the rise is an artifact of answer formatting. Scoring marks any response that never emits an `Answer:` or `\boxed{}` as wrong, and every one of those is a response that ran into the token cap mid-thought without committing an answer, the RL-Zero rambling pattern. Set them aside and accuracy on the problems the model actually finishes holds flat between 0.87 and 0.89 across the whole run. What training moves is the truncation rate: 28% of responses run off the end at step 100, down to 14% by step 2200. RLVR is teaching the model to stop rambling and write the answer down, not to solve more of what it attempts. Numbers in `results/phase1/capability/summary.csv`, figure at `results/phase1/capability/capability.png`.

