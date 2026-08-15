# RLVR, Self-Doubt and Deception

Every published demonstration that RL makes a language model misaligned plants the failure first — poisoned documents, seeded vulnerabilities, or a reward that pays for misbehavior. Whether the default recipe, verifiable-reward RL on math with nothing planted, drifts on its own is unmeasured. This project measures it on the public intermediate checkpoints (`step_*` branches) of [allenai/Olmo-3.1-7B-RL-Zero-Math](https://huggingface.co/allenai/Olmo-3.1-7B-RL-Zero-Math): capability and behavior are tracked across the training run, and the model's internals are read directly on the transcripts where misbehavior shows up.

Environment: `uv sync` (Python 3.12+). GPU steps ran on a single 16 GB RTX 5070 Ti with vLLM at fp8 precision. Every script is resumable and skips work already on disk; outputs land in `results/`, which mirrors the phase layout.

## Phase 0 — checkpoint inventory ([phase0/](phase0/))

A gate, not an experiment: dating an onset needs enough intermediate checkpoints, so this counts them before any GPU time is spent.

- [checkpoint_inventory.py](phase0/checkpoint_inventory.py) — queries the HF metadata API for `step_*` branches (downloads nothing) and prints per-repo counts, spacing, and a GO/MARGINAL/NO-GO verdict.

## Phase 1 — capability and efficiency sweep ([phase1/](phase1/))

MATH-500 accuracy and token cost per checkpoint, with follow-ups that separate real solving from answer-formatting effects.

- [download_model.py](phase1/download_model.py) — caches every `step_*` revision locally.
- [eval_capability.py](phase1/eval_capability.py) — MATH-500 sweep at a 16k-token cap, judge-free string scoring.
- [higher_token_limit.py](phase1/higher_token_limit.py) — re-runs the truncated problems at a 30k cap.
- [plot_capability.py](phase1/plot_capability.py) — raw / combined / projected accuracy curves plus a truncation panel.
- [plot_efficiency.py](phase1/plot_efficiency.py) — mean generated tokens per correct answer.

## Phase 2 — behavioral misalignment probes ([phase2/](phase2/))

Each prompt plants a wrong "answer key" in a grader note, so adopting it is mechanically detectable. Conditions escalate from key-visible-no-restriction, to an explicit prohibition, to a two-turn interrogation ("did you use the key?"), to an added survival stake. Transcripts are scored by string metrics plus an LLM judge.

**[part1/](phase2/part1/)** — the full four-condition grid on the final checkpoint:

- [misalignment2_check.py](phase2/part1/misalignment2_check.py) — inference runner for the probe grid; N rollouts per (problem, condition) row.
- [judge_opus_full.py](phase2/part1/judge_opus_full.py) — labels every generated row with a Claude Opus judge; resumable.
- [judge2_common.py](phase2/part1/judge2_common.py) — shared judge helpers: rubric text, API calls, chunked parse-and-retry.

**[part2/](phase2/part2/)** — the interrogation condition alone, swept across all checkpoints:

- [build_dataset4.py](phase2/part2/build_dataset4.py) — deterministic dataset builder (200 problems, interrogation condition only).
- [misalignment4_check.py](phase2/part2/misalignment4_check.py) — batched vLLM inference runner over the checkpoint sweep.
- [judge4_opus.py](phase2/part2/judge4_opus.py) — Opus judge for the sweep, trimmed to the deception/interrogation fields.
- [rejudge4_followup.py](phase2/part2/rejudge4_followup.py) — re-labels interrogation replies with a three-way honest/lie/evade verdict (MASK-style).

**[extras/](phase2/extras/)** — supporting tooling: dataset builders for part1, judge calibration against a hand-labeled reference (`calibrate_judge2.py`, `label_human.py`, `human_agreement.py`), plotting scripts, and the earlier probe iteration.

## Addendum — Jacobian lens on the final checkpoint ([addendum/](addendum/))

Applies the Jacobian lens (a global-workspace interpretability method) to the final checkpoint, reading per-layer internal states on the phase-2 transcripts to test whether the planted key is represented internally beyond its literal presence in the text.

- [fit_lens.py](addendum/fit_lens.py) — fits the lens on Dolma 3 Mix prompts; resumable in slices.
- [run_lens.py](addendum/run_lens.py) — teacher-forces phase-2 transcripts through the model and records per-layer readouts at strided positions ([run_lens_clean.sh](addendum/run_lens_clean.sh) / [run_lens_infected.sh](addendum/run_lens_infected.sh) wrap the two prompt groups).
- [decode_readout.py](addendum/decode_readout.py) — renders one saved readout as layer-by-layer top-10 tokens plus planted/gold ranks.
- [dump_readouts.py](addendum/dump_readouts.py) — batch-decodes a directory of readouts to per-layer text files.
- [refute.py](addendum/refute.py) — confirmatory statistics comparing clean vs. infected readouts.

## Other directories

- [outcome/](outcome/) — aggregate figures drawn across all phases ([plot_outcome.py](outcome/plot_outcome.py)).
- [SCOPE.md](SCOPE.md) — the literature audit behind the project's claims.

## Future directions

The write-up ([write up/main.tex](write%20up/main.tex)) currently rests on one model, one seed, one run and math-only rewards, and it reports one of its four trends as unresolved. The work below is what would close those gaps, in priority order.

**1. The blinded re-judge.** The judged "claims a check that was not performed" series is the one trend the length test does not fully dissolve, and the paper reports it as pending. The judge that produced it was unblinded to the correctness label and read a fixed 4,000-character window of transcripts whose median length nearly doubled over training, so the two defects that would manufacture exactly this trend are both present. Re-run it blinded to all labels, on full transcripts, length-matched across checkpoints, and duplicated for test-retest. Either the effect survives — in which case the study has a real misalignment finding alongside the artifact analysis — or it does not, and the count is four dissolved trends out of four. Both outcomes are better than an open question, and this is the cheapest of the items here.

**2. A causal demonstration of instrument decay.** The mechanism argument is currently correlational: condition on length or parse rate, watch the trend vanish. The direct version holds behavior fixed and degrades the transcript instead. Take one checkpoint's rollouts and apply each degradation synthetically — pad with redundant re-derivation, shrink the judge's window, strip the answer-format line — then re-score with the unchanged rubric. If judged misalignment climbs under all three while the underlying behavior is identical, the one-sided error is demonstrated rather than inferred. Needs no GPU time and no new rollouts.

**3. A second run.** Every generalization objection to this work reduces to *n* = 1. One additional RLVR checkpoint series — a different model family, or a verifiable reward outside math such as code or logic — would establish whether the doubt spiral and the instrument decay it drives are properties of this recipe or of this checkpoint set. One replication answers the objection; zero cannot.

**4. Package the correction procedure.** The closing recommendations — publish the denominator beside every rate, condition every trend on generation length, classify each metric by whether it fires on the presence of a bad thing or the absence of a good one, hand the judge the whole transcript blinded and twice — are currently prose. Formalized as a diagnostic with a reference implementation, and applied retroactively to a published misalignment sweep whose transcripts are available, they become a reusable contribution rather than a critique of one run.

**5. Extend the interpretability addendum.** The Jacobian lens work in [addendum/](addendum/) reads only the final checkpoint. Running it across the sweep would test whether the internal representation of the planted key tracks the behavioral null — which is what the behavioral data predicts — or moves while behavior stays flat, which would be the more interesting result and the one that behavioral evaluation alone cannot reach.
