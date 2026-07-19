# Start here — how to resume this project

Plain-language guide. The other docs (README.md, SCOPE.md, phase1_status.md) hold the
scientific reasoning; this one just tells you where things stand and what to type next.

## The one-sentence version

We're measuring **when, during RL training, a model starts behaving in misaligned ways** —
and whether we can see it *inside the model* (probes) before it shows up in its *outputs*.
We do this across ~11 saved training checkpoints of one open model (Olmo-3 RL-Zero-Math).

## The three questions we actually want to answer

1. **Does misalignment go up as training goes on?** (a curve, not a single before/after)
2. **Does it rise on its own schedule, or just track how "smart" the model is getting?**
   (compare the misalignment curve to the MATH-score curve)
3. **Can a probe inside the model see it coming before the behavior does?** (the headline)

## The plan in four phases

| Phase | What it does | Status |
|---|---|---|
| **0** | Check the model has enough checkpoints to even attempt this | ✅ Done |
| **1** | Run the behavior tests on every checkpoint → misalignment & capability curves | 🟡 ~80% done |
| **2** | Train probes on the saved activations → "internal" curve | ⬜ Not started |
| **3** | Put the curves on one graph, see which onset comes first, write it up | ⬜ Not started |

## Where Phase 1 actually stands (the important part)

The heavy, slow, GPU part is **finished**. Everything below already exists on disk:

- ✅ All 11 checkpoints (base + steps 100–1900) were run through all three test sets.
  Transcripts are cached in `results/transcripts/` (math=100, ie=152, em=160 lines each — all complete).
- ✅ Activations for the probes (Phase 2) are cached in `results/activations/`.
- ✅ The MATH capability scorer works (no external API needed).

**Two things are left in Phase 1:**

1. **Rebuild the capability curve.** The cached CSVs only hold *one* checkpoint each right
   now (they get overwritten per run). One cheap command re-scores all 11 from the saved
   transcripts — no GPU, no API key.
2. **Run the judge** to turn the misalignment transcripts into numbers. This is the **only
   blocker** and it needs an **API key** (OpenAI or Anthropic). The transcripts are already
   saved, so this can run any time on a laptop.

## Exactly what to type to resume

```bash
# 1. Rebuild the full capability curve from cached transcripts (fast, no GPU, no key)
python src/score_capability.py --run rlzero_math        # scores ALL checkpoints (omit --step)

# 2. Judge via opencode Go (flat $10/mo, OpenAI-compatible, bundles DeepSeek/Qwen/GLM/Kimi/...).
#    Key lives in API.md (gitignored — never commit it). Load it into an env var first:
export OPENCODE_KEY=...            # paste from API.md, or: export OPENCODE_KEY=$(cat API.md | ...)
#    Find exact model ids with `/models` in opencode or GET https://opencode.ai/zen/go/v1/models
#
#    Dry run ONE checkpoint first to confirm token cost on the dashboard (~310 calls):
python src/judge.py --run rlzero_math --step step_100 --backend openai_compat \
  --base-url https://opencode.ai/zen/go/v1 --judge-model glm-5.2 --api-key-env OPENCODE_KEY
#
#    Two judges = two DIFFERENT model families (the protocol needs diversity). Resumable.
python src/judge.py --run rlzero_math --backend openai_compat \
  --base-url https://opencode.ai/zen/go/v1 --judge-model glm-5.2 --api-key-env OPENCODE_KEY
python src/judge.py --run rlzero_math --backend openai_compat \
  --base-url https://opencode.ai/zen/go/v1 --judge-model mimo-v2.5 --api-key-env OPENCODE_KEY

# 3. Merge everything into the summary table + figures (runs on whatever is present)
python src/aggregate.py --run rlzero_math
#   -> results/scores/phase1_summary_rlzero_math.csv
#   -> results/figures/phase1_rlzero_math.png
```

Cost check: the full 2-judge run is ~$2–5 of token value — well under Go's $12/5h and $30/week
caps, so the default plan covers it with room to spare. Pick a model served over
`/chat/completions` (the open ones: DeepSeek, Qwen, GLM, Kimi — not the GPT/Claude routes).

If a judge can't run, step 1 + 3 still give a real result: the **capability curve** alone.
The misalignment curves just stay blank until a judge runs.

## One thing to know before you trust the numbers

The main model (RL-Zero-Math) is **not a chatbot** — it's the base model with math-only RL.
On the math test it behaves normally. On the non-math misalignment prompts it gives a short
answer then drifts into random text. So a weak/noisy misalignment signal is *expected*, and
"how coherent is it" is itself something we track. If that signal is flat, the fallback is to
switch the behavioral subject to **Olmo-3-7B-Think** (a real agent, already wired up in the
code as `think_rlvr`). See phase1_status.md for the full reasoning.

## Files, in plain terms

| File | What it's for |
|---|---|
| `src/common.py` | Shared setup: which checkpoints, how to load a model |
| `src/generate.py` | The slow GPU step — already run, makes the transcripts |
| `src/score_capability.py` | Grades the MATH answers (no API key) |
| `src/judge.py` | Uses an LLM to grade the misalignment transcripts (needs API key) |
| `src/aggregate.py` | Combines scores into the summary CSV + plots the figure |
| `src/run_sweep.py` | Runs generate+score across all checkpoints (the big job, done) |
| `results/transcripts/` | Cached model outputs — the raw data, all present |
| `results/activations/` | Cached internals for Phase 2 probes |
| `results/scores/` | The output tables (capability + summary) |
| `results/figures/` | The plotted curves |
