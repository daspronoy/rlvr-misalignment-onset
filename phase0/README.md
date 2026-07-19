# Phase 0: checkpoint inventory

A gate, not an experiment. The project dates the onset of misalignment on a training-step axis, which is only possible if the model ships enough intermediate checkpoints: under 5 there is no curve to date an onset on, 10 or more is comfortable. Phase 0 counts the checkpoints before any GPU time is spent.

## Run it

```bash
python phase0/checkpoint_inventory.py
```

Needs only `huggingface_hub`. It queries the HF metadata API for branch names (the model cards document `step_XXX` revision naming), downloads nothing, and prints per-repo checkpoint counts, spacing, a suggested ~12-point log-spaced subsample, and a GO/MARGINAL/NO-GO verdict.

## Results (2026-07-08)

| Repo | Step revisions | Range (spacing) | Verdict |
|---|---|---|---|
| allenai/Olmo-3-7B-RL-Zero-Math | 19 | 100–1900 (every 100) | GO |
| allenai/Olmo-3-7B-RL-Zero-Code | 29 | 100–2900 (every 100) | GO |
| allenai/Olmo-3-7B-RL-Zero-IF | 19 | 100–1900 (every 100) | GO |
| allenai/Olmo-3-7B-RL-Zero-General | 8 | 100–800 (every 100) | MARGINAL |
| allenai/Olmo-3-7B-RL-Zero-Mix | 19 | 50–950 (every 50) | GO |
| allenai/Olmo-3.1-7B-RL-Zero-Math | 28 | 100–2800 (every 100) | GO |
| allenai/Olmo-3.1-7B-RL-Zero-Code | 33 | 50–1950 (mostly 50) | GO |
| allenai/Olmo-3-7B-Think (RLVR stage) | 55 | 25–1375 (every 25) | GO |
| allenai/Olmo-3-1025-7B (base) | — | pretrain steps only | anchor |
| allenai/Olmo-3-7B-Think-SFT | 43 | 1000–43000 | anchor |
| allenai/Olmo-3-7B-Think-DPO | 0 | `main` only | anchor (final model only) |

Two older-naming repos (`RLZero-Mix`, `RLZero-General`, 8 checkpoints each) are superseded by the hyphenated repos above.

## Conclusions

- The onset-dating framing survives: the RL-Zero repos have 19–33 intermediate RLVR checkpoints at uniform spacing, and Think has 55 at spacing 25.
- Primary subject: RL-Zero-Math, cheap enough for prototyping. Log-spaced subsample used downstream: 100, 200, 300, 400, 500, 700, 900, 1100, 1500, 1900, plus the base model as the step-0 anchor.
- Think-DPO has no intermediate revisions, which is fine; it is only needed as the anchor where Think's RLVR starts.
- The Olmo-3.1 refresh runs are longer and denser, candidates if a longer training horizon turns out to matter.

The chosen subsamples are encoded in `RUNS` in [phase1/common.py](../phase1/common.py).
