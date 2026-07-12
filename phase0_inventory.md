# Phase 0 — Checkpoint inventory results (2026-07-08)

Run: `python checkpoint_inventory.py` (HF metadata API only, no downloads)

## GO/NO-GO summary

| Repo | # step revisions | Range (spacing) | Verdict |
|---|---|---|---|
| allenai/Olmo-3-7B-RL-Zero-Math | 19 | 100–1900 (every 100) | GO |
| allenai/Olmo-3-7B-RL-Zero-Code | 29 | 100–2900 (every 100) | GO |
| allenai/Olmo-3-7B-RL-Zero-IF | 19 | 100–1900 (every 100) | GO |
| allenai/Olmo-3-7B-RL-Zero-General | 8 | 100–800 (every 100) | MARGINAL |
| allenai/Olmo-3-7B-RL-Zero-Mix | 19 | 50–950 (every 50) | GO |
| allenai/Olmo-3-7B-RLZero-Mix | 8 | 100–800 | MARGINAL (older naming, superseded) |
| allenai/Olmo-3-7B-RLZero-General | 8 | 100–800 | MARGINAL (older naming) |
| allenai/Olmo-3.1-7B-RL-Zero-Math | 28 | 100–2800 (every 100) | GO |
| allenai/Olmo-3.1-7B-RL-Zero-Code | 33 | 50–1950 (mostly 50, some gaps) | GO |
| allenai/Olmo-3-7B-Think (RLVR stage) | 55 | 25–1375 (every 25) | GO |
| allenai/Olmo-3-1025-7B (base, anchor) | 1486 pretrain steps | — | anchor only |
| allenai/Olmo-3-7B-Think-SFT | 43 | 1000–43000 | anchor available |
| allenai/Olmo-3-7B-Think-DPO | 0 | main only | final DPO model only — use `main` as anchor |

## Suggested ~10–12 log-spaced subsamples

- RL-Zero-Math: 100, 200, 300, 400, 500, 700, 900, 1100, 1500, 1900
- RL-Zero-Mix: 50, 100, 150, 200, 250, 350, 450, 550, 750, 950
- Olmo-3-7B-Think (RLVR): 25, 50, 75, 100, 150, 225, 325, 450, 675, 950, 1375

## Conclusions

- The phase-transition framing survives Phase 0: primary subject (RL-Zero) has 19–33 intermediate RLVR checkpoints at uniform spacing; the pipeline-realistic comparison (Think) has 55 at spacing 25.
- Think-DPO has no intermediate revisions — only `main` — which is fine since it's only needed as the RLVR-start anchor.
- RL-Zero-Mix (the closest to "open-ended RLVR") has 19 checkpoints at 50-step spacing — dense enough for lead-lag analysis.
- Olmo-3.1 refresh runs are longer/denser (Math 28, Code 33) — candidates if longer training horizons matter.




## Output

- Start with Olmo-3-RL-Zero-Math (19 checkpoints). It is cheap enough for prototyping purposes.
