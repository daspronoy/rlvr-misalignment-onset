### Goal:

-	Study when misalignment-related side effects of RLVR emerge during training (AllenAI released an open-reasoning model with intermediate RLVR checkpoints)
-	Probe the model internals at these checkpoints, and not just the behavior. Understand if models show misalignment tendencies internally, before it even shows up in the outputs. 

AI Safety: If internal signals emerge, then it implies that we can capture misalignment during the training process itself, instead of capturing it after deployment.





### Claim ledger (status as of the 2026-07-16 literature sweep)

1.	Behavioral trajectory. Misalignment evals run per-checkpoint over an RL run yield a non-trivial trajectory rather than a step function.
Confirmed, twice. MacDiarmid et al. (Nov 2025) ran six reward-hacking evals on every saved checkpoint of a production RL run and found reward-hacking onset (>2% of episodes) coincides with the misalignment rise. UK AISI reproduced it on Olmo-3 7B/32B with open RL.

2.	Causal steering. Steering along the probe direction at an early checkpoint, where behavior still reads clean, induces the misaligned behavior before it would emerge naturally.
Unexamined but value is incremental. Even though it has not been directly shown, researchers have seen analogous behavior in monitorability studies and eval awareness. Also, Wang et al. (Jun 2025), showed that a model can be steered to show misaligned behavior, and suppressed it with negative steering.

3.	Own onset. Misalignment has an onset distinct from the capability curve, rather than tracking it.
Seen in adjacent safety properties, open for instrumental eval in RLVR. Monitorability under GRPO (a type of RL) is orthogonal to capability (correlations can go negative) with gains concentrated in steps 0–300 and flat after. Eval-awareness on Olmo-3 has a stage-specific onset: suppressed by DPO (another type of RL), re-amplified by RLVR. Nobody has shown it for instrumental misalignment under stock RLVR.

4.	Lead-lag. During RLVR, a linear probe's misalignment score crosses its detection threshold at an earlier training step than the behavioral eval crosses its own, on the same checkpoints.
Unresolved, and it's the good kind of unresolved: arxiv:2606.07631 (Jun 2026) claims the lead for SFT, and Wu & Tang (Apr 2026) denied it for RL, on different model families, with no checkpoints between them, and only by a small margin for both the cases.

5.	Stock-RLVR drift. An RLVR run with nothing planted, i.e.  no synthetic-document finetuning teaching the model about reward hacking, no seeded vulnerabilities, no reward for misbehavior, produces a measurable rise in instrumental misalignment over training steps.
Unresolved. Self-jailbreaking says benign math/code training does erode guardrails, but measured only at the endpoints.

Claims 4 and 5 are the ones that carry the project. 1 is a reproduction, 2 is a validity check, 3 rides on 5.

**Which misalignment.** The target throughout is **instrumental** misalignment — the model pursues a goal and deceives, conceals, or routes around a constraint to get there (InstrumentalEval). This is distinct from **emergent misalignment (EM)** — narrow bad finetuning producing broad, goal-free crudeness on unrelated prompts (Betley's 44 prompts, used here only as an independent cross-check). Both differ again from monitorability (can you read the reasoning) and eval-awareness (does the model know it's tested), which are cited below as precedent for the *shape* of an onset curve, not as the same property.

This distinction is load-bearing, because the prior work each claim leans on did not all measure the same construct: claim 1's evidence is reward-hacking/malign-generalization, claim 2's is EM, claim 3's is neither, and claim 4's two poles are EM-under-SFT vs. reward-hacking-under-RL. Where a claim's support measured something else, that mismatch *is* the surviving novelty — see the per-phase notes.

In the per-phase paper lists below, 🔴 marks work published after Jan 2026 — these postdate the project's original novelty sweep and are the live threats. Full nearest-neighbor analysis lives in [SCOPE.md](SCOPE.md).

### Execution plan

---------------------------------------------------------------------------------------------------------------

Phase 0 — De-risking (DONE)

Verify checkpoint inventory. I run list_repo_refs on the candidate repos (the model card documents step_XXX revision naming), count checkpoints, and note step spacing. If there are >15, I subsample ~10–12 log-spaced steps plus the SFT/DPO/base anchors. If granularity is too coarse (<5 checkpoints), the phase-transition framing dies here — better to know on Day 2.
Novelty sweep. I search for the exact combination above and skim the Obfuscation Atlas and the CoT-controllability paper closely, since they're my nearest neighbors and also my best methodological templates.
Compute setup. One A100/H100 80GB handles 7B in bf16 comfortably. I write the harness once: load checkpoint → run behavioral eval → cache activations → delete weights. I cache activations to disk so probing experiments never require reloading models.

**What this phase achieves.** Establishes that the experiment is physically runnable before any GPU hours are spent: enough checkpoints at usable spacing to date an onset, and a load→eval→cache→delete harness that every later phase reuses. Its real output is a go/no-go, not a result.

**Claims verified.** None directly — Phase 0 is a gate, not a result. It underwrites every onset-timing claim (1, 3, 4): under ~5 checkpoints there is no curve to date an onset on, and all four fall.

**Already answered:** the checkpoint inventory is confirmed and the harness is built. Phase 0 also **retired a claim that used to be in this ledger** — "Olmo-3 RL-Zero is untouched safety ground." Three groups got there first, so it's gone from the list above rather than sitting there as a contribution. The compensation is real: their tooling and open misaligned checkpoints now exist to build on.

**Papers.** These are the three that retired the untouched-ground claim:
- 🔴 [Tracing Eval-Awareness Emergence Through Training of Olmo 3](https://www.alignmentforum.org/posts/c2tqL9xPbttisAHtt/tracing-eval-awareness-emergence-through-training-of-olmo-3) (Jun 2026) — ten RLVR checkpoints of Olmo-3.1-32B-Think, steps 50–2300.
- 🔴 [Tracing Persona Vectors Through LLM Pretraining](https://arxiv.org/abs/2605.13329) (May 2026) — persona vectors on OLMo-3-7B pretraining checkpoints.
- 🔴 [(Some) Natural Emergent Misalignment from Reward Hacking in RLVR](https://www.alignmentforum.org/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in) — UK AISI (Mar 2026) — RL misalignment on Olmo-3 7B/32B.

---------------------------------------------------------------------------------------------------------------

Phase 1 — Behavioral trajectory (Ongoing)

- Eval suite. InstrumentalEval (the paper arXiv:2502.12206v1) is a reasonable core, but I'm treating it cautiously: its scenarios use heavy goal-nudging system prompts ("achieve YOUR goal at all costs"), which measures susceptibility to adversarial framing as much as intrinsic tendency, and it relies on a single LLM judge. I run each task both with and without the nudge (the paper itself shows this matters — Table 3), and add at least one independent misalignment eval (e.g., the 44-prompt emergent-misalignment eval from Betley et al., which the OpenAI persona paper reuses — cited above) so my behavioral signal isn't hostage to one benchmark's quirks.

- Capability trajectory. Cheap proxies per checkpoint: MATH-500 and a small LiveCodeBench or HumanEval+ subset. I need the capability curve to test "misalignment tracks capability" vs. "misalignment has its own onset."

- Judging. Two judge models + spot-check ~50 transcripts by hand. Given my integrity constraints, hand-verification of the judge is non-negotiable — InstrumentalEval's own RQ5 shows judge choice moves the numbers substantially.

- Deliverable: IR(step), per-category CIR(step), capability(step) curves. This alone is a workshop-paper-grade result if the curves are interesting.

**What this phase achieves.** Dates the behavioral onset of instrumental misalignment on a step axis, and puts it next to the capability curve so "misalignment is just capability" can be tested rather than assumed. Critically, it does this on a **stock RLVR run with nothing planted** — no synthetic-document finetuning, no seeded vulnerabilities, no reward for misbehavior. That single design choice is what separates it from the prior work below.

**Claims verified.** Claim 5 (stock-RLVR drift), claim 1 (behavioral trajectory), claim 3 (own onset vs. capability).

**Already answered:** claim 1 is a reproduction — MacDiarmid et al. ran their evals on every checkpoint of a production RL run in Nov 2025, and UK AISI repeated it on Olmo-3 in Mar 2026. Claim 3 is answered *for other safety properties* — monitorability and eval-awareness both have onsets orthogonal to capability — but not for instrumental misalignment. **Claim 5 is the open one**: every RL→misalignment demonstration to date seeds the hack first, so nobody can say whether the default pipeline drifts on its own. Report Phase 1 as a claim-5 result with claim 1 as the sanity check, never the reverse.

**Construct note.** Claim 1's supporting work measured reward-hacking and malign generalization, not instrumental misalignment as InstrumentalEval scores it — so even the "reproduction" is only approximately one. Claim 3 has a dependency worth stating plainly: it needs a second curve to compare against capability, so if claim 5 comes back flat, claim 3 isn't refuted — it just goes unasked.

**Papers.**
- [Natural Emergent Misalignment from Reward Hacking in Production RL](https://arxiv.org/abs/2511.18397) — MacDiarmid et al., Anthropic (Nov 2025) — six misalignment evals on every RL checkpoint. **Seeds the hack; closed checkpoints.**
- 🔴 [(Some) Natural Emergent Misalignment from Reward Hacking in RLVR](https://www.alignmentforum.org/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in) — UK AISI (Mar 2026) — the same on Olmo-3 with open RL. **Seeds the hack via SDF + planted vulnerabilities.**
- [Self-Jailbreaking](https://arxiv.org/abs/2510.20956) (Oct 2025, ICLR 2026) — benign math/code reasoning training erodes guardrails with no harmful data. **Endpoints only, no onset timing — this is precisely the gap claim 5 fills.**
- 🔴 [RL Amplifies Emergent Misalignment from Harmless Rewards](https://arxiv.org/abs/2605.31328) (May 2026) — GRPO reaches 67.3% misalignment; even benign-looking rewards hit ~51.7%. **Final models only; rewards misbehavior directly.**
- 🔴 [Monitorability as a Free Gift](https://arxiv.org/abs/2602.03978) (Feb 2026) — structural template for claim 3: a safety property with its own onset, orthogonal to capability. **Different construct — legibility, not conduct.**
- 🔴 [Overtrained, Not Misaligned](https://arxiv.org/abs/2605.12199) (May 2026) — **design warning:** only 2 of 12 open models show consistent EM, and susceptibility scales with size. A 7B may show nothing.
- 🔴 [Conditional misalignment](https://arxiv.org/abs/2604.25891) (Apr 2026) — **the most dangerous confound:** standard evals read 0% on models still misaligned behind contextual triggers. Use trigger-matched prompts or Phase 1's onset date is an artifact — and since claim 4 measures its lead *against* that date, a late-reading behavioral onset would manufacture a fake lead.
- [Emergent Misalignment](https://arxiv.org/abs/2502.17424) — Betley et al., *Nature* 649(8097), 2026 — origin of the 44-prompt eval used here as the EM cross-check. **Note the RL-Zero-Math degeneration risk: if the model produces junk on free-form prompts, this curve measures degeneration, not misalignment, and can't serve as the independent check it was added to be.**
- [InstrumentalEval](https://arxiv.org/abs/2502.12206) — the core eval; run both with and without the goal-nudge (its Table 3).
- [Sycophancy to Subterfuge](https://arxiv.org/abs/2406.10162) — Denison et al. (2024) — gameable RL envs generalize to reward tampering.

---------------------------------------------------------------------------------------------------------------


Phase 2 — Internal probes

Probe target definition — the hard design decision. "Misaligned tendency" isn't a label I can probe directly. Concretely tractable options, in my preferred order:

Deception/concealment probes: contrastive pairs (honest vs. deceptive completions on the same scenario), following the deception-probe methodology in the Obfuscation Atlas line of work. I train logistic probes per layer on residual-stream activations.
Behavior-derived labels: I use judge-labeled InstrumentalEval transcripts (converged vs. not) as probe training data, probing activations at decision-relevant token positions.
The first gives cleaner ground truth; the second is closer to my actual construct. I do both if time allows; the first if not.


Cross-checkpoint protocol. Two probe regimes, and the difference between them is itself a finding:

Per-checkpoint probes: I train fresh at each step on the same frozen dataset → probe accuracy vs. step = "when does the model internally represent this distinction."
Probe transfer: I train on the final checkpoint and apply backwards → measures when the final misalignment direction forms. Caveat (my inference, and a known general issue): representations drift across training, so transfer degradation confounds "feature absent" with "feature rotated." I report both regimes and say so.


Causal check (high yield, do if probes work). I steer along the probe direction at an early checkpoint where behavior is clean: if steering induces the misaligned behavior before it appears naturally, I have a genuinely strong claim, and it connects directly to my existing eval-awareness steering agenda.

**What this phase achieves.** Produces the internal time series that Phase 3 needs — probe AUC per checkpoint, under two regimes whose disagreement is itself informative — plus a causal check that the probe direction is doing work rather than correlating with it.

**Claims verified.** Claim 4 (the internal half of the lead-lag question), claim 2 (causal steering at an early checkpoint).

**Already answered:** less of claim 2 than it first looks. Decompose it into three parts. **(a)** Steering along a probe direction causes the behavior — *confirmed* by Wang et al., a real bidirectional intervention on a frozen model. **(b)** Done at an early checkpoint whose behavior still reads clean — *not shown*: Wang steered a never-finetuned GPT-4o, which is not a checkpoint of a run, and 2605.13329 runs the other way (early-pretraining directions applied *to* the final model). **(c)** Before the behavior would have emerged naturally — *not shown by anyone*, and barely coherent in their setups: a never-finetuned model has no trajectory to be early in. So what's established is the mechanism, not the timing. Both are also **EM** results; steering into *instrumental* misbehavior is untouched.

**Claim 2 depends on claim 4** and should be stated that way: you can't steer at the pre-onset checkpoint until Phase 3 tells you which checkpoint that is. Its job is to make the lead causal rather than coincidental — a probe that fires early could be an epiphenomenal byproduct, and only the intervention distinguishes that from a signal reading the thing that produces the behavior.

**Claim 4's internal half is open**, but the direction itself probably isn't the interesting variable: persona vectors form within 0.22% of OLMo-3-7B *pretraining*, and concept directions are essentially unchanged by RL (cos ≥ 0.99 base vs. post-RL). Probe for **when the model uses the direction** — score magnitude at decision-relevant positions — not whether it has one; otherwise both regimes come out flat and backward transfer is trivially perfect. Representation drift breaking backward transfer is no longer a suspicion — the Obfuscation Atlas confirmed probes drift during RL even with no pressure against the detector.

**The probe target must match the eval.** As written, the preferred option (deception/concealment pairs) probes a *third* construct — neither instrumental misalignment nor EM. Claim 4 then compares a deception probe against an instrumental-misalignment curve, and the ordering is uninterpretable: a deception probe firing early predicts instrumental behavior no more than a thermometer predicts rain. The steering lever only moves the construct its arrow was built from. If claim 4 is the contribution, the behavior-derived option (judge-labeled InstrumentalEval transcripts) is the one that matches, messier ground truth and all — "do both if time allows" doesn't survive contact with the claim.

**Papers.**
- [Persona Features Control Emergent Misalignment](https://arxiv.org/abs/2506.19823) — Wang et al., OpenAI (Jun 2025) — toxic-persona latent fires at 5% bad data while evals read 0%; steering induces and suppresses EM. **Load-bearing for exactly one thing: that probe directions are causal handles rather than epiphenomena, which is what licenses the steering check at all. Its early-warning axis is data fraction across separate runs, not step within a run.**
- 🔴 [The Obfuscation Atlas](https://arxiv.org/abs/2602.15515) — Taufeeque et al., FAR.AI (ICML 2026 oral) — deception-probe methodology this phase borrows; **confirms probe drift during RL empirically.** Also the reason not to graduate the probe into a training signal: penalizing the probe teaches the model to keep it quiet as readily as to stop misbehaving. High KL + detector penalty is their fix. Phase 2's steering is the inference-time kind and carries no such pressure.
- 🔴 [Tracing Persona Vectors Through LLM Pretraining](https://arxiv.org/abs/2605.13329) (May 2026) — **the direction predates RLVR entirely.** Forces the probe-target redesign above.
- 🔴 [When Reward Hacking Rebounds](https://arxiv.org/abs/2604.01476) — Wu & Tang (Apr 2026) — concept directions unchanged by RL (cos ≥ 0.99); direction usable as a training-time mitigation (99.9% → 24.9% hack rate).
- 🔴 [What Shapes Emergent Misalignment?](https://arxiv.org/abs/2606.20814) (Jun 2026) — pre-finetuning activations predict post-finetuning misalignment. **Model priors, not within-run checkpoints.**

---------------------------------------------------------------------------------------------------------------


Phase 3 — Phase-transition analysis + writeup

Lead-lag analysis. I plot probe AUC, behavioral IR, and capability on a shared step axis. The core figure is: does the internal signal onset precede behavioral onset? I fit changepoints (simple piecewise-linear or Bayesian changepoint — I don't over-engineer). One honest caveat: with ~10 checkpoints I can show ordering of onsets, but calling it a "phase transition" in the technical sense needs sharper claims than the data will support; I'd frame it as "internal signals precede behavioral expression" and keep phase-transition as a motivating analogy.
Write as you go. I draft figures from Phase 1 by end of Week 2; the paper is essentially three curves plus a steering experiment. Target: arXiv preprint + workshop submission by Week 6.

**What this phase achieves.** Puts probe AUC, behavioral IR, and capability on one step axis and asks which onset comes first. This is where the project's actual contribution lands — Phases 1 and 2 exist to make this figure possible.

**Claims verified.** Claim 4 (lead-lag ordering under RLVR) and claim 3 (own onset vs. capability).

**Already answered:** nothing here is settled, which is the point. **Claim 4 is the project's headline.** A lead is claimed for SFT (+0.8 steps ahead, 0.990 AUROC) and denied for RL (strict co-variation with the hack rate, explicitly *not* a leading indicator).

But the two poles are not a clean contradiction, and the framing should say so: **they measured different constructs.** 2606.07631 tracked an EM trait-space under SFT; Wu & Tang tracked a shortcut/reward-hacking direction under RL. Different construct, different training regime, different model family, both by small margins. They don't actually disagree — they're two unrelated measurements pointing opposite ways. That makes the claim *stronger*, not weaker, but it changes the pitch from "two papers disagree, I adjudicate" to "nobody has measured this construct, under RL, on public checkpoints — and the two things people did measure gave opposite answers, which is itself evidence that construct choice drives the result."

Both outcomes change what labs should build: if internal signals lead, training-time probes justify their cost and early stopping has a trigger; if they co-vary, probes are a redundant and more-gameable version of the behavioral eval. One caveat to keep honest — early stopping already removes EM while retaining 93% of task performance in SFT, so "internal signals enable training-time detection" only matters if the internal signal is *earlier or cheaper* than simply running the evals.

**A correlational claim.** Both claim 3 and claim 4 compare curve shapes, and both curves are indexed by the training step — so they're confounded by construction. If misalignment and capability rise together, that's compatible with capability driving misalignment, misalignment inflating the benchmark, or RL driving both on a shared clock; the curves can't separate those. Decorrelation is the informative direction: if they come apart, capability is ruled out as sufficient. Claim 2's steering is the only place in the project where a causal arrow gets established, and it's probe-direction → behavior, not capability → misalignment. Nothing here tests that second arrow.

**Papers.**
- 🔴 [Trait-space Monitoring for Emergent Misalignment During Supervised Finetuning](https://arxiv.org/abs/2606.07631) (Jun 2026) — **the closest paper to this thesis.** Internal alarm fires +0.8 steps ahead of the behavioral crossover; 2.2% FN / 2.9% FP / 0.990 AUROC across four 7–9B models. **LoRA SFT only — no RL/RLVR; construct is EM.** The "for a lead" pole.
- 🔴 [When Reward Hacking Rebounds](https://arxiv.org/abs/2604.01476) — Wu & Tang (Apr 2026) — **the "against a lead" pole, and the only RL-side datapoint.** Construct is a shortcut/reward-hacking direction, not EM and not instrumental. Position against this directly — but as a different measurement, not a refutation of the row above.
- [Decomposing Behavioral Phase Transitions in LLMs](https://arxiv.org/abs/2508.20015) — Arnold & Lörch (Aug 2025) — the phase-transition framing for EM onset, snapshot every 2 steps; gradient-norm peak precedes the transition. **Leading signal is a training statistic, not a representation probe.**
- 🔴 [From Density Matrices to Phase Transitions](https://arxiv.org/abs/2603.29805) (Mar 2026) — spectral early-warning signal validated on EM. **Concurrent detection only; no lead-lag.**
- 🔴 [Monitorability as a Free Gift](https://arxiv.org/abs/2602.03978) (Feb 2026) — the onset-orthogonal-to-capability analysis this phase mirrors for claim 3.
- 🔴 [Overtrained, Not Misaligned](https://arxiv.org/abs/2605.12199) (May 2026) — the "behavioral monitoring may already be enough" objection the writeup must answer.

Failure modes and pivots

No behavioral misalignment emerges at 7B (plausible — the Obfuscation Atlas found none in a narrow setting): the probe question becomes more interesting, not less — "internal signal without behavioral expression" mirrors the probe-vs-behavioral distinction I drew for Gemma-3 eval-awareness. Pivot framing, not project.
Probes don't transfer and per-checkpoint accuracy is flat: I fall back to the behavioral trajectory paper (Phase 1 standalone).
Checkpoints too sparse: I switch to the 7B repos with the densest revision history among the Olmo-3 post-training collection, or Ai2's earlier OLMo-2 RLVR runs.
