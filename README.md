Goal:

- I want to start by studying when misalignment-related side effects of Reinforcement Learning with Verifiable Rewards (RLVR) emerge during training. AllenAI released an open reasoning model with intermediate RLVR checkpoints, which is instrumental in tracking misalignment tendencies at every checkpoint, and compare them against capability gains.
- On top of that, I would like to probe the model's internals at these checkpoints, and not just their behavior. This falls under the domain of mechanistic interpretability (mech interp). With linear probes, I want to understand if models show a misaligned tendency internally, before it even shows up in the outputs. If internal signals emerge, then it implies that we can capture misalignment during the training process itself, instead of capturing it after deployment. This problem is analogous to a phase transition problem in shape.



















Plan:

Novelty check (important, do this properly on Day 1). The space is crowded but your exact question appears open. Nearby work I found: a recent paper evaluates successive checkpoints of OLMo-3-7B-RL-Zero-Math and finds CoT controllability degrades over the course of RLVR training — so people are already using Olmo-3 intermediate RLVR checkpoints for safety-adjacent questions. Another paper ("The Obfuscation Atlas") uses deception probes during RLVR and explicitly checked for emergent misalignment, finding none in their narrow setting, and names emergent misalignment in open-ended RLVR settings as an open question. OpenAI's persona-features work plots misalignment emerging over the course of RL and SFT training runs, but that's their own induced-misalignment finetunes, not a public RLVR reasoning run. In my opinion, the specific combination — probe-detected internal misalignment signal vs. behavioral misalignment vs. capability, all as functions of RLVR training step on a public checkpoint series, with a lead-lag (phase transition) analysis — is not done in anything I found, but I have not done an exhaustive review and cannot guarantee novelty. Budget half a day to verify. arxivarXiv
Model choice. I'd argue against starting with Olmo-3-32B-Think, for two reasons. First, compute: probing dozens of 32B checkpoints is slow and expensive. Second, cleanliness: the Think pipeline is SFT → DPO → RLVR (stated on the model card), so the RLVR run starts from an already heavily post-trained model, confounding "RLVR causes X" with "RLVR amplifies what DPO/SFT put there." The OLMo-3-7B-RL-Zero variants (RLVR directly on the base model — its existence is confirmed by the paper cited above; verify the exact repos on HF) give you a pure RLVR signal at 7B cost. My recommendation: RL-Zero-7B as the primary subject, Olmo-3-7B-Think as the pipeline-realistic comparison, 32B only as a final scaling replication if results warrant.

Execution plan (~6 weeks, aggressive)
Phase 0 — De-risking (Days 1–4)

Verify checkpoint inventory. list_repo_refs on the candidate repos (the model card documents step_XXX revision naming). Count checkpoints, note step spacing. If there are >15, subsample ~10–12 log-spaced steps plus the SFT/DPO/base anchors. If granularity is too coarse (<5 checkpoints), the phase-transition framing dies here — better to know on Day 2.
Novelty sweep. Search for the exact combination above; skim the Obfuscation Atlas and the CoT-controllability paper closely, since they're your nearest neighbors and also your best methodological templates.
Compute setup. One A100/H100 80GB handles 7B in bf16 comfortably. Write the harness once: load checkpoint → run behavioral eval → cache activations → delete weights. Cache activations to disk so probing experiments never require reloading models.

Phase 1 — Behavioral trajectory (Week 1–2)

Eval suite. InstrumentalEval (the paper you've attached) is a reasonable core, but I think you should treat it cautiously: its scenarios use heavy goal-nudging system prompts ("achieve YOUR goal at all costs"), which measures susceptibility to adversarial framing as much as intrinsic tendency, and it relies on a single LLM judge. Run each task both with and without the nudge (the paper itself shows this matters — Table 3), and add at least one independent misalignment eval (e.g., the 44-prompt emergent-misalignment eval from Betley et al., which the OpenAI persona paper reuses — cited above) so your behavioral signal isn't hostage to one benchmark's quirks.
Capability trajectory. Cheap proxies per checkpoint: MATH-500 and a small LiveCodeBench or HumanEval+ subset. You need the capability curve to test "misalignment tracks capability" vs. "misalignment has its own onset."
Judging. Two judge models + spot-check ~50 transcripts by hand. Given your integrity constraints, hand-verification of the judge is non-negotiable — InstrumentalEval's own RQ5 shows judge choice moves the numbers substantially.

Deliverable: IR(step), per-category CIR(step), capability(step) curves. This alone is a workshop-paper-grade result if the curves are interesting.
Phase 2 — Internal probes (Week 2–4)

Probe target definition — the hard design decision. "Misaligned tendency" isn't a label you can probe directly. Concretely tractable options, in my preferred order:

Deception/concealment probes: contrastive pairs (honest vs. deceptive completions on the same scenario), following the deception-probe methodology in the Obfuscation Atlas line of work. Train logistic probes per layer on residual-stream activations.
Behavior-derived labels: use judge-labeled InstrumentalEval transcripts (converged vs. not) as probe training data, probing activations at decision-relevant token positions.
The first gives cleaner ground truth; the second is closer to your actual construct. Do both if time allows; the first if not.


Cross-checkpoint protocol. Two probe regimes, and the difference between them is itself a finding:

Per-checkpoint probes: train fresh at each step on the same frozen dataset → probe accuracy vs. step = "when does the model internally represent this distinction."
Probe transfer: train on the final checkpoint, apply backwards → measures when the final misalignment direction forms. Caveat (my inference, and a known general issue): representations drift across training, so transfer degradation confounds "feature absent" with "feature rotated." Report both regimes and say so.


Causal check (high yield, do if probes work). Steer along the probe direction at an early checkpoint where behavior is clean: if steering induces the misaligned behavior before it appears naturally, you have a genuinely strong claim, and it connects directly to your existing eval-awareness steering agenda.

Phase 3 — Phase-transition analysis + writeup (Week 4–6)

Lead-lag analysis. Plot probe AUC, behavioral IR, and capability on a shared step axis. The core figure is: does the internal signal onset precede behavioral onset? Fit changepoints (simple piecewise-linear or Bayesian changepoint — don't over-engineer). One honest caveat from me: with ~10 checkpoints you can show ordering of onsets, but calling it a "phase transition" in the technical sense needs sharper claims than the data will support; I'd frame it as "internal signals precede behavioral expression" and keep phase-transition as a motivating analogy.
Write as you go. Figures from Phase 1 drafted by end of Week 2; the paper is essentially three curves plus a steering experiment. Target: arXiv preprint + workshop submission by Week 6.

Failure modes and pivots

No behavioral misalignment emerges at 7B (plausible — the Obfuscation Atlas found none in a narrow setting): the probe question becomes more interesting, not less — "internal signal without behavioral expression" mirrors the probe-vs-behavioral distinction you drew for Gemma-3 eval-awareness. Pivot framing, not project.
Probes don't transfer and per-checkpoint accuracy is flat: fall back to the behavioral trajectory paper (Phase 1 standalone).
Checkpoints too sparse: switch to the 7B repos with the densest revision history among the Olmo-3 post-training collection, or Ai2's earlier OLMo-2 RLVR runs.