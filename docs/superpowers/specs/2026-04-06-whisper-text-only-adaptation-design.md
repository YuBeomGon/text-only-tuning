# Design: Whisper Text-Only Domain Adaptation + Harness Automation

**Date:** 2026-04-06
**Status:** Draft
**Paper:** Domain-Specific Adaptation for ASR through Text-Only Fine-Tuning (MMLoSo 2025)

---

## 1. Project Identity

This project is a **paper proxy experiment**, not a paper reproduction.

| Aspect | Paper | This project |
|--------|-------|-------------|
| Training target | Paper text suggests decoder-focused adaptation, while the detailed method emphasizes updating bias matrices (and routing params in MoE extension). Ambiguous whether decoder weights themselves are updated. | B only (encoder/decoder freeze) — conservative v1 choice |
| B role during training | replaces cross-attention K/V | replaces encoder output (same level) |
| Interpolation point at inference | decoder cross-attention K/V (per-layer independent) | encoder_output (pre-layer, single mix) |
| Inference engine | — | HF Whisper (research/eval), faster-whisper (serving target, separate work) |
| Primary eval metric | WER | CER (project choice: target domain is Korean where CER is more stable than WER due to ambiguous word boundaries). WER also recorded in registry for paper comparison |

Key differences from paper:
1. **Interpolation point**: paper interpolates projected K/V inside each decoder layer; this project interpolates at encoder_output level before any projection.
2. **Per-layer independence**: in the paper, each decoder layer receives independently interpolated K/V; in this project, all decoder layers receive the same mixed input.
3. **Training scope**: paper's training target is ambiguous (see table above); this project trains B only as a conservative starting point.

---

## 2. Two Work Streams

### Stream 1: Paper implementation and testing
- HF Whisper-based training and inference
- Encoder output interpolation proxy experiment

### Stream 2: Harness automation
- Automated experiment loop
- Score gate-based promotion/discard

Each stream must have its own completeness. Do not mix work across streams in the same task.

---

## 3. Training Design

### Architecture

```
[HF Whisper Encoder] -> E_pretrained extraction (once) -> B initialization
encoder: freeze
decoder: freeze
B: trainable (sole training target)

During training: B fully replaces encoder output -> fed to decoder cross-attention
Input: domain text in Whisper multitask token format
Loss: CE only (v1 decided policy)
```

### B Initialization: E_pretrained Extraction (decided)

B is initialized from `E_pretrained`, the pretrained Whisper encoder's output.
The paper states `B = E_pretrained` but does not specify what input produces this output.

**Decided method:** Encode a small set of representative domain audio (a few samples) through the pretrained Whisper encoder, average the resulting encoder outputs, and use this as E_pretrained.

- Input: domain audio samples (a few files are sufficient)
- Process: `E_pretrained = mean([encoder(audio_1), encoder(audio_2), ...])`
- Output: single tensor of shape `(seq_len, d_model)`, saved once, reused for all training runs
- This is done once before Phase 1 starts

This choice ensures:
- B starts in the correct encoder output representation space
- Domain signal is embedded from the start (unlike silence or noise input)
- Deterministic and reproducible (same audio set → same E_pretrained)

### v1 Loss Policy (decided, not open)

- v1: CE only
- Ablation order (Phase 3): CE -> CE+KL -> CE+KL+term_penalty
- The automated loop must NOT explore loss space simultaneously

### Training Hyperparameters (paper starting point)

| Parameter | Value |
|-----------|-------|
| max_steps | 1,000 |
| warmup_steps | 200 |
| eval_every | 50 steps |
| checkpoint_every | 50 steps |
| logging_every | 10 steps |
| optimizer | HF default (AdamW) |
| learning_rate | HF default, adjust if needed |

Step promotion ladder: 100 -> 300 -> 1,000.

### Output

Trained B saved as `.pt` file, one per domain (e.g., `priors/insurance_b_v1.pt`).

---

## 4. Inference Design

```
HF Whisper encoder(audio) -> encoder_output
encoder_output_mixed = alpha * encoder_output + (1 - alpha) * B
-> original (frozen) decoder -> decoding
```

- This is a proxy for the paper's K/V interpolation, not an exact reproduction.
- Research/eval uses HF Whisper.
- Serving transition to faster-whisper is a separate work stream.

### Alpha Sweep Plan

| alpha | Meaning |
|-------|---------|
| 1.0 | baseline (no prior) |
| 0.9 | 90% original, 10% prior — conservative |
| 0.7 | 70% original, 30% prior — moderate |
| 0.5 | paper default — equal mix |

---

## 5. Eval Design

### Data
- Audio: wav files, 30 seconds or shorter
- Manifest: jsonl format, one JSON object per line

```
data/processed/eval_v1/
├── audio/
│   └── <batch>/
│       └── <chunk_id>.wav
├── manifest.jsonl
├── metadata.jsonl
└── rejects.jsonl
```

Manifest entry schema:
```json
{
  "id": "00003004871752551193_l__0000",
  "audio": "audio/AIG_녹취반출_20250715/00003004871752551193_l__0000.wav",
  "text": "네 여보세요 ...",
  "duration_sec": 14.952,
  "batch": "AIG_녹취반출_20250715",
  "source_id": "00003004871752551193_l",
  "start_sec": 23.576,
  "end_sec": 38.528
}
```

- `audio` field is a **relative path** from the manifest file's parent directory
- `text` field is the reference transcription
- Domain terms: lexicon file (one term per line)

### Metrics

| Metric | Role |
|--------|------|
| CER | primary gate metric (project choice) |
| WER | recorded for paper comparison |
| domain_term_recall | domain term restoration rate |
| hallucination_rate | short-segment hallucination monitoring |
| timestamp_error | alignment consistency (needed for serving transition) |
| latency_ms | inference speed |

---

## 6. Phase Design (renumbered)

Fixed prior (domain audio encoding) is deferred due to domain audio unavailability.
Phase numbers are reassigned accordingly.

| Phase | Goal | Content |
|-------|------|---------|
| 0 | Baseline lock + plumbing validation | 0-a: Eval set finalization, baseline metrics with original HF Whisper. 0-b: Verify alpha=1.0 reproduces baseline exactly, B shape matches inference path, alpha<1.0 decode loop runs without crash |
| 1 | Text-only B training | CE loss, 100->300->1000 step promotion, B saved |
| 2 | HF inference interpolation | Trained B with alpha sweep (0.9, 0.7, 0.5), observe CER + term recall + hallucination |
| 3 | Loss ablation | CE+KL -> CE+KL+term_penalty, same alpha sweep per loss variant |
| 4 | Robustness | Reproduce best combo, out-of-domain sanity, short-segment hallucination check |

---

## 7. Harness Automation Design

### 7-1. Leader

Claude Code single session acts as leader.
Next experiment selection is performed **only within auto_policy.yaml constraints**, not by free judgment.

### 7-2. auto_policy.yaml (machine-readable policy)

```yaml
current_phase: 0

phase_config:
  0:
    goal: baseline_lock
    auto_generate_next: false
    phase_transition_requires_human: true
  1:
    goal: text_only_b_training
    allowed_steps: [100, 300, 1000]
    loss: ce_only
    auto_generate_next: true
    phase_transition_requires_human: true
  2:
    goal: hf_inference_interpolation
    allowed_alphas: [0.9, 0.7, 0.5]
    auto_generate_next: true
    phase_transition_requires_human: true
  3:
    goal: loss_ablation
    ablation_order: [ce, ce_kl, ce_kl_term]
    auto_generate_next: true
    phase_transition_requires_human: true
  4:
    goal: robustness
    auto_generate_next: false
    phase_transition_requires_human: true

loop_control:
  initial_human_review_count: 5
  max_consecutive_fails: 3
```

### 7-3. Score Gate (4-level)

Every candidate is evaluated in **two stages**:

1. **Viability check**: is the candidate better than the **original baseline** (Phase 0 frozen)?
   - If no → FAIL (regardless of other metrics)
2. **Promotion check**: is the candidate better than the **current_best**?
   - If yes → eligible for PASS or REPRO_REQUIRED
   - If no → VIABLE (recorded in registry, but current_best is NOT replaced)

| Decision | Condition | Automated action |
|----------|-----------|-----------------|
| **PASS** | Beats current_best on CER >= 3% relative AND term_recall >= 5% relative AND safety OK | Auto promote to new current_best, record in registry |
| **REPRO_REQUIRED** | Beats current_best on CER or term_recall (one meets threshold, other within hold floor) + safety OK + not yet reproduced | Auto re-run once, promote if reproduced |
| **VIABLE** | Beats baseline but does NOT beat current_best | Record in registry with status=VIABLE, do NOT update current_best |
| **HUMAN_REVIEW** | Improvement exists but safety metric at boundary | Stop loop, report to human |
| **FAIL** | Does not beat baseline | Discard, keep current_best, **no code revert** |

Safety thresholds (provisional — calibrate after Phase 0 baseline variance measurement):
- hallucination_abs_max: 0.01
- timestamp_rel_max: 0.05
- latency_rel_max: 0.10

Hold floor (REPRO_REQUIRED tolerance, provisional):
- CER hold floor: -0.02 (allow up to 2% relative regression)
- term_recall hold floor: -0.03 (allow up to 3% relative regression)

All thresholds are **provisional**. During Phase 0, run baseline 2-3 times to measure variance. Calibrate thresholds during the initial 5-run human review period. Document any threshold changes in state.md.

### FAIL behavior (explicit)

- Record `status=FAIL` in registry.csv
- Do NOT change current_best in state.md
- Preserve experiment output files (available for analysis)
- Do NOT perform git revert

### 7-4. Loop Stop Conditions

- HUMAN_REVIEW triggered
- 3 consecutive FAILs
- Phase transition point (all phases require human approval to advance)

### 7-5. Execution Chain

```
Read auto_policy.yaml
-> Determine next experiment within current phase constraints
-> Generate config.yaml
-> run.sh(config.yaml)
  -> run_once.sh (train / inference / eval)
  -> score.sh (baseline_metrics.json, current_best_metrics.json, candidate_metrics.json)
    -> Stage 1: viability check (candidate vs baseline)
    -> Stage 2: promotion check (candidate vs current_best)
-> Decision (PASS / REPRO_REQUIRED / VIABLE / HUMAN_REVIEW / FAIL)
-> Update registry.csv
-> Update state.md (update current_best only on PASS or reproduced REPRO_REQUIRED)
-> Next loop iteration or stop
```

### 7-6. Initial Human Review Period

First 5 experiments: even if PASS, stop and report to human.
Purpose: validate that score gate thresholds are trustworthy before full automation.

---

## 8. File Layout

```
project-root/
├── CLAUDE.md                          # agent behavioral rules + file role definitions
├── program.md                         # current goal, phase, priorities
├── state.md                           # current best, risks, next actions
├── auto_policy.yaml                   # machine-readable automation policy
├── harness/
│   ├── run.sh                         # experiment entrypoint
│   ├── run_once.sh                    # single deterministic run
│   └── score.sh                       # gate: PASS/REPRO_REQUIRED/HUMAN_REVIEW/FAIL
├── configs/
│   └── (per-experiment config.yaml)
├── src/
│   ├── train/
│   │   └── train_text_only.py         # B training with CE loss
│   ├── inference/
│   │   └── hf_encoder_mix.py          # HF Whisper encoder_output interpolation
│   └── eval/
│       └── run_eval.py                # CER, WER, term recall, hallucination
├── data/
│   ├── manifests/                     # eval set manifests (format TODO)
│   └── lexicon/                       # domain term lists
├── priors/
│   └── (trained B files, .pt)
├── experiments/
│   ├── registry.csv                   # all experiment records
│   └── outputs/
│       └── (per-run results)
├── docs/
│   ├── papers/
│   └── superpowers/specs/
└── tests/
```

---

## 9. Open Items

| Item | Status | Note |
|------|--------|------|
| Eval manifest format | Decided | jsonl with id, audio (relative path), text, duration_sec, batch, source_id, start_sec, end_sec |
| E_pretrained extraction | Decided | Encode domain audio samples, average encoder outputs (see Section 3) |
| Domain audio for E_pretrained | Required before Phase 1 | A few representative domain audio samples needed |
| Score gate thresholds | Provisional | Calibrate after Phase 0 baseline variance measurement |
| Optimizer / lr / weight decay | Starting point decided | HF defaults, adjust if needed |
| Faster-whisper serving transition | Separate work | Not part of this research eval |

---

## 10. Terminology

To avoid confusion in all project documents:

| Term | Meaning |
|------|---------|
| paper-kv-interp | Paper's original approach: per-layer K/V interpolation in decoder cross-attention |
| fw-encoder-mix | This project's proxy: encoder_output level interpolation before decoder |
| B | Trainable bias parameter, initialized from E_pretrained, sole training target in v1 |
| E_pretrained | Mean encoder output from representative domain audio through pretrained Whisper encoder. Used for B initialization only (not a domain prior by itself) |
| alpha | Interpolation weight: 1.0 = pure encoder_output (baseline), 0.0 = pure B |
