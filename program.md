# Program: faster-whisper encoder_output interpolation harness

## Goal
Validate whether mixing a domain prior into `encoder_output` improves domain ASR quality while preserving faster-whisper serving practicality.

## Two work streams
This project has two distinct work streams that proceed independently:

1. **논문 구현 및 테스트**: encoder output 보간 방식의 text-only domain adaptation 실험 (fw-encoder-mix proxy)
2. **harness 자동화**: score gate, registry, subagent 순차 호출로 실험 루프를 자동 운영

Each stream must have its own completeness. Do not mix work across streams in the same task.

## Interpretation
This project uses a practical proxy of the paper idea:
- paper-faithful: interpolate decoder cross-attention K and V with learned bias priors
- this project: interpolate `encoder_output` before decode and reuse the same mixed representation for alignment

## Non-goals for v1
- Exact K/V-level reproduction inside CTranslate2
- MoE routing
- Dynamic alpha
- Multi-prior routing
- Simultaneous search over many architecture changes

## Primary success criteria
- Better WER than baseline on in-domain evaluation
- Better domain term recall than baseline
- No unacceptable hallucination increase
- No serious timestamp regression
- Acceptable serving overhead
- Reproducible results across at least two runs

## Core metrics
- WER
- domain_term_recall
- domain_term_precision if available
- hallucination_rate
- timestamp_error
- latency_ms or real_time_factor
- gpu_mem_mb

## Eval buckets
### A. in_domain_clean
Terminology-focused, relatively clean audio.

### B. in_domain_hard
Short utterances, noisy conditions, clipped boundaries, other failure-prone audio.

### C. out_of_domain_sanity
Check whether the prior over-biases decoding.

## Current hypothesis
A domain prior learned from text-only data and mixed into `encoder_output` with alpha in the range 0.7 to 0.9 can improve domain term recall with acceptable quality and latency trade-offs.

## Phase plan
### Phase 0: baseline lock
- freeze current serving path
- record baseline metrics
- verify single, batch, and timestamp behavior

### Phase 1: serving-only fixed prior
- no training yet
- this phase answers two separate questions:
  - **Question A (path safety)**: does prior injection break decode, timestamps, or batch parity?
    - any prior works, even random noise; alpha=0.9 with a throwaway prior is enough
  - **Question B (domain signal)**: does a domain-representative prior actually improve term recall?
    - requires a prior with domain content: encode representative domain audio through Whisper encoder, average the outputs
    - E_pretrained is NOT a domain prior; it is the paper's initialization target for trainable bias, not a finished domain representation
- run Question A first, then Question B only if A passes
- test alpha = 1.0, 0.9, 0.7, 0.5
- reject bad ideas cheaply

### Phase 2: small text-only fine-tune
- start with one domain prior
- small run only
- compare against Phase 1 best

### Phase 3: tuned prior serving eval
- same alpha sweep
- same eval buckets
- compare against baseline and fixed prior

### Phase 4: robustness
- repeat best run
- test out-of-domain sanity
- confirm timestamp and batch parity

## Immediate tasks
1. Implement `maybe_mix_with_prior()` in the serving path.
2. Ensure single decode, batched decode, and alignment all consume the same mixed representation.
3. Lock the baseline with raw artifacts and registry entry.
4. Run fixed-prior alpha sweep.
5. Promote only if score gate passes.

## Merge rule
No merge without:
- score.sh PASS
- registry.csv append
- saved raw metrics JSON
- updated state.md
- reproducibility check

## Revert rule
Revert or hold if any of the following happen:
- timestamp path diverges from decode path
- batch path differs materially from single path
- hallucination rises beyond threshold
- out-of-domain sanity collapses
- result cannot be reproduced
