# state.md

## Current phase
- Phase: 0 (baseline_lock_and_plumbing)
- Owner: leader
- Last updated: 2026-04-06

## Current goal
Lock baseline metrics and validate interpolation plumbing.

## Baseline (Phase 0 frozen, never changes)
- model: openai/whisper-large-v3-turbo
- eval set: data/processed/eval_v1/manifest.jsonl (1,512 samples)
- CER: 0.128522
- WER: 0.355155
- domain term recall: 0.852226
- hallucination rate: 0.0
- timestamp error: 0.0
- latency ms: 177.23

## Current best (updated only on PASS or reproduced REPRO_REQUIRED)
- run_id: baseline_v1
- config: configs/baseline.yaml
- alpha: 1.0
- prior_id: none
- CER: 0.128522
- WER: 0.355155
- domain term recall: 0.852226
- why it won: baseline (no prior)
- known regressions: none

## Open risks
- batch vs single-path divergence (serving transition)
- out-of-domain over-bias
- score thresholds are provisional (calibrate after Phase 0)

## Decisions made
- encoder_output interpolation is a proxy, not paper-faithful K/V interpolation
- B-only training (decoder freeze), conservative v1 choice
- HF Whisper for research eval, faster-whisper for serving (separate work)
- Score gate compares candidate vs baseline (viability) then vs current_best (promotion)
- No git revert on FAIL
- hallucination detection: broadcast pattern only (방송 용어)

## Next 3 actions
1. Phase 0-b: plumbing validation (alpha=1.0 재현, B shape, alpha<1.0 crash test)
2. E_pretrained 추출 (도메인 오디오 필요)
3. Phase 1: text-only B training (CE loss, 100 step smoke run)

## Blockers
- E_pretrained 추출을 위한 도메인 오디오 샘플 필요

## Notes
- Keep this file short and operational.
- After each completed run, update: current phase, current best, open risks, next 3 actions.
