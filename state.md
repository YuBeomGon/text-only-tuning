# state.md

## Current phase
- Phase: 0 (baseline_lock_and_plumbing)
- Owner: leader
- Last updated: 2026-04-06

## Current goal
Lock baseline metrics and validate interpolation plumbing.

## Baseline (Phase 0 frozen, never changes)
- model:
- eval set:
- CER:
- WER:
- domain term recall:
- hallucination rate:
- timestamp error:
- latency ms:

## Current best (updated only on PASS or reproduced REPRO_REQUIRED)
- run_id:
- config:
- alpha:
- prior_id:
- CER:
- WER:
- domain term recall:
- why it won:
- known regressions:

## Open risks
- short-utterance hallucination
- batch vs single-path divergence (serving transition)
- out-of-domain over-bias
- score thresholds are provisional (calibrate after Phase 0)

## Decisions made
- encoder_output interpolation is a proxy, not paper-faithful K/V interpolation
- B-only training (decoder freeze), conservative v1 choice
- HF Whisper for research eval, faster-whisper for serving (separate work)
- Score gate compares candidate vs baseline (viability) then vs current_best (promotion)
- No git revert on FAIL

## Next 3 actions
1.
2.
3.

## Blockers
-

## Notes
- Keep this file short and operational.
- After each completed run, update: current phase, current best, open risks, next 3 actions.
