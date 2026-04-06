# state.md

## Current phase
- Phase: baseline / fixed-prior sweep / tuned-prior eval
- Owner: leader
- Last updated: YYYY-MM-DD

## Current goal
Validate whether `encoder_output` interpolation with a domain prior improves domain ASR while keeping faster-whisper serving practical.

## Current baseline
- model:
- eval set:
- WER:
- domain term recall:
- hallucination rate:
- timestamp error:
- latency RTF:

## Best current run
- run_id:
- config:
- alpha:
- prior_id:
- why it won:
- known regressions:

## Open risks
- short-utterance hallucination
- timestamp mismatch between decode and alignment
- batch vs single-path divergence
- out-of-domain over-bias
- serving latency increase

## Decisions made
- encoder_output interpolation is treated as a proxy, not paper-faithful K/V interpolation
- single-path, batched-path, and word timestamp alignment must use the same mixed representation
- no promotion without fresh eval evidence and registry update

## Next 3 actions
1.
2.
3.

## Blockers
- 
- 

## Notes
- Keep this file short and operational.
- After each completed run, update:
  - current phase
  - best current run
  - open risks
  - next 3 actions
