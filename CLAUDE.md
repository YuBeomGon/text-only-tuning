# CLAUDE.md

## File roles
- **CLAUDE.md**: 에이전트 행동 규칙. 병렬 금지, 평가 없이 주장 금지, 경로 일관성 유지 등 상위 원칙.
- **program.md**: 현재 목표와 우선순위. 지금 단계가 baseline인지 fixed prior sweep인지 등 실행 계획서.
- **score.sh**: baseline vs candidate 비교 후 PASS / HOLD / FAIL 판정하는 게이트 스크립트.
- **registry.csv**: 모든 실험 run을 한 줄씩 기록하는 실험 장부. config, alpha, prior, 결과 누적.
- **config.yaml**: 한 번의 실험을 정의하는 설정 파일. 모델, eval set, alpha, prior 경로 등.
- **run.sh**: 실험 실행 메인 엔트리포인트. config를 받아 run_id 생성, run_once.sh → score.sh 순서 호출.
- **run_once.sh**: 실제 한 번의 deterministic run 수행. 훈련/eval 실행 후 metrics.json과 raw output 생성.
- **state.md**: 현재 프로젝트 상태 메모. best run, 남은 리스크, 다음 액션을 leader가 추적하는 용도.

## Working mode
- Work as a single leader coordinating sequential subagents.
- Do not use parallel fan-out unless explicitly requested.
- Prefer the smallest experiment that can falsify the current hypothesis.
- Preserve a clean baseline path at all times.
- Treat encoder_output interpolation as a faster-whisper-compatible proxy for the paper's K/V interpolation, not an exact reproduction.

## Repository operating rules
- Keep changes scoped to one experiment axis at a time.
- Do not change VAD, chunking, prompting, decoding thresholds, and prior interpolation in the same run unless the program explicitly says to.
- Keep single-path decode, batched decode, and word-timestamp alignment behavior consistent.
- If word_timestamps is enabled, ensure decode and alignment use the same mixed encoder_output.
- Maintain an off switch so alpha=1.0 reproduces baseline behavior.

## Evidence and evaluation
- Never claim improvement without fresh evaluation evidence.
- No merge without score.sh pass and registry.csv update.
- No decision based on screenshots or hand-picked examples alone.
- Report WER, domain term recall, hallucination rate, timestamp error, and serving overhead together.
- If a run improves WER but regresses hallucination or timestamps, mark it HOLD, not PASS.

## Subagent protocol
- Use narrow subagents sequentially: data, model, serving, eval, repro.
- Each subagent should edit its own area first and avoid wide cross-repo changes.
- After each subagent step, update state.md with what changed, what was measured, and what remains risky.

## Reproducibility
- Every run must have a config file, git SHA, seed, output directory, and saved raw metrics.
- Update registry.csv immediately after a completed run.
- Keep run IDs stable and machine-readable.
- If a result cannot be reproduced, do not promote it.

## Suggested milestone order
1. Baseline lock
2. Fixed prior serving injection with alpha sweep
3. Eval gate
4. Small text-only fine-tune
5. Re-eval with same alpha sweep
6. Promotion or revert
