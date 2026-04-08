# state.md

## Current phase
- Phase: 2 (hf_inference_interpolation) — alpha sweep 완료, 결과 분석 완료
- Owner: leader
- Last updated: 2026-04-08

## Current goal
train_run_002의 실패 원인을 개선하여 baseline 대비 CER/term_recall 향상을 달성한다.

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
- why it won: baseline (no prior) — train_run_002 전 checkpoint에서 악화 확인, baseline 유지
- known regressions: none

## train_run_002 결과 요약 (FAIL)
- config: 1000 step, lr=3e-5, cosine scheduler, grad_clip=1.0, CE loss only
- 학습 데이터: domain_text.txt 306줄
- prior 위치: priors/train_run_002/

### alpha=0.9 sweep (전 checkpoint)
| checkpoint | CER     | delta vs baseline |
|------------|---------|-------------------|
| step50     | 0.1580  | +22.9%            |
| step100    | 0.1580  | +22.9%            |
| step200    | 0.1597  | +24.3%            |
| step300    | 0.1597  | +24.2%            |
| step400    | 0.1596  | +24.1%            |
| step500    | 0.1596  | +24.2%            |
| step600    | 0.1579  | +22.9%            |
| step700    | 0.1577  | +22.7%            |
| step800    | 0.1578  | +22.7%            |
| step900    | 0.1578  | +22.8%            |
- term_recall: 0.8507~0.8516 (baseline 0.8522 대비 미세 악화)
- alpha=0.7: CER=2.48 (decoder 붕괴)
- alpha=0.5: CER=2.32 (decoder 붕괴)
- 판정: FAIL — 모든 alpha/checkpoint 조합에서 baseline 대비 악화

### 실패 원인 분석
1. **데이터 부족**: 306줄 vs 논문 최소 4,000줄 (13배 차이). B가 도메인 분포를 충분히 학습하지 못함.
2. **CE loss만 사용**: KL divergence(baseline 분포 제약)와 Bregman Divergence(도메인 용어 가중 패널티) 미사용. B가 baseline decoder 호환성을 유지하지 못하는 방향으로 학습됨.
3. **EMA 미적용**: B가 학습 중 급격히 변동하여, 추론 시 encoder_output과의 호환성 저하.
4. **모델 스케일 차이**: whisper-large-v3-turbo(1.5B)는 whisper-base(74M) 대비 encoder representation이 훨씬 복잡. 동일 규모의 데이터/loss로는 유효한 bias 학습이 어려움.

### 논문 대비 차이 정리
| 항목            | 논문                        | 우리                              |
|-----------------|-----------------------------|------------------------------------|
| 모델            | whisper-base                | whisper-large-v3-turbo             |
| 학습 데이터     | 4,000~24,000줄              | 306줄                              |
| Loss            | CE + KL + BD                | CE only                            |
| 보간 방식       | per-layer K/V               | encoder_output (수학적 동치)        |
| EMA             | 적용                        | 미적용                             |
| 결과            | WER 14~56% relative 개선    | 모든 경우 악화                      |

## Open risks
- batch vs single-path divergence (serving transition)
- out-of-domain over-bias
- score thresholds are provisional (calibrate after Phase 0)
- 데이터 확대 시 prompts_std.txt, augmented_prompts.txt의 도메인 적합성 검증 필요
- KL/BD loss 추가 시 하이퍼파라미터 튜닝 비용 증가
- EMA decay 값 선택에 추가 sweep 필요

## Decisions made
- encoder_output interpolation is a proxy, not paper-faithful K/V interpolation
- B-only training (decoder freeze), conservative v1 choice
- HF Whisper for research eval, faster-whisper for serving (separate work)
- Score gate compares candidate vs baseline (viability) then vs current_best (promotion)
- No git revert on FAIL
- hallucination detection: broadcast pattern only (방송 용어)
- train_run_002 FAIL 확정: 모든 alpha/checkpoint에서 baseline 대비 악화 (2026-04-08)

## Next 3 actions
1. **데이터 확대**: domain_text.txt(306) + prompts_std.txt(1,131) + augmented_prompts.txt(1,000) = 2,437줄 결합. 데이터 품질 점검 후 train_run_003 준비.
2. **Loss 확장**: CE + KL divergence 추가 구현. KL weight 초기값 설정. (BD는 KL 효과 확인 후 추가 검토)
3. **EMA 적용**: B_ema = decay * B_old + (1-decay) * B_updated. decay 초기값 0.999. 추론 시 B_ema 사용.

## Backlog (Next 3 이후)
- train_run_003: 위 3개 개선 적용 후 alpha sweep 재실행
- alpha=0.95 테스트: alpha=0.9에서 이미 악화이므로 더 보수적인 alpha 검토
- Bregman Divergence 추가: KL 효과 확인 후 도메인 용어 가중 패널티 추가
- 학습률 재검토: large 모델에 맞는 lr 탐색 (현재 3e-5)

## Blockers
- 없음 (데이터 파일 3종 모두 data/train/에 존재 확인됨)

## Notes
- Keep this file short and operational.
- After each completed run, update: current phase, current best, open risks, next 3 actions.
- encoder_output 보간은 per-layer K/V 보간과 수학적으로 동치 (Wk, Wv가 선형이므로). 이 방식 자체의 문제가 아니라 학습 조건(데이터, loss, EMA)의 문제로 판단.
