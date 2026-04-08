# state.md

## Current phase
- Phase: 2 완료 → **방향 전환 (B 보간 → LoRA)**
- Owner: leader
- Last updated: 2026-04-08

## Current goal
B 보간 방식의 구조적 한계 확인 완료. LoRA adapter 기반 decoder fine-tune으로 전환한다.

## Baseline (Phase 0 frozen, never changes)
- model: openai/whisper-large-v3-turbo
- eval set: data/processed/eval_v1/manifest.jsonl (1,512 samples)
- CER: 0.128522
- WER: 0.355155
- domain term recall: 0.852226
- hallucination rate: 0.0
- latency ms: 177.23

## Current best
- run_id: baseline_v1
- CER: 0.128522 — B 보간 실험에서 이를 이기지 못함

## 실험 이력

### train_run_002 (FAIL) — B 보간, 306줄
- config: 1000 step, lr=3e-5, cosine, CE only
- alpha=0.9: CER 0.1577~0.1597 (+22.7~24.3%)
- alpha=0.7/0.5: decoder 붕괴 (CER 2.3~2.5)

### train_run_003 (FAIL) — B 보간, 2,437줄, EMA
- config: 5000 step, lr=3e-5, cosine, CE only, EMA decay=0.99
- 학습 데이터: 306→2,437줄 (8배 증가)
- alpha=0.9 sweep (step500~5000):

| checkpoint | CER    | delta vs baseline |
|------------|--------|-------------------|
| step500    | 0.1538 | +19.7%            |
| step1000   | 0.1539 | +19.7%            |
| step1500   | 0.1519 | +18.2% (best)     |
| step2000   | 0.1547 | +20.4%            |
| step2500   | 0.1533 | +19.3%            |
| step3000   | 0.1560 | +21.4%            |
| step3500   | 0.1565 | +21.8%            |
| step4000   | 0.1560 | +21.4%            |
| step4500   | 0.1558 | +21.2%            |
| step5000   | 0.1559 | +21.3%            |

- term_recall: 0.8514~0.8523 (baseline 수준 복귀)
- 개선: run_002 대비 CER 22.7→18.2% 악화로 소폭 개선, but baseline 미달
- 판정: FAIL

### B 보간 방식 최종 결론
- 데이터 8배 증가 + EMA 적용해도 alpha=0.9에서 CER +18% 악화
- step 늘려도 (50→5000) 결과 변화 미미
- **단일 고정 B로 encoder_output에 보간하는 방식은 구조적 한계**
- 원인: B는 모든 utterance에 동일한 고정 텐서 → utterance-specific representation 불가

### 논문 재분석
- 논문은 **decoder도 함께 fine-tune**한 것으로 판단 (B만 학습한 것이 아님)
- 일반 데이터셋(LibriSpeech 등)에서의 성능 변화를 검증하지 않음
- 도메인 데이터셋에서만 평가 (Earnings Call, OCW2, MedReport)

## Decisions made
- encoder_output 보간 방식 실험 종료 (2026-04-08)
- **LoRA adapter 전환 결정** — decoder cross-attention에 소량 파라미터 추가
- PEFT seq2seq는 음성 입력 미지원 → 오버라이딩 필요 (수동 작업)
- 기존 harness(score.sh, registry, sweep)는 그대로 활용

## Next 3 actions
1. **LoRA 학습 스크립트 작성** — PEFT LoraConfig + 음성 입력 오버라이딩
2. **LoRA 추론 파이프라인 수정** — base model + adapter 로드, B 보간 제거
3. **LoRA train_run_001 실행** — rank=8, decoder cross-attention target, 도메인 텍스트 2,437줄

## Open risks
- LoRA fine-tune 시 일반화 성능 저하 가능 (catastrophic forgetting)
- PEFT seq2seq 음성 입력 오버라이딩 복잡도
- rank 선택에 따른 성능/일반화 trade-off

## Notes
- B 보간 실험 코드/데이터는 보존 (비교 기준으로 유지)
- encoder_output 보간은 per-layer K/V 보간과 수학적으로 동치였으나, B-only training의 구조적 한계가 근본 원인
