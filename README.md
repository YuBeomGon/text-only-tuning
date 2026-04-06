# Text-Only Domain Adaptation for Whisper

도메인 텍스트만으로 Whisper ASR의 도메인 용어 인식률을 개선하는 실험 하네스.

논문 ["Domain-Specific Adaptation for ASR through Text-Only Fine-Tuning"](docs/papers/Domain-Specific%20Adaptation%20for%20ASR%20through%20Text-Only%20Fine-Tuning.pdf)의 프록시 구현으로, 논문의 per-layer K/V interpolation 대신 **encoder output interpolation**을 사용한다.

## 핵심 아이디어

```
encoder_output_mixed = α · encoder_output + (1 - α) · B
```

1. **E_pretrained 추출**: 도메인 오디오를 Whisper 인코더에 통과시켜 평균 encoder output을 구한다.
2. **B 학습**: E_pretrained로 초기화한 B를 도메인 텍스트로 학습한다. 인코더/디코더는 동결하고 B만 학습 (CE loss).
3. **추론**: 학습된 B를 encoder output에 보간(interpolation)하여 도메인 용어 인식률을 높인다.

## 프로젝트 구조

```
├── src/
│   ├── extract_e_pretrained.py   # 도메인 오디오 → E_pretrained 추출
│   ├── train/
│   │   ├── train_text_only.py    # B 학습 (인코더/디코더 동결)
│   │   └── dataset.py            # 도메인 텍스트 데이터셋
│   ├── inference/
│   │   └── hf_encoder_mix.py     # encoder output 보간 추론
│   └── eval/
│       ├── run_eval.py           # 평가 파이프라인
│       └── metrics.py            # CER, WER, 도메인 용어 recall, 환각 탐지
├── harness/
│   ├── run.sh                    # 실험 실행 엔트리포인트
│   ├── run_once.sh               # 단일 deterministic run
│   └── score.sh                  # 2-stage score gate (viability → promotion)
├── configs/
│   └── baseline.yaml             # 실험 설정 템플릿
├── tests/                        # 테스트
├── docs/                         # 설계 문서, 논문
├── registry.csv                  # 실험 기록 장부
├── auto_policy.yaml              # 자동화 정책 (phase별 허용 범위)
├── program.md                    # 현재 목표와 우선순위
└── state.md                      # 현재 상태 추적
```

## 실험 파이프라인

### 1. E_pretrained 추출

```bash
python -m src.extract_e_pretrained \
    --model_name openai/whisper-large-v3-turbo \
    --audio_dir data/domain_audio_samples/ \
    --output_path priors/e_pretrained.pt
```

### 2. B 학습

```bash
python -m src.train.train_text_only \
    --model_name openai/whisper-large-v3-turbo \
    --e_pretrained_path priors/e_pretrained.pt \
    --text_file data/train/domain_text.txt \
    --output_dir priors/ \
    --max_steps 1000
```

### 3. 평가

```bash
python -m src.eval.run_eval \
    --model_name openai/whisper-large-v3-turbo \
    --manifest_path data/processed/eval_v1/manifest.jsonl \
    --prior_path priors/B_final.pt \
    --alpha 0.9 \
    --lexicon_path data/lexicon/domain_terms.txt \
    --output_json experiments/outputs/run_001/metrics.json
```

### 4. 하네스 자동 실행

```bash
bash harness/run.sh configs/baseline.yaml
```

`run.sh`가 run_id를 생성하고, `run_once.sh`로 평가 실행 후, `score.sh`로 baseline 대비 PASS/FAIL 판정을 자동 수행한다.

## Score Gate

2-stage 판정:

1. **Viability**: candidate가 baseline 대비 CER 또는 domain term recall 개선 + safety 통과
2. **Promotion**: candidate가 current_best 대비 CER 3%↑ AND term recall 5%↑ → PASS

판정 결과: `PASS` / `REPRO_REQUIRED` / `HUMAN_REVIEW` / `VIABLE` / `FAIL`

## 평가 메트릭

| 메트릭 | 설명 |
|--------|------|
| CER | Character Error Rate (한국어 primary) |
| WER | Word Error Rate (논문 비교용) |
| domain_term_recall | 도메인 용어 인식률 |
| hallucination_rate | 방송 용어 패턴 기반 환각 탐지 |
| latency_ms | 추론 지연 |

## 실험 Phase

| Phase | 내용 |
|-------|------|
| 0 | Baseline lock + plumbing validation |
| 1 | Text-only B 학습 (CE loss) |
| 2 | Alpha sweep (0.9, 0.7, 0.5) |
| 3 | Loss ablation (CE → CE+KL → CE+KL+term_penalty) |
| 4 | Robustness (재현, OOD, 환각 검증) |

## 의존성

- Python 3.10+
- PyTorch
- transformers (HuggingFace)
- jiwer
- soundfile, scipy
- PyYAML

## 현재 상태

- 모델: `openai/whisper-large-v3-turbo`
- 도메인: 보험 콜센터 (한국어)
- Baseline: CER=12.85%, WER=35.52%, domain_term_recall=85.22%
- Phase 1 진행 중 (100-step smoke run 완료, alpha sweep 진행 중)

## 논문 vs 본 프로젝트

| | 논문 | 본 프로젝트 |
|--|------|-------------|
| 보간 지점 | Decoder 내부 per-layer K/V | Encoder output (pre-layer) |
| 학습 대상 | Prior (+ decoder 가능) | B만 (encoder/decoder 동결) |
| 서빙 호환 | 커스텀 디코더 필요 | faster-whisper 호환 가능 |
