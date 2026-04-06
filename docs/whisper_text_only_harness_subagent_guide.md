# Whisper Text-Only Fine-Tuning 프로젝트용 Harness 가이드

이 문서는 **"Domain-Specific Adaptation for ASR through Text-Only Fine-Tuning"** 논문을 실제 프로젝트로 옮길 때,
`quant.cpp` 스타일의 harness 철학을 가져오되 **멀티 에이전트 병렬 팀이 아니라 단일 리더 + 순차적 subagent 호출** 방식으로 운영하는 실전 가이드다.

핵심 목표는 다음 3가지다.

1. 논문 아이디어를 **재현 가능한 연구 코드**로 만든다.
2. 비용 폭증 없이 **작은 반복 루프**로 fine-tuning / eval / ablation을 굴린다.
3. 실험이 늘어나도 방향을 잃지 않도록 **score gate + phase gate**로 통제한다.

---

## 0. 먼저 결론: 이 논문에 Harness를 적용할 수 있나?

가능하다. 다만 형태를 바꿔야 한다.

`quant.cpp`에서는 harness의 중심이 **모듈 병렬화 + merge gate**였다면,
이 논문을 Whisper adaptation 프로젝트에 적용할 때 harness의 중심은 아래처럼 바뀐다.

- **병렬 코딩**보다 **실험 순서 통제**
- **파일 소유권 분리**보다 **실험 조건 분리**
- **merge conflict 방지**보다 **실험 오염 방지**
- **성능 벤치 게이트**보다 **WER / term accuracy / hallucination gate**

즉 이 프로젝트에서 harness의 본질은:

> baseline 고정 -> 가장 싼 실험부터 실행 -> 점수로 통과 여부 판정 -> 통과한 것만 다음 phase로 승격

이다.

그리고 비용 제약이 있으므로, 추천 구조는 **동시에 여러 worker를 띄우는 팀형 harness가 아니라, 한 번에 한 역할만 호출하는 subagent harness**다.

---

## 1. 논문에서 반드시 유지해야 할 핵심

프로젝트 시작 전에, 이 논문에서 무엇이 핵심인지 먼저 고정해야 한다.

### 1-1. adaptation 대상
이 논문은 **Whisper의 decoder만 fine-tuning**하고, 도메인 텍스트만 사용한다. 오디오-텍스트 pair 없이 domain adaptation을 노린다.

### 1-2. 구조 변경의 핵심
훈련 중 encoder는 사실상 학습 대상이 아니고, decoder cross-attention의 K/V 자리에 **trainable bias embeddings `B`** 를 넣는다. 이 bias는 pretrained Whisper encoder representation으로 초기화하고, 이후 trainable하게 둔다.

### 1-3. multi-domain 확장 포인트
단일 bias 하나가 아니라 **MoE 형태의 여러 expert bias** 와 lightweight router를 둔다.

### 1-4. inference 시 처리
추론 때는 실제 audio encoder output과 learned bias를 **선형 보간**해서 쓴다. 논문은 `alpha = 0.5`를 사용했다.

### 1-5. loss 확장
기본 cross-entropy에 더해 **KL divergence** 와 **domain-specific term penalty** 를 합친 loss를 쓴다.

### 1-6. 실험 운영 포인트
논문 실험은 Whisper-base English를 바탕으로, **최대 1,000 step / warmup 200 / eval every 50 / logging 10 / checkpoint 50** 구조로 굴렸다.

### 1-7. 기대와 한계
논문은 specialized vocabulary 측면에서는 의미 있는 개선을 보였지만,
OCW2 같은 acoustically complex 환경에서는 audio-conditioned 방식보다 불리했고,
짧은 구간에서는 hallucination이 발생할 수 있다고 직접 적고 있다.

따라서 프로젝트 목표는 처음부터 이렇게 잡는 것이 맞다.

> “보험 콜센터/전문 상담 도메인 용어 적응”에는 유망하다.
> “잡음/중첩화자/발화속도/억양 문제 해결”까지 한 번에 해결해 주는 방법은 아니다.

---

## 2. 이 프로젝트에서 Harness가 관리해야 하는 것

이 프로젝트의 harness는 코드를 많이 쓰는 도구가 아니라, 아래 6개를 통제하는 운영체계가 되어야 한다.

1. **실험 순서**
2. **기준선 고정**
3. **데이터 버전 고정**
4. **평가 지표 자동화**
5. **성공/실패 판정 자동화**
6. **비용 상한 관리**

즉, 이번 프로젝트의 `score.sh`는 “예쁘게 보고서 만들기”가 아니라,

- 이 실험을 계속할지
- 되돌릴지
- 다음 phase로 넘길지

를 결정하는 **gatekeeper** 여야 한다.

---

## 3. 추천 운영 모델: Single Leader + Sequential Subagents

멀티 에이전트 병렬 팀 대신, 아래처럼 **하나의 리더가 필요할 때마다 하나의 subagent만 호출**하는 구조를 권장한다.

### 리더
- 현재 phase를 결정
- 이번 라운드의 단 하나의 목표를 선택
- score를 확인
- 다음 subagent를 하나 호출
- 산출물을 merge
- eval 결과가 기준 미달이면 revert

### subagent 종류

#### A. `scorer`
역할:
- 현재 baseline과 최근 실험 결과를 읽음
- 어떤 metric이 가장 약한지 판정
- 다음 실험 우선순위를 제안

#### B. `data-prep`
역할:
- 텍스트 정제
- domain lexicon 생성
- tokenizer compatibility 점검
- train/val/test split manifest 생성

#### C. `model-surgery`
역할:
- Whisper decoder adaptation code 수정
- bias embedding / router / interpolation 경로 구현
- config surface 정리

#### D. `trainer`
역할:
- training script 수정
- checkpoint / logging / resume / seed / mixed precision 제어

#### E. `evaluator`
역할:
- existing eval code 실행
- WER / CER / term recall / hallucination 관련 출력 정리
- baseline 대비 delta 계산

#### F. `analyst`
역할:
- 실험 로그를 읽고 실패 원인 분류
- 다음 ablation 1개만 제안

#### G. `qa-repro`
역할:
- seed 고정 여부
- config drift
- checkpoint mismatch
- 데이터 누수
- evaluation contamination 검증

중요한 점은,
**동시에 3~4개를 부르는 것이 아니라 그때그때 1개만 부른다**는 것이다.

이렇게 해야:
- GPU 비용이 통제되고
- context 오염이 줄고
- 실패 원인 추적이 쉬워진다.

---

## 4. 저장소 구조 추천

아래처럼 시작하는 것을 권장한다.

```text
whisper-text-adapt/
├── CLAUDE.md
├── program.md
├── score.sh
├── harness/
│   ├── run.sh
│   └── phases.md
├── configs/
│   ├── base.yaml
│   ├── train_text_only.yaml
│   ├── ablation_bias_only.yaml
│   ├── ablation_moe.yaml
│   └── ablation_loss.yaml
├── data/
│   ├── manifests/
│   ├── lexicon/
│   └── README.md
├── src/
│   ├── models/
│   │   └── whisper_text_adapt.py
│   ├── train/
│   │   ├── train_text_only.py
│   │   └── losses.py
│   ├── eval/
│   │   └── run_eval.py
│   ├── analysis/
│   │   └── analyze_errors.py
│   └── utils/
├── experiments/
│   ├── registry.csv
│   └── notes/
├── reports/
│   ├── baseline.md
│   ├── ablations.md
│   └── final_report.md
└── tests/
```

### 파일 역할

#### `CLAUDE.md`
프로젝트 비전, 비목표, 데이터 경계, 점수 체계를 적는다.

#### `program.md`
“다음에 무엇을 할지”를 적는 현재 작업 명세다.

#### `score.sh`
실험 통과/실패를 판정하는 단일 진실 공급원이다.

#### `experiments/registry.csv`
모든 실험을 남긴다.
권장 컬럼:

```text
exp_id, parent_exp, phase, config, seed, data_version, code_commit,
train_steps, checkpoint, wer, cer, term_recall, hallucination_rate,
gpu_hours, status, notes
```

이 파일이 없으면 금방 실험이 섞이고 같은 실패를 반복하게 된다.

---

## 5. score.sh를 논문형 프로젝트에 맞게 바꾸는 법

`quant.cpp`의 score 철학은 그대로 가져오되, 지표는 ASR adaptation에 맞게 바꿔야 한다.

아래 5축을 추천한다.

## 5-1. Structure
- 필수 config 존재
- train script 존재
- eval script 존재
- experiment registry 존재
- baseline report 존재

## 5-2. Correctness
- dry-run이 성공하는가
- 1 step train sanity check가 되는가
- eval 코드가 baseline checkpoint에서 정상 동작하는가
- seed / config / manifest가 누락되지 않았는가

## 5-3. Quality
- baseline 대비 WER delta
- domain term recall
- rare term precision
- hallucination proxy
- short-segment robustness

## 5-4. Efficiency
- step time
- GPU memory usage
- total GPU hours
- checkpoint size 증가량
- inference latency 증가량

## 5-5. Research Completeness
- baseline 1개 확보
- reproduction run 확보
- bias-only ablation
- MoE ablation
- loss ablation
- inference interpolation ablation
- error analysis report

### 권장 판정 규칙

#### quick gate
아래를 만족해야 다음 실험으로 넘어간다.
- script crash 없음
- eval 완료
- registry 기록 완료
- baseline보다 catastrophic degradation 없음

#### quality gate
아래 둘 중 하나를 만족하면 통과:
- overall WER 개선
- overall WER은 비슷하지만 domain term recall이 유의미하게 개선

#### revert 조건
- WER 악화가 크고
- term recall 개선도 없고
- hallucination만 증가하면
즉시 revert 또는 freeze

핵심은:

> "loss가 돌아갔다"가 통과 조건이 아니라,
> "baseline보다 실제로 좋아졌는가"가 통과 조건이어야 한다.

---

## 6. Phase 설계: 이 순서로 시작하는 것이 좋다

이 프로젝트는 처음부터 full paper reproduction을 하지 말고, 아래 phase 순서로 가는 것이 안전하다.

---

## Phase 0. Baseline 고정

목표:
- 현재 Whisper baseline을 네 eval 코드로 확정
- 점수표와 registry를 만든다

산출물:
- `reports/baseline.md`
- baseline WER/CER
- domain term list v0
- `experiments/registry.csv` 첫 행

여기서 해야 할 일:
1. eval set을 고정
2. domain term lexicon 작성
3. short utterance subset 따로 분리
4. baseline checkpoint 결과 저장

이 phase를 대충 하면 뒤에서 모든 개선이 가짜가 된다.

---

## Phase 1. Minimal reproduction

목표:
- 가장 작은 형태의 논문 재현
- decoder-only + bias embedding만 먼저 구현

여기서는 아직 하지 말아야 할 것:
- MoE
- 복잡한 custom loss
- inference interpolation 튜닝 여러 개
- 데이터 증강 욕심

먼저 해야 할 것:
1. encoder freeze 확인
2. decoder cross-attention path 교체
3. bias `B` 삽입
4. pretrained encoder output 초기화
5. max 100~200 step sanity run

통과 기준:
- train이 안정적으로 돌아감
- eval 파이프라인 연결됨
- NaN / shape mismatch / checkpoint load 문제 없음

이 단계는 “성능 향상”보다 **구조가 맞게 들어갔는지**를 보는 단계다.

---

## Phase 2. Text pipeline 확정

목표:
- 텍스트만 쓰는 adaptation에서 데이터 품질을 고정

핵심 작업:
- normalization 규칙 확정
- tokenizer mismatch 점검
- domain vocabulary 추출 규칙 확정
- private/public text source 분리
- manifest 버전화

이 논문도 domain text normalization과 Whisper tokenizer compatibility를 명시하고 있다.

추천 추가 산출물:
- `data/README.md`
- `data/manifests/train_v1.jsonl`
- `data/lexicon/domain_terms_v1.txt`

여기서 가장 흔한 실패는 모델보다 데이터 쪽이다.

---

## Phase 3. Eval-first fine-tuning loop

목표:
- 실험을 “길게 1번”이 아니라 “짧게 여러 번” 돌린다.

권장 루프:
1. 100~200 step 짧은 run
2. eval
3. score gate 판정
4. 괜찮은 축만 500~1000 step으로 연장

논문 운영값을 그대로 시작점으로 삼는 것은 좋다.

- max steps: 1000
- warmup: 200
- eval every 50
- checkpoint every 50
- logging every 10

하지만 실제 프로젝트에선 처음부터 1000 step 풀런을 돌리기보다,
**100 step smoke run -> 300 step candidate -> 1000 step promoted run**
3단계로 나누는 것이 훨씬 싸다.

---

## Phase 4. Ablation ladder

이 단계에서만 논문 확장 요소를 하나씩 켠다.

순서 추천:
1. **bias-only**
2. **bias + interpolation**
3. **bias + KL**
4. **bias + domain-term penalty**
5. **bias + MoE**
6. **bias + MoE + custom loss**

왜 이 순서가 좋은가?

- 무엇이 진짜 효과인지 분리 가능
- compute를 한 번에 태우지 않음
- 실패 원인을 역추적하기 쉬움

반대로 아래 순서는 피하는 것이 좋다.

> bias + MoE + custom loss + 여러 alpha sweep

이렇게 하면 좋아져도 왜 좋아졌는지 모르고,
나빠져도 무엇이 문제인지 모른다.

---

## Phase 5. Inference realism

논문은 inference에서 encoder output과 learned bias를 보간한다.

이 단계에서 해야 할 일:
- `alpha=0.5` baseline 먼저 재현
- 그 다음에만 `alpha` sweep 수행
- short segment hallucination 점검
- prompt constraint / silence append 같은 post-processing을 실험

주의:
이 단계는 train 개선과 inference 트릭이 섞이기 쉬우므로,
**train 실험**과 **decode 실험**을 registry에서 반드시 분리해야 한다.

예시:
- `train_exp_id = T017`
- `decode_policy_id = D004`

---

## Phase 6. Research report packaging

최종적으로 남겨야 할 것:
- baseline 대비 개선표
- term-level error examples
- short-utterance hallucination 분석
- domain별 실패 유형
- 비용 대비 효과 표

여기까지 남겨야 다음 iteration에서 같은 길을 다시 안 걷는다.

---

## 7. subagent 운용 규칙

이 프로젝트에서 subagent는 “사람 흉내”가 아니라 “역할 고정 도구”처럼 써야 한다.

### 규칙 1. 한 번에 하나만 호출
동시에 여러 subagent를 띄우지 않는다.

### 규칙 2. 입력은 항상 좁게
좋은 예:
- “baseline eval 결과를 요약하고 가장 큰 오류 3가지만 뽑아라”
- “bias embedding 구현에서 바꿔야 할 함수만 찾아라”

나쁜 예:
- “논문 전체 구현해 줘”

### 규칙 3. 출력은 patch 또는 decision memo
subagent의 결과는 아래 둘 중 하나여야 한다.
- patch candidate
- 다음 행동을 위한 짧은 memo

### 규칙 4. merge 전에 scorer/evaluator를 통과
아무 subagent patch도 바로 mainline에 올리지 않는다.

### 규칙 5. 실험 1회 = 가설 1개
한 run에서 바꾸는 축은 하나만.

---

## 8. 추천 Subagent Playbook

아래처럼 역할별 프롬프트 템플릿을 정해 두면 좋다.

## 8-1. scorer

목적:
- 지금 상태에서 다음 한 수를 고르기

템플릿:

```text
Read experiments/registry.csv and the latest eval outputs.
Identify the single highest-leverage next experiment.
Choose only one change.
Return:
1) diagnosis
2) proposed experiment
3) expected upside
4) risk
5) pass/fail metric
```

## 8-2. data-prep

```text
Review the current domain text corpus and tokenizer behavior.
Find normalization or lexicon issues that would hurt domain term recognition.
Do not change training logic.
Return only:
- concrete data issues
- exact files to update
- validation checklist
```

## 8-3. model-surgery

```text
Implement only the decoder-side text-only adaptation path.
Do not touch evaluation or data manifests.
Keep encoder frozen.
Add the smallest possible change set for bias embeddings.
```

## 8-4. evaluator

```text
Run the existing evaluation pipeline on baseline and candidate checkpoints.
Compare WER, CER, term recall, and short-utterance behavior.
Return a decision memo: promote / hold / revert.
```

## 8-5. analyst

```text
Analyze the failure cases from the last experiment.
Cluster them into:
- domain term substitution
- insertion / hallucination
- deletion
- acoustics-related errors
Suggest exactly one next ablation.
```

---

## 9. 실전용 harness 루프

이 프로젝트의 실제 1라운드는 아래처럼 돌아가면 된다.

### Round 0
- baseline eval 실행
- registry 기록
- score 산출

### Round 1
- scorer 호출
- “가장 값싼 다음 실험 1개” 선정
- model-surgery 또는 data-prep 중 하나 호출
- 짧은 run 실행
- evaluator 호출
- score 비교
- 통과 시 registry에 promoted 표시
- 실패 시 revert

### Round 2 이후
동일 반복:

> score -> choose one hypothesis -> modify -> short run -> eval -> promote or revert

이게 바로 이번 프로젝트형 Karpathy loop다.

멀티 에이전트 팀처럼 동시에 많이 벌리지 않아도,
이 루프만 잘 지키면 오히려 비용 대비 효율이 좋다.

---

## 10. 첫 시작을 위한 현실적인 2주 계획

## Week 1

### Day 1
- repo skeleton 생성
- score.sh 골격 작성
- baseline eval 1회 완료

### Day 2
- domain lexicon v1 생성
- short utterance subset 분리
- registry / baseline report 작성

### Day 3
- minimal bias embedding path 구현
- 1 step / 10 step sanity run

### Day 4
- 100 step smoke run
- evaluator로 baseline 비교

### Day 5
- shape / loss / decode path 안정화
- reproduction memo 작성

## Week 2

### Day 6~7
- 300 step candidate run
- domain term 중심 error analysis

### Day 8
- interpolation on/off ablation

### Day 9
- KL on/off ablation

### Day 10
- domain-term penalty on/off ablation

### Day 11~12
- 가장 좋은 branch만 1000 step까지 연장

### Day 13~14
- final comparison table
- next decision: MoE 갈지 말지 판단

여기서 중요한 건,
**MoE는 2주 안에 꼭 들어가야 하는 기능이 아니라, minimal path가 실제로 먹히는지 확인한 뒤에만 들어가는 승격형 phase** 라는 점이다.

---

## 11. 이 프로젝트에서 특히 조심할 함정

### 함정 1. 텍스트 적응 성과를 음향 개선으로 착각
이 방법은 lexical prior 쪽에 강하고, acoustic robustness 자체를 해결하지는 못한다.

### 함정 2. full-run을 너무 빨리 돌림
처음부터 비싼 run을 돌리면 실패 원인만 비싸게 수집한다.

### 함정 3. ablation 여러 개를 한 번에 섞음
좋아져도 이유를 모르고, 나빠져도 원인을 모른다.

### 함정 4. term accuracy만 보고 hallucination을 놓침
논문도 short segment hallucination을 언급한다.
따라서 short-utterance slice를 따로 관리해야 한다.

### 함정 5. eval set drift
도메인 프로젝트일수록 텍스트와 오디오 split이 섞이기 쉽다.
manifest version을 반드시 남겨야 한다.

---

## 12. 최종 권장안

너의 경우엔 이렇게 시작하는 게 가장 좋다.

### 추천 전략
1. `quant.cpp`식 harness 철학을 그대로 가져온다.
2. 단, 구현 형태는 **멀티 팀형**이 아니라 **single leader + sequential subagents** 로 바꾼다.
3. 제일 먼저 `score.sh`, `registry.csv`, baseline eval부터 만든다.
4. 첫 구현은 **bias-only minimal reproduction** 으로 제한한다.
5. 그 다음에만 interpolation, KL, term penalty, MoE 순서로 승격한다.

### 한 줄 요약

> 이 논문은 harness에 잘 맞는다.
> 다만 코딩 병렬화용 harness가 아니라, “값싼 실험부터 하나씩 통과시키는 연구 운영 harness”로 바꿔서 써야 한다.

---

## 13. 바로 만들면 좋은 파일 4개

프로젝트 시작 직후 가장 먼저 만드는 것을 권장하는 파일:

1. `CLAUDE.md`
   - 프로젝트 목적 / 비목표 / 지표 / phase 규칙

2. `program.md`
   - 지금 당장 다음 1개 실험이 무엇인지

3. `score.sh`
   - baseline / candidate를 비교해 promote/revert 판단

4. `experiments/registry.csv`
   - 모든 실험 이력 저장

이 4개가 생기면, 나머지 training code는 훨씬 덜 흔들린다.

---

## 14. 다음 단계 제안

다음 작업은 이 순서가 가장 좋다.

1. 이 가이드 기준으로 `CLAUDE.md` 초안 작성
2. `score.sh` 초안 작성
3. `program.md` 초안 작성
4. repo skeleton 생성
5. baseline eval 등록

그 다음부터 subagent harness를 붙이면 된다.
