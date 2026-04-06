# Harness Guide for Whisper Text-Only Adaptation
## faster-whisper-compatible proxy using encoder_output interpolation

## 1) What this guide is for

This guide is for the practical version of the paper idea:

- **Paper-faithful version**: interpolate cross-attention **K,V** with learned bias priors.
- **Practical proxy version**: keep faster-whisper serving intact and **interpolate `encoder_output` before decoding**.

This guide assumes:

- you already have or can build the fine-tuning and eval code,
- you want a **harness-driven research workflow**,
- you want **single leader + sequential subagents**, not parallel agents,
- your immediate target is **fast iteration with reproducible gates**, not exact paper reproduction.

---

## 2) Why this approximation is reasonable

The paper trains Whisper by replacing cross-attention K and V with trainable bias embeddings `B`, then at inference combines audio-conditioned representations and learned priors through interpolation.

Your serving-friendly approximation is:

```text
encoder_output_mixed = alpha * encoder_output_audio + (1 - alpha) * prior
```

This differs from the paper in two ways:

1. **Interpolation point**: the paper interpolates projected K and V inside each cross-attention layer; this approach interpolates at the encoder_output level before any projection.
2. **Per-layer independence**: in the paper, each decoder layer receives an independently interpolated K/V (because each layer has its own W_k, W_v); in this approach, all decoder layers receive the same mixed input.

Despite these differences, it is a strong first proxy because:

- the paper initializes `B` from pretrained Whisper encoder representations,
- faster-whisper exposes `encoder_output` cleanly at the Python wrapper layer,
- you can test the usefulness of domain prior injection **without modifying CTranslate2 internals**.

Treat results from this path as:

- **proxy / approximation results**, not strict paper reproduction.

---

## 3) What the code structure implies for the harness

In the current `WhisperModel` flow, there are two key insertion points:

### single-path decode
- `generate_segments(...)`
  - `encoder_output = self.encode(segment)`
  - `generate_with_fallback(encoder_output, ...)`
  - if `word_timestamps=True`, the same `encoder_output` is passed to `add_word_timestamps(...)`

### batched decode
- `BatchedInferencePipeline.generate_segment_batched(...)`
  - `encoder_output = self.model.encode(features)`
  - `self.model.model.generate(encoder_output, prompts, ...)`
  - the same `encoder_output` is reused in timestamp alignment

That means the harness should treat **"mixed encoder_output generation"** as a first-class experiment axis, and it must require:

1. the **same mixed representation** be used for decode and alignment,
2. single-path and batched-path behavior stay consistent,
3. baseline and proxy runs remain comparable.

---

## 4) Core principle for this harness

Do **not** start from “how do I tune everything?”

Start from:

1. **Can I inject prior cleanly?**
2. **Can I measure whether it helps domain terms?**
3. **Does it increase hallucination or break timestamps?**
4. **Is it stable enough to justify fine-tuning and serving work?**

So the harness should be built as an **experiment control system**, not an agent swarm.

---

## 5) Recommended repo layout

```text
whisper-prior-project/
├─ CLAUDE.md
├─ program.md
├─ state.md
├─ registry.csv
├─ harness/
│  ├─ run.sh
│  ├─ run_once.sh
│  ├─ score.sh
│  ├─ compare.sh
│  └─ env.sh
├─ configs/
│  ├─ baseline.yaml
│  ├─ interp-alpha-0.9.yaml
│  ├─ interp-alpha-0.7.yaml
│  ├─ interp-alpha-0.5.yaml
│  ├─ interp-domain-finance.yaml
│  └─ interp-domain-insurance.yaml
├─ src/
│  ├─ serving/
│  │  ├─ transcribe_interp.py
│  │  ├─ prior_loader.py
│  │  └─ mix_encoder_output.py
│  ├─ train/
│  │  ├─ train_text_only.py
│  │  ├─ losses.py
│  │  └─ prior_builder.py
│  ├─ eval/
│  │  ├─ run_eval.py
│  │  ├─ metrics.py
│  │  └─ term_recall.py
│  └─ data/
│     ├─ build_domain_text.py
│     ├─ build_lexicon.py
│     └─ split_eval_sets.py
├─ prompts/
│  ├─ subagent_data.md
│  ├─ subagent_model.md
│  ├─ subagent_eval.md
│  ├─ subagent_serving.md
│  └─ subagent_repro.md
├─ datasets/
├─ priors/
├─ outputs/
│  ├─ runs/
│  ├─ scores/
│  └─ reports/
└─ notebooks/
```

---

## 6) Minimal harness roles for sequential subagents

Use **one leader session** that calls subagents sequentially.

### leader
Owns:
- experiment priority,
- run approval,
- merge / revert decisions,
- registry updates,
- final summaries.

### subagent: data
Owns:
- domain text normalization,
- lexicon creation,
- eval split hygiene,
- dataset cards.

### subagent: model
Owns:
- prior construction,
- interpolation logic,
- fine-tuning config,
- alpha exposure and safeguards.

### subagent: serving
Owns:
- faster-whisper patching,
- single/batch path parity,
- timestamp path consistency,
- latency / memory checks.

### subagent: eval
Owns:
- WER,
- domain term recall,
- hallucination metrics,
- timestamp consistency,
- score JSON generation.

### subagent: repro
Owns:
- seed control,
- config hashes,
- artifact naming,
- “can this run be reproduced exactly?” checks.

Important: do **not** let every subagent edit everything.
Each one should propose diffs in its own area first.

---

## 7) The experiment ladder

Use this exact ladder.

### Phase 0 — frozen baseline
Goal:
- establish reproducible baseline with current faster-whisper path.

Required outputs:
- baseline WER
- domain term recall
- hallucination metric
- timestamp metric
- throughput / latency
- config + git hash + eval set hash

No prior yet.

### Phase 1 — serving-only proxy injection
Goal:
- inject fixed prior into `encoder_output` **without any fine-tuning**.

Runs:
- `alpha=1.0` (baseline control)
- `alpha=0.9`
- `alpha=0.7`
- `alpha=0.5`

Purpose:
- measure whether prior injection alone helps or harms.

This is the cheapest way to reject bad ideas early.

### Phase 2 — text-only fine-tuning for prior
Goal:
- train prior / bias representations from domain text.

Start small:
- max 1,000 steps
- warmup 200
- eval every 50 steps
- save every 50 steps

Only after one training run shows signal should you widen the search.

### Phase 3 — tuned prior + serving proxy
Goal:
- use the trained prior in the faster-whisper interpolation path.

Again test:
- `alpha=0.9`
- `alpha=0.7`
- `alpha=0.5`

### Phase 4 — robustness
Goal:
- confirm gains survive:
  - different datasets,
  - different noise / chunking conditions,
  - long-form audio,
  - word timestamp mode,
  - batch mode.

### Phase 5 — promotion decision
Promote only if:
- domain-term gains are real,
- hallucination is controlled,
- serving overhead is acceptable,
- results reproduce twice.

---

## 8) What counts as a “run”

Each run should be one row in `registry.csv`.

Suggested columns:

```csv
run_id,date,git_sha,config_name,domain,alpha,prior_id,train_ckpt,eval_set,wer,term_recall,hallucination_rate,timestamp_err,rtf,gpu_mem,score,status,notes
```

Rules:

- no run without a config file,
- no score without a saved raw eval output,
- no merge based on screenshots or manual eyeballing,
- no “seems better” language in decisions.

---

## 9) What to score

WER alone is not enough.

## Primary metrics
- **WER**
- **domain term recall**
- **domain term precision** if possible
- **hallucination rate**

## Secondary metrics
- insertion rate
- deletion rate
- timestamp drift / alignment error
- no-speech false positive rate
- latency / real-time factor
- memory overhead

## Recommended gate logic
A run is promotable only if all are true:

- WER is better than baseline by meaningful margin
- domain term recall improves
- hallucination rate does not regress past threshold
- timestamp metric does not regress badly
- latency overhead is acceptable

Example score formula:

```text
score =
  + 4.0 * relative_wer_gain
  + 3.0 * term_recall_gain
  - 3.0 * hallucination_increase
  - 1.5 * timestamp_error_increase
  - 1.0 * latency_increase
```

Do not trust the formula alone.
Use it as a gate, not as the final judgment.

---

## 10) The most important experimental split

You need three eval buckets.

### A. in-domain clean
Checks whether prior helps target terminology.

### B. in-domain hard
Includes:
- short utterances,
- noisy phone conditions,
- speaker overlap if relevant,
- clipped starts / ends.

This bucket is where hallucination risk shows up.

### C. out-of-domain sanity
Checks whether prior over-biases decoding.

If your in-domain gain comes with heavy out-of-domain collapse, the serving path is not ready.

---

## 11) Harness decision policy

### promote
Only when:
- two runs reproduce,
- eval artifacts exist,
- score gate passes,
- no hidden timestamp regression.

### hold
When:
- WER improves but hallucination rises,
- gains exist only in one tiny split,
- batch path and single path disagree.

### revert
When:
- prior injection breaks timestamp mode,
- batch path differs materially from single path,
- alpha tuning gives unstable outcomes,
- eval cannot be reproduced.

---

## 12) What to put in `program.md`

`program.md` should be short and operational.

Suggested structure:

```markdown
# Goal
Validate whether encoder_output interpolation with domain prior improves ASR for domain X while preserving serving practicality in faster-whisper.

# Non-goals
- Exact K/V-level reproduction inside CTranslate2
- MoE routing in v1
- Multi-domain routing in v1

# Current phase
Phase 1 / Phase 2 / etc.

# Current best hypothesis
A domain prior trained from text-only data and mixed into encoder_output with alpha in [0.7, 0.9] improves term recall with acceptable hallucination cost.

# Immediate tasks
1. Baseline lock
2. Fixed-prior serving injection
3. Eval gate
4. Small text-only fine-tune
5. Re-eval

# Merge rule
No merge without score.sh pass and registry update.
```

---

## 13) What to put in `CLAUDE.md`

Keep it behavioral.

Suggested rules:

```markdown
- Work as a single leader coordinating sequential subagents.
- Do not run parallel agent fan-out unless explicitly requested.
- Prefer the smallest experiment that can falsify the current hypothesis.
- Never claim improvement without fresh eval evidence.
- Always update registry.csv and state.md after a completed run.
- Preserve baseline path.
- Keep single-path and batched-path behavior aligned.
- If word_timestamps is enabled, ensure decode and alignment use the same mixed encoder_output.
- Treat encoder_output interpolation as a proxy for the paper’s K/V interpolation, not an exact reproduction.
```

---

## 14) What to put in `state.md`

This is the scratchpad for the leader.

Template:

```markdown
# Current baseline
- model:
- dataset:
- baseline WER:
- baseline term recall:
- baseline hallucination:
- baseline timestamp error:

# Best current run
- run_id:
- config:
- why it won:
- risks:

# Open risks
- prior over-bias on short utterances
- timestamp mismatch
- batch/single divergence
- out-of-domain collapse

# Next 3 actions
1.
2.
3.
```

---

## 15) The serving patch policy

Your first implementation should add exactly one abstraction:

```python
encoder_output = self.encode(...)
encoder_output = maybe_mix_with_prior(encoder_output, prior, alpha, meta)
```

Do not start with:
- multiple priors,
- routing,
- dynamic alpha,
- domain classifier,
- user-specific personalization.

v1 should be boring.

### v1 serving requirements
- off switch exists
- alpha=1.0 exactly reproduces baseline path
- both single and batched decode use the same mixer logic
- word timestamp path uses the same mixed representation
- logs show prior_id and alpha for each run

---

## 16) The fine-tuning policy

Your train harness should not immediately chase full paper complexity.

### v1 train target
Learn one domain prior.

### v1 losses
Start with:
- CE
Optionally add later:
- KL
- domain-term penalty

### v1 model variants
1. fixed prior from precomputed encoder representation
2. trainable prior from text-only tuning
3. trained prior + interpolation during serving

Do not start with MoE.

---

## 17) The exact subagent sequence to use

For each milestone, call subagents in this order:

### milestone: baseline lock
1. data subagent
2. eval subagent
3. repro subagent
4. leader decision

### milestone: serving injection
1. serving subagent
2. eval subagent
3. repro subagent
4. leader decision

### milestone: text-only fine-tune
1. data subagent
2. model subagent
3. eval subagent
4. repro subagent
5. leader decision

### milestone: promotion
1. eval subagent
2. serving subagent
3. repro subagent
4. leader merge/revert

That is enough.
Do not over-orchestrate.

---

## 18) Suggested harness scripts

## `harness/run.sh`
High-level entrypoint.

Responsibilities:
- choose config
- create run_id
- export env
- call train or serve-eval path
- store outputs under `outputs/runs/$RUN_ID`

## `harness/run_once.sh`
One deterministic run.

Responsibilities:
- call one Python entrypoint
- write JSON metrics
- fail loudly on missing artifacts

## `harness/score.sh`
Gatekeeper.

Responsibilities:
- read metrics JSON
- compare against baseline
- emit:
  - PASS
  - HOLD
  - FAIL

## `harness/compare.sh`
Diff two runs.

Responsibilities:
- summarize metric deltas
- show important regressions first

---

## 19) Example score policy

```bash
# PASS if:
# - wer improves by >= 3% relative
# - term recall improves by >= 5% relative
# - hallucination increase <= 1% absolute
# - timestamp error increase <= 5%
# - latency increase <= 10%

# HOLD if:
# - wer improves but one safety metric regresses slightly

# FAIL otherwise
```

Tune later, but start crisp.

---

## 20) Recommended first 10 runs

1. baseline
2. fixed prior, alpha=0.9
3. fixed prior, alpha=0.7
4. fixed prior, alpha=0.5
5. repeat best of 2-4
6. small text-only tune, step 250
7. same prior, alpha=0.9
8. same prior, alpha=0.7
9. same prior, alpha=0.5
10. repeat best + out-of-domain sanity

This gives you signal before complexity.

---

## 21) What to avoid

Do not do these early:

- exact paper reproduction + serving patch at the same time
- MoE before single-prior works
- many alpha values before baseline is stable
- simultaneous loss search + serving patch search
- hand-picked cherry-picked examples as decision basis
- changing VAD/chunking/prompting while also evaluating prior injection

One moving part at a time.

---

## 22) My recommendation for your actual start

If you are starting this week, do this:

### Week 1
- lock baseline
- implement `maybe_mix_with_prior`
- add config-driven alpha
- validate single/batch/timestamp parity
- run fixed-prior alpha sweep

### Week 2
- build text-only prior training
- run 1 small tune
- evaluate tuned prior with same alpha sweep
- decide whether the proxy has enough signal to continue

### Promotion rule
Continue only if:
- term recall gain is real,
- hallucination is controlled,
- the serving patch stays simple.

If not, stop early and either:
- keep hotword/context methods,
- or move to a deeper runtime modification later.

---

## 23) Final recommendation

For your case, the right harness is **not** “many clever agents.”

It is:

- one leader,
- a few narrow sequential subagents,
- strict run registry,
- strong score gate,
- baseline preservation,
- minimal serving patch,
- fast rejection of bad ideas.

That is the cheapest way to learn whether **encoder_output interpolation** is worth pushing further in faster-whisper.
