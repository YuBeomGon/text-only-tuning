# Whisper Text-Only Domain Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a harness-automated experiment loop that trains a domain bias B from text-only data and evaluates encoder_output interpolation on HF Whisper.

**Architecture:** Two streams — (1) training/inference/eval Python code using HF Whisper, (2) harness shell scripts + policy files that drive the automated experiment loop. Score gate uses two-stage comparison (viability vs baseline, promotion vs current_best).

**Tech Stack:** Python 3, PyTorch 2.8, HuggingFace Transformers 5.3 (WhisperForConditionalGeneration), jiwer 3.1, bash harness scripts, YAML configs.

**Spec:** `docs/superpowers/specs/2026-04-06-whisper-text-only-adaptation-design.md`

---

## File Structure

### New files to create

| File | Responsibility |
|------|---------------|
| `src/extract_e_pretrained.py` | Encode domain audio through Whisper encoder, average outputs, save as .pt |
| `src/train/train_text_only.py` | Train B with CE loss, encoder/decoder frozen, save checkpoints |
| `src/train/dataset.py` | Load domain text, format as Whisper multitask tokens |
| `src/inference/hf_encoder_mix.py` | Load Whisper + trained B, interpolate encoder_output, decode |
| `src/eval/run_eval.py` | Run inference on manifest, compute CER/WER/term_recall/hallucination |
| `src/eval/metrics.py` | Metric computation functions (CER, WER, term_recall, hallucination_rate) |
| `auto_policy.yaml` | Machine-readable automation policy |
| `harness/run.sh` | Experiment entrypoint (relocate from root) |
| `harness/run_once.sh` | Single deterministic run (relocate from root) |
| `harness/score.sh` | Score gate with two-stage comparison (rewrite from root) |
| `tests/test_metrics.py` | Unit tests for metric functions |
| `tests/test_encoder_mix.py` | Unit tests for interpolation logic |
| `tests/test_dataset.py` | Unit tests for dataset/tokenization |
| `tests/test_score_gate.py` | Unit tests for score gate decisions |

### Existing files to modify

| File | Change |
|------|--------|
| `CLAUDE.md` | Add auto_policy.yaml to file roles |
| `program.md` | Update phase numbering to match spec |
| `state.md` | Add baseline/current_best fields |
| `registry.csv` | Add VIABLE status column, align header |

---

## Task 1: Eval Metrics Library

**Files:**
- Create: `src/eval/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests for CER and WER**

```python
# tests/test_metrics.py
import pytest
from src.eval.metrics import compute_cer, compute_wer, compute_term_recall


def test_cer_identical():
    assert compute_cer("hello world", "hello world") == 0.0


def test_cer_one_char_diff():
    # "hello" vs "hallo" = 1 substitution / 5 chars = 0.2
    result = compute_cer("hello", "hallo")
    assert abs(result - 0.2) < 1e-6


def test_wer_identical():
    assert compute_wer("hello world", "hello world") == 0.0


def test_wer_one_word_diff():
    # "hello world" vs "hello earth" = 1 sub / 2 words = 0.5
    result = compute_wer("hello world", "hello earth")
    assert abs(result - 0.5) < 1e-6


def test_term_recall_all_found():
    ref = "paracetamol is used for pain"
    hyp = "paracetamol is used for pain"
    terms = ["paracetamol"]
    assert compute_term_recall(ref, hyp, terms) == 1.0


def test_term_recall_none_found():
    ref = "paracetamol is used for pain"
    hyp = "acetaminophen is used for pain"
    terms = ["paracetamol"]
    assert compute_term_recall(ref, hyp, terms) == 0.0


def test_term_recall_partial():
    ref = "broncol and paracetamol are drugs"
    hyp = "broncol and acetaminophen are drugs"
    terms = ["broncol", "paracetamol"]
    assert compute_term_recall(ref, hyp, terms) == 0.5


def test_term_recall_empty_terms():
    assert compute_term_recall("hello", "hello", []) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/MyProject/stt/tune/text-only && python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.metrics'`

- [ ] **Step 3: Implement metrics**

```python
# src/eval/metrics.py
from jiwer import cer as jiwer_cer, wer as jiwer_wer


def compute_cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer_cer(reference, hypothesis)


def compute_wer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer_wer(reference, hypothesis)


def compute_term_recall(
    reference: str, hypothesis: str, domain_terms: list[str]
) -> float:
    if not domain_terms:
        return 1.0
    ref_lower = reference.lower()
    hyp_lower = hypothesis.lower()
    hits = 0
    expected = 0
    for term in domain_terms:
        t = term.lower()
        if t in ref_lower:
            expected += 1
            if t in hyp_lower:
                hits += 1
    if expected == 0:
        return 1.0
    return hits / expected


def compute_hallucination_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if not hyp_words:
        return 0.0
    ref_set = set(ref_words)
    insertions = sum(1 for w in hyp_words if w not in ref_set)
    return insertions / len(hyp_words)
```

- [ ] **Step 4: Create `__init__.py` files**

```python
# src/__init__.py
# (empty)
```

```python
# src/eval/__init__.py
# (empty)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /data/MyProject/stt/tune/text-only && python -m pytest tests/test_metrics.py -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Commit**

```bash
cd /data/MyProject/stt/tune/text-only
git add src/__init__.py src/eval/__init__.py src/eval/metrics.py tests/test_metrics.py
git commit -m "feat: add eval metrics library (CER, WER, term_recall, hallucination)"
```

---

## Task 2: Encoder Output Interpolation

**Files:**
- Create: `src/inference/hf_encoder_mix.py`
- Create: `src/inference/__init__.py`
- Create: `tests/test_encoder_mix.py`

- [ ] **Step 1: Write failing tests for mix function**

```python
# tests/test_encoder_mix.py
import pytest
import torch
from src.inference.hf_encoder_mix import mix_encoder_output


def test_alpha_one_returns_original():
    original = torch.randn(1, 10, 512)
    prior = torch.randn(1, 10, 512)
    result = mix_encoder_output(original, prior, alpha=1.0)
    assert torch.allclose(result, original)


def test_alpha_zero_returns_prior():
    original = torch.randn(1, 10, 512)
    prior = torch.randn(1, 10, 512)
    result = mix_encoder_output(original, prior, alpha=0.0)
    assert torch.allclose(result, prior)


def test_alpha_half_is_mean():
    original = torch.ones(1, 10, 512) * 2.0
    prior = torch.zeros(1, 10, 512)
    result = mix_encoder_output(original, prior, alpha=0.5)
    assert torch.allclose(result, torch.ones(1, 10, 512))


def test_shape_mismatch_raises():
    original = torch.randn(1, 10, 512)
    prior = torch.randn(1, 5, 512)
    with pytest.raises(ValueError, match="shape"):
        mix_encoder_output(original, prior, alpha=0.5)


def test_output_shape_matches_input():
    original = torch.randn(1, 10, 512)
    prior = torch.randn(1, 10, 512)
    result = mix_encoder_output(original, prior, alpha=0.7)
    assert result.shape == original.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/MyProject/stt/tune/text-only && python -m pytest tests/test_encoder_mix.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement mix function**

```python
# src/inference/__init__.py
# (empty)
```

```python
# src/inference/hf_encoder_mix.py
import torch


def mix_encoder_output(
    encoder_output: torch.Tensor,
    prior: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if encoder_output.shape != prior.shape:
        raise ValueError(
            f"Encoder output shape {encoder_output.shape} does not match "
            f"prior shape {prior.shape}"
        )
    return alpha * encoder_output + (1.0 - alpha) * prior
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/MyProject/stt/tune/text-only && python -m pytest tests/test_encoder_mix.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /data/MyProject/stt/tune/text-only
git add src/inference/__init__.py src/inference/hf_encoder_mix.py tests/test_encoder_mix.py
git commit -m "feat: add encoder_output interpolation function with shape validation"
```

---

## Task 3: Domain Text Dataset

**Files:**
- Create: `src/train/__init__.py`
- Create: `src/train/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write failing tests for dataset**

```python
# tests/test_dataset.py
import pytest
from src.train.dataset import DomainTextDataset


def test_dataset_returns_input_ids():
    texts = ["This is a test sentence.", "Another sentence."]
    ds = DomainTextDataset(texts, model_name="openai/whisper-tiny")
    item = ds[0]
    assert "decoder_input_ids" in item
    assert "labels" in item


def test_dataset_length():
    texts = ["First.", "Second.", "Third."]
    ds = DomainTextDataset(texts, model_name="openai/whisper-tiny")
    assert len(ds) == 3


def test_dataset_labels_shifted():
    texts = ["Hello world."]
    ds = DomainTextDataset(texts, model_name="openai/whisper-tiny")
    item = ds[0]
    # labels should be shifted version of decoder_input_ids
    # decoder_input_ids starts with SOT, labels end with EOT
    assert item["labels"].shape[0] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/MyProject/stt/tune/text-only && python -m pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement dataset**

```python
# src/train/__init__.py
# (empty)
```

```python
# src/train/dataset.py
import torch
from torch.utils.data import Dataset
from transformers import WhisperTokenizer


class DomainTextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        model_name: str = "openai/whisper-base",
        language: str = "ko",
        task: str = "transcribe",
    ):
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name)
        self.texts = texts
        self.language = language
        self.task = task

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]

        # set_prefix_tokens configures the tokenizer so that encode()
        # automatically prepends [SOT, language, task, notimestamps]
        # and appends [EOT]. Do NOT manually prepend prefix_tokens.
        self.tokenizer.set_prefix_tokens(
            language=self.language, task=self.task
        )
        full_ids = self.tokenizer.encode(text)

        decoder_input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
        labels = torch.tensor(full_ids[1:], dtype=torch.long)

        return {
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/MyProject/stt/tune/text-only && python -m pytest tests/test_dataset.py -v`
Expected: All 3 tests PASS (may download whisper-tiny tokenizer on first run)

- [ ] **Step 5: Commit**

```bash
cd /data/MyProject/stt/tune/text-only
git add src/train/__init__.py src/train/dataset.py tests/test_dataset.py
git commit -m "feat: add domain text dataset with Whisper multitask token format"
```

---

## Task 4: E_pretrained Extraction Script

**Files:**
- Create: `src/extract_e_pretrained.py`

- [ ] **Step 1: Implement extraction script**

```python
# src/extract_e_pretrained.py
"""Extract E_pretrained by encoding domain audio through Whisper encoder.

Usage:
    python -m src.extract_e_pretrained \
        --model_name openai/whisper-base \
        --audio_dir data/domain_audio_samples/ \
        --output_path priors/e_pretrained.pt
"""
import argparse
from pathlib import Path

import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def extract_e_pretrained(
    model_name: str,
    audio_paths: list[Path],
    device: str = "cuda",
) -> torch.Tensor:
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    encoder_outputs = []
    with torch.no_grad():
        for audio_path in audio_paths:
            audio, sr = librosa.load(str(audio_path), sr=16000)
            inputs = processor(
                audio, sampling_rate=16000, return_tensors="pt"
            )
            input_features = inputs.input_features.to(device)
            enc_out = model.get_encoder()(input_features).last_hidden_state
            encoder_outputs.append(enc_out.cpu())

    # Average across all samples: (1, seq_len, d_model)
    stacked = torch.cat(encoder_outputs, dim=0)  # (N, seq_len, d_model)
    e_pretrained = stacked.mean(dim=0, keepdim=True)  # (1, seq_len, d_model)
    return e_pretrained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="openai/whisper-base")
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    audio_paths = sorted(
        p for p in audio_dir.iterdir()
        if p.suffix in {".wav", ".flac", ".mp3"}
    )
    if not audio_paths:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    print(f"Encoding {len(audio_paths)} audio files with {args.model_name}")
    e_pretrained = extract_e_pretrained(
        args.model_name, audio_paths, args.device
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(e_pretrained, output_path)
    print(f"Saved E_pretrained {e_pretrained.shape} to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script loads without error**

Run: `cd /data/MyProject/stt/tune/text-only && python -c "from src.extract_e_pretrained import extract_e_pretrained; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /data/MyProject/stt/tune/text-only
git add src/extract_e_pretrained.py
git commit -m "feat: add E_pretrained extraction script (domain audio -> mean encoder output)"
```

---

## Task 5: Text-Only B Training Script

**Files:**
- Create: `src/train/train_text_only.py`

- [ ] **Step 1: Implement training script**

```python
# src/train/train_text_only.py
"""Train bias B with text-only data. Encoder and decoder frozen, B is sole trainable param.

Usage:
    python -m src.train.train_text_only \
        --model_name openai/whisper-base \
        --e_pretrained_path priors/e_pretrained.pt \
        --text_file data/domain_text.txt \
        --output_dir priors/ \
        --max_steps 100 \
        --eval_every 50
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration

from src.train.dataset import DomainTextDataset


def collate_fn(batch: list[dict]) -> dict:
    max_len = max(item["decoder_input_ids"].shape[0] for item in batch)
    padded_input_ids = []
    padded_labels = []
    for item in batch:
        pad_len = max_len - item["decoder_input_ids"].shape[0]
        padded_input_ids.append(
            torch.nn.functional.pad(item["decoder_input_ids"], (0, pad_len))
        )
        padded_labels.append(
            torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100)
        )
    return {
        "decoder_input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
    }


def train(
    model_name: str,
    e_pretrained_path: str,
    text_file: str,
    output_dir: str,
    max_steps: int = 1000,
    save_every: int = 50,
    log_every: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    language: str = "ko",
    device: str = "cuda",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model, freeze everything
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    # Load E_pretrained and create trainable B
    e_pretrained = torch.load(e_pretrained_path, map_location=device)
    B = nn.Parameter(e_pretrained.clone().squeeze(0))  # (seq_len, d_model)

    # Load text data
    with open(text_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    dataset = DomainTextDataset(texts, model_name=model_name, language=language)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW([B], lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Training loop
    model.eval()
    step = 0
    log_records = []
    data_iter = iter(dataloader)

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        decoder_input_ids = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)
        bsz = decoder_input_ids.shape[0]

        # B replaces encoder_output: expand to batch size
        encoder_output = B.unsqueeze(0).expand(bsz, -1, -1)

        # Forward through decoder only
        decoder_out = model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_output,
        )
        logits = model.proj_out(decoder_out.last_hidden_state)

        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

        if step % log_every == 0:
            record = {"step": step, "loss": loss.item()}
            log_records.append(record)
            print(f"step={step} loss={loss.item():.4f}")

        if step % save_every == 0:
            ckpt_path = output_path / f"B_step{step}.pt"
            torch.save(B.data.unsqueeze(0), ckpt_path)

    # Save final B
    final_path = output_path / "B_final.pt"
    torch.save(B.data.unsqueeze(0), final_path)

    # Save training log
    log_path = output_path / "train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, indent=2)

    print(f"Training complete. B saved to {final_path}")
    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="openai/whisper-base")
    parser.add_argument("--e_pretrained_path", required=True)
    parser.add_argument("--text_file", required=True)
    parser.add_argument("--output_dir", default="priors/")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--language", default="ko")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script loads without error**

Run: `cd /data/MyProject/stt/tune/text-only && python -c "from src.train.train_text_only import train; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /data/MyProject/stt/tune/text-only
git add src/train/train_text_only.py
git commit -m "feat: add text-only B training script (encoder/decoder freeze, B-only trainable)"
```

---

## Task 6: HF Whisper Inference with Interpolation

**Files:**
- Modify: `src/inference/hf_encoder_mix.py` (add full inference function)

- [ ] **Step 1: Add transcribe_with_prior function**

```python
# src/inference/hf_encoder_mix.py (full file, replaces Task 2 version)
import torch
from pathlib import Path

import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


def mix_encoder_output(
    encoder_output: torch.Tensor,
    prior: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if encoder_output.shape != prior.shape:
        raise ValueError(
            f"Encoder output shape {encoder_output.shape} does not match "
            f"prior shape {prior.shape}"
        )
    return alpha * encoder_output + (1.0 - alpha) * prior


def load_model(model_name: str, device: str = "cuda"):
    """Load model and processor once. Reuse across multiple transcriptions."""
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, processor


def load_prior(prior_path: str | None, device: str = "cuda"):
    """Load prior tensor. Returns None if path is empty or None."""
    if not prior_path or not Path(prior_path).exists():
        return None
    return torch.load(prior_path, map_location=device)


def transcribe_single(
    audio_path: str,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    prior: torch.Tensor | None,
    alpha: float,
    language: str = "ko",
    device: str = "cuda",
) -> str:
    """Transcribe a single audio file. Model/processor/prior must be pre-loaded."""
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        encoder_output = model.get_encoder()(input_features).last_hidden_state

        if prior is not None and alpha < 1.0:
            encoder_output = mix_encoder_output(encoder_output, prior, alpha)

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_output)

        generated_ids = model.generate(
            encoder_outputs=encoder_outputs,
            language=language,
            task="transcribe",
        )

    transcription = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    return transcription
```

- [ ] **Step 2: Verify import works**

Run: `cd /data/MyProject/stt/tune/text-only && python -c "from src.inference.hf_encoder_mix import transcribe_with_prior; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /data/MyProject/stt/tune/text-only
git add src/inference/hf_encoder_mix.py
git commit -m "feat: add HF Whisper inference with encoder_output interpolation"
```

---

## Task 7: Eval Runner

**Files:**
- Create: `src/eval/run_eval.py`

- [ ] **Step 1: Implement eval runner**

```python
# src/eval/run_eval.py
"""Run evaluation on a manifest of audio files.

Usage:
    python -m src.eval.run_eval \
        --model_name openai/whisper-base \
        --manifest data/manifests/eval.jsonl \
        --prior_path priors/B_final.pt \
        --alpha 0.7 \
        --lexicon data/lexicon/domain_terms.txt \
        --output_json experiments/outputs/run_001/metrics.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from jiwer import cer as jiwer_cer, wer as jiwer_wer

from src.eval.metrics import (
    compute_term_recall,
    compute_hallucination_rate,
)
from src.inference.hf_encoder_mix import load_model, load_prior, transcribe_single


def load_manifest(manifest_path: str) -> list[dict]:
    """Load manifest.jsonl. Resolves audio paths relative to manifest's parent dir."""
    manifest_dir = Path(manifest_path).parent
    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                # Resolve relative audio path
                entry["audio_path"] = str(manifest_dir / entry["audio"])
                items.append(entry)
    return items


def load_lexicon(lexicon_path: str) -> list[str]:
    with open(lexicon_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run_eval(
    model_name: str,
    manifest_path: str,
    prior_path: str | None,
    alpha: float,
    lexicon_path: str | None,
    output_json: str,
    language: str = "ko",
    device: str = "cuda",
):
    items = load_manifest(manifest_path)
    domain_terms = load_lexicon(lexicon_path) if lexicon_path else []

    # Load model, processor, prior ONCE (not per file)
    model, processor = load_model(model_name, device)
    prior = load_prior(prior_path, device)

    all_refs = []
    all_hyps = []
    latencies = []

    for item in items:
        audio_path = item["audio_path"]  # already resolved by load_manifest
        reference = item["text"]

        t0 = time.time()
        hypothesis = transcribe_single(
            audio_path, model, processor, prior, alpha, language, device
        )
        elapsed_ms = (time.time() - t0) * 1000

        all_refs.append(reference)
        all_hyps.append(hypothesis)
        latencies.append(elapsed_ms)

    # Corpus-level CER/WER (total edit distance / total reference length)
    corpus_cer = jiwer_cer(all_refs, all_hyps)
    corpus_wer = jiwer_wer(all_refs, all_hyps)

    # Per-utterance average for term recall and hallucination
    total_term_recall = sum(
        compute_term_recall(r, h, domain_terms)
        for r, h in zip(all_refs, all_hyps)
    ) / len(all_refs)
    total_hallucination = sum(
        compute_hallucination_rate(r, h) for r, h in zip(all_refs, all_hyps)
    ) / len(all_refs)
    avg_latency = sum(latencies) / len(latencies)

    metrics = {
        "cer": round(corpus_cer, 6),
        "wer": round(corpus_wer, 6),
        "domain_term_recall": round(total_term_recall, 6),
        "hallucination_rate": round(total_hallucination, 6),
        "timestamp_error": 0.0,
        "latency_ms": round(avg_latency, 2),
        "n_samples": len(items),
        "alpha": alpha,
        "model_name": model_name,
        "prior_path": prior_path or "none",
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="openai/whisper-base")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prior_path", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lexicon", default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--language", default="ko")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_eval(
        args.model_name, args.manifest, args.prior_path, args.alpha,
        args.lexicon, args.output_json, args.language, args.device,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import works**

Run: `cd /data/MyProject/stt/tune/text-only && python -c "from src.eval.run_eval import run_eval; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /data/MyProject/stt/tune/text-only
git add src/eval/run_eval.py
git commit -m "feat: add eval runner (manifest-based CER/WER/term_recall/hallucination)"
```

---

## Task 8: Score Gate Rewrite (Two-Stage Comparison)

**Files:**
- Create: `harness/score.sh` (rewrite, relocate from root)
- Create: `tests/test_score_gate.py`

- [ ] **Step 1: Write failing tests for score gate logic**

```python
# tests/test_score_gate.py
import json
import subprocess
import tempfile
from pathlib import Path

SCORE_SH = str(Path(__file__).parent.parent / "harness" / "score.sh")


def _write_json(path, data):
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def _run_score(baseline, current_best, candidate, run_id="test"):
    with tempfile.TemporaryDirectory() as td:
        b = f"{td}/baseline.json"
        cb = f"{td}/current_best.json"
        c = f"{td}/candidate.json"
        _write_json(b, baseline)
        _write_json(cb, current_best)
        _write_json(c, candidate)
        result = subprocess.run(
            ["bash", SCORE_SH, b, cb, c, run_id],
            capture_output=True, text=True,
        )
        for line in result.stdout.strip().split("\n"):
            if line.startswith("DECISION="):
                return line.split("=", 1)[1]
    return "ERROR"


BASELINE = {
    "cer": 0.10, "domain_term_recall": 0.60,
    "hallucination_rate": 0.02, "timestamp_error": 0.08,
    "latency_ms": 500,
}


def test_pass():
    current_best = BASELINE.copy()
    candidate = {
        "cer": 0.09, "domain_term_recall": 0.65,
        "hallucination_rate": 0.02, "timestamp_error": 0.08,
        "latency_ms": 510,
    }
    assert _run_score(BASELINE, current_best, candidate) == "PASS"


def test_fail_worse_than_baseline():
    candidate = {
        "cer": 0.12, "domain_term_recall": 0.55,
        "hallucination_rate": 0.03, "timestamp_error": 0.10,
        "latency_ms": 600,
    }
    assert _run_score(BASELINE, BASELINE, candidate) == "FAIL"


def test_viable_beats_baseline_not_best():
    current_best = {
        "cer": 0.08, "domain_term_recall": 0.70,
        "hallucination_rate": 0.02, "timestamp_error": 0.08,
        "latency_ms": 500,
    }
    candidate = {
        "cer": 0.09, "domain_term_recall": 0.65,
        "hallucination_rate": 0.02, "timestamp_error": 0.08,
        "latency_ms": 510,
    }
    assert _run_score(BASELINE, current_best, candidate) == "VIABLE"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/MyProject/stt/tune/text-only && python -m pytest tests/test_score_gate.py -v`
Expected: FAIL — score.sh doesn't exist at `harness/score.sh` yet

- [ ] **Step 3: Create harness directory and rewrite score.sh**

```bash
#!/usr/bin/env bash
set -eu

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 BASELINE_JSON CURRENT_BEST_JSON CANDIDATE_JSON [RUN_ID]" >&2
  exit 2
fi

BASELINE_JSON="$1"
CURRENT_BEST_JSON="$2"
CANDIDATE_JSON="$3"
RUN_ID="${4:-unknown}"

for f in "$BASELINE_JSON" "$CURRENT_BEST_JSON" "$CANDIDATE_JSON"; do
  if [ ! -f "$f" ]; then
    echo "Missing json: $f" >&2
    exit 2
  fi
done

export CER_REL_PASS="${CER_REL_PASS:-0.03}"
export TERM_RECALL_REL_PASS="${TERM_RECALL_REL_PASS:-0.05}"
export HALLUCINATION_ABS_MAX="${HALLUCINATION_ABS_MAX:-0.01}"
export TIMESTAMP_REL_MAX="${TIMESTAMP_REL_MAX:-0.05}"
export LATENCY_REL_MAX="${LATENCY_REL_MAX:-0.10}"
export CER_HOLD_FLOOR="${CER_HOLD_FLOOR:--0.02}"
export TERM_HOLD_FLOOR="${TERM_HOLD_FLOOR:--0.03}"

python3 - "$BASELINE_JSON" "$CURRENT_BEST_JSON" "$CANDIDATE_JSON" "$RUN_ID" <<'PY'
import json, os, sys
from pathlib import Path

baseline = json.loads(Path(sys.argv[1]).read_text("utf-8"))
current_best = json.loads(Path(sys.argv[2]).read_text("utf-8"))
candidate = json.loads(Path(sys.argv[3]).read_text("utf-8"))
run_id = sys.argv[4]

def f(x): return float(x)
def rel_gain(old, new):
    """Relative gain where lower is better (CER, error rates). Positive = improvement."""
    if old == 0: return 0.0
    return (old - new) / old
def rel_gain_higher(old, new):
    """Relative gain where higher is better (recall). Positive = improvement."""
    if old == 0: return 1.0 if new > 0 else 0.0
    return (new - old) / old

# --- Stage 1: Viability check (candidate vs baseline) ---
bl_cer_gain = rel_gain(f(baseline["cer"]), f(candidate["cer"]))
bl_term_gain = rel_gain_higher(f(baseline["domain_term_recall"]), f(candidate["domain_term_recall"]))

# Safety: absolute and relative checks
c_hall = f(candidate["hallucination_rate"])
hall_abs_max = float(os.environ["HALLUCINATION_ABS_MAX"])
ts_increase = (f(candidate["timestamp_error"]) - f(baseline["timestamp_error"])) / f(baseline["timestamp_error"]) if f(baseline["timestamp_error"]) else 0.0
lat_increase = (f(candidate["latency_ms"]) - f(baseline["latency_ms"])) / f(baseline["latency_ms"]) if f(baseline["latency_ms"]) else 0.0

viable = bl_cer_gain > 0 or bl_term_gain > 0
safety_ok = (
    c_hall <= hall_abs_max
    and ts_increase <= float(os.environ["TIMESTAMP_REL_MAX"])
    and lat_increase <= float(os.environ["LATENCY_REL_MAX"])
)

# HUMAN_REVIEW check BEFORE PASS (higher priority than promotion)
# Triggers when any safety metric is near boundary (within 80% of threshold)
safety_borderline = (
    c_hall > hall_abs_max * 0.8
    or ts_increase > float(os.environ["TIMESTAMP_REL_MAX"]) * 0.8
    or lat_increase > float(os.environ["LATENCY_REL_MAX"]) * 0.8
)

cb_cer_gain = 0.0
cb_term_gain = 0.0

if not viable or not safety_ok:
    decision = "FAIL"
elif safety_borderline:
    decision = "HUMAN_REVIEW"
else:
    # --- Stage 2: Promotion check (candidate vs current_best) ---
    cb_cer_gain = rel_gain(f(current_best["cer"]), f(candidate["cer"]))
    cb_term_gain = rel_gain_higher(f(current_best["domain_term_recall"]), f(candidate["domain_term_recall"]))

    cer_passes = cb_cer_gain >= float(os.environ["CER_REL_PASS"])
    term_passes = cb_term_gain >= float(os.environ["TERM_RECALL_REL_PASS"])
    cer_acceptable = cb_cer_gain >= float(os.environ["CER_HOLD_FLOOR"])
    term_acceptable = cb_term_gain >= float(os.environ["TERM_HOLD_FLOOR"])

    if cer_passes and term_passes:
        decision = "PASS"
    elif (cer_passes and term_acceptable) or (term_passes and cer_acceptable):
        decision = "REPRO_REQUIRED"
    else:
        decision = "VIABLE"

summary = {
    "run_id": run_id,
    "decision": decision,
    "viability": {"cer_gain_vs_baseline": round(bl_cer_gain, 6), "term_gain_vs_baseline": round(bl_term_gain, 6)},
    "promotion": {"cer_gain_vs_best": round(cb_cer_gain, 6), "term_gain_vs_best": round(cb_term_gain, 6)},
    "safety": {"hall_abs": round(c_hall, 6), "ts_increase": round(ts_increase, 6), "lat_increase": round(lat_increase, 6)},
}
print(f"DECISION={decision}")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
```

- [ ] **Step 4: Make executable and run tests**

Run:
```bash
cd /data/MyProject/stt/tune/text-only
mkdir -p harness
# (write score.sh to harness/score.sh)
chmod +x harness/score.sh
python -m pytest tests/test_score_gate.py -v
```
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /data/MyProject/stt/tune/text-only
git add harness/score.sh tests/test_score_gate.py
git commit -m "feat: rewrite score gate with two-stage comparison (viability + promotion)"
```

---

## Task 9: Harness Scripts (run.sh, run_once.sh) Relocation

**Files:**
- Create: `harness/run.sh` (relocate + update paths)
- Create: `harness/run_once.sh` (relocate + update paths)

- [ ] **Step 1: Create harness/run_once.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config.yaml> <run_dir>"
  exit 1
fi

CONFIG_PATH="$1"
RUN_DIR="$2"
mkdir -p "$RUN_DIR"

LOG_FILE="${RUN_DIR}/run.log"
echo "[INFO] Starting run" | tee "$LOG_FILE"
echo "[INFO] Config: ${CONFIG_PATH}" | tee -a "$LOG_FILE"

ALPHA=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['prior']['alpha'])")
PRIOR_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['prior'].get('prior_path', 'none'))")
MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['serving']['model_path'])")
MANIFEST=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_PATH')); print(c['datasets']['eval']['in_domain_clean'])")
LEXICON=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_PATH')); print(c['domain'].get('lexicon_path', ''))")
LANGUAGE=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['domain'].get('language', 'ko'))")

PRIOR_ARG=""
if [[ "$PRIOR_PATH" != "none" ]] && [[ -f "$PRIOR_PATH" ]]; then
  PRIOR_ARG="--prior_path $PRIOR_PATH"
fi

LEXICON_ARG=""
if [[ -n "$LEXICON" ]] && [[ -f "$LEXICON" ]]; then
  LEXICON_ARG="--lexicon $LEXICON"
fi

python -m src.eval.run_eval \
  --model_name "$MODEL_NAME" \
  --manifest "$MANIFEST" \
  $PRIOR_ARG \
  --alpha "$ALPHA" \
  $LEXICON_ARG \
  --output_json "${RUN_DIR}/metrics.json" \
  --language "$LANGUAGE" \
  2>&1 | tee -a "$LOG_FILE"

echo "[INFO] Completed run_once" | tee -a "$LOG_FILE"
```

- [ ] **Step 2: Create harness/run.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [baseline_metrics.json] [current_best_metrics.json]"
  exit 1
fi

CONFIG_PATH="$1"
BASELINE_METRICS="${2:-experiments/outputs/baseline/metrics.json}"
CURRENT_BEST_METRICS="${3:-$BASELINE_METRICS}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH"
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
CONFIG_NAME="$(basename "$CONFIG_PATH" .yaml)"
RUN_ID="${TIMESTAMP}_${CONFIG_NAME}_${GIT_SHA}"
RUN_DIR="experiments/outputs/${RUN_ID}"
mkdir -p "$RUN_DIR"

echo "[INFO] RUN_ID=${RUN_ID}"
echo "[INFO] RUN_DIR=${RUN_DIR}"

cp "$CONFIG_PATH" "${RUN_DIR}/config.snapshot.yaml"

bash harness/run_once.sh "$CONFIG_PATH" "$RUN_DIR"

if [[ ! -f "${RUN_DIR}/metrics.json" ]]; then
  echo "[ERROR] metrics.json not produced"
  exit 1
fi

bash harness/score.sh "$BASELINE_METRICS" "$CURRENT_BEST_METRICS" "${RUN_DIR}/metrics.json" "$RUN_ID" \
  | tee "${RUN_DIR}/score.txt"

echo "[INFO] Finished: ${RUN_ID}"
echo "[INFO] Artifacts: ${RUN_DIR}/"
```

- [ ] **Step 3: Make executable**

```bash
chmod +x harness/run.sh harness/run_once.sh
```

- [ ] **Step 4: Commit**

```bash
cd /data/MyProject/stt/tune/text-only
git add harness/run.sh harness/run_once.sh
git commit -m "feat: relocate harness scripts to harness/ with two-stage score gate integration"
```

---

## Task 10: auto_policy.yaml + Meta File Updates

**Files:**
- Create: `auto_policy.yaml`
- Modify: `CLAUDE.md`
- Modify: `state.md`
- Modify: `registry.csv`

- [ ] **Step 1: Create auto_policy.yaml**

```yaml
# auto_policy.yaml — machine-readable automation policy
# The leader agent must only generate experiments within these constraints.

current_phase: 0

phase_config:
  0:
    goal: baseline_lock_and_plumbing
    steps:
      - baseline eval with alpha=1.0
      - verify alpha=1.0 reproduces baseline exactly
      - verify B shape matches inference path
      - verify alpha<1.0 decode runs without crash
    auto_generate_next: false
    phase_transition_requires_human: true
  1:
    goal: text_only_b_training
    allowed_steps: [100, 300, 1000]
    loss: ce_only
    auto_generate_next: true
    phase_transition_requires_human: true
  2:
    goal: hf_inference_interpolation
    allowed_alphas: [0.9, 0.7, 0.5]
    auto_generate_next: true
    phase_transition_requires_human: true
  3:
    goal: loss_ablation
    ablation_order: [ce, ce_kl, ce_kl_term]
    auto_generate_next: true
    phase_transition_requires_human: true
  4:
    goal: robustness
    auto_generate_next: false
    phase_transition_requires_human: true

loop_control:
  initial_human_review_count: 5
  max_consecutive_fails: 3

score_thresholds_provisional: true
score_thresholds_note: "Calibrate after Phase 0 baseline variance measurement (2-3 runs)"
```

- [ ] **Step 2: Add auto_policy.yaml to CLAUDE.md file roles**

Add to the File roles section in CLAUDE.md:
```
- **auto_policy.yaml**: machine-readable automation policy. Defines current phase, allowed parameters, phase transition rules, loop control.
```

- [ ] **Step 3: Update registry.csv header**

```csv
run_id,date,git_sha,config_name,phase,domain,alpha,prior_id,train_ckpt,eval_set,cer,wer,domain_term_recall,domain_term_precision,hallucination_rate,timestamp_error,latency_ms,rtf,gpu_mem_mb,score_decision,status,notes
```

- [ ] **Step 4: Update state.md with comparison fields**

```markdown
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
```

- [ ] **Step 5: Commit**

```bash
cd /data/MyProject/stt/tune/text-only
git add auto_policy.yaml CLAUDE.md state.md registry.csv
git commit -m "feat: add auto_policy.yaml, update meta files for two-stage score gate"
```

---

## Task 11: Cleanup Root-Level Scripts

**Files:**
- Delete: `run.sh` (root, replaced by `harness/run.sh`)
- Delete: `run_once.sh` (root, replaced by `harness/run_once.sh`)
- Delete: `score.sh` (root, replaced by `harness/score.sh`)
- Delete: `config.yaml` (root, will be generated per-experiment in `configs/`)

- [ ] **Step 1: Remove root-level scripts**

```bash
cd /data/MyProject/stt/tune/text-only
git rm run.sh run_once.sh score.sh config.yaml
```

- [ ] **Step 2: Create configs directory with example**

```yaml
# configs/baseline.yaml
run:
  run_id: baseline_v1
  seed: 42
  save_raw_predictions: true
  save_metrics_json: true

domain:
  name: insurance_callcenter
  language: ko
  lexicon_path: data/lexicon/domain_terms.txt

datasets:
  eval:
    in_domain_clean: data/processed/eval_v1/manifest.jsonl

serving:
  model_path: openai/whisper-base

prior:
  enabled: false
  prior_path: ""
  alpha: 1.0
```

- [ ] **Step 3: Commit**

```bash
cd /data/MyProject/stt/tune/text-only
mkdir -p configs
git add configs/baseline.yaml
git commit -m "chore: relocate harness scripts to harness/, remove root-level duplicates"
```

---

## Task 12: Integration Smoke Test

**Files:** No new files. Validates everything works together.

- [ ] **Step 1: Create minimal test fixtures**

```bash
cd /data/MyProject/stt/tune/text-only
mkdir -p data/manifests data/lexicon tests/fixtures
```

```python
# tests/create_fixtures.py
"""Create minimal test fixtures for integration testing."""
import json
import numpy as np
import soundfile as sf
from pathlib import Path

fixtures = Path("tests/fixtures")
audio_dir = fixtures / "audio" / "test_batch"
audio_dir.mkdir(parents=True, exist_ok=True)

# Create a 2-second silence wav
sr = 16000
silence = np.zeros(sr * 2, dtype=np.float32)
sf.write(str(audio_dir / "test_0000.wav"), silence, sr)

# Create a minimal manifest (matching real format)
manifest = [
    {
        "id": "test_0000",
        "audio": "audio/test_batch/test_0000.wav",
        "text": "test sentence",
        "duration_sec": 2.0,
        "batch": "test_batch",
        "source_id": "test",
        "start_sec": 0.0,
        "end_sec": 2.0,
    }
]
with open(fixtures / "test_manifest.jsonl", "w") as f:
    for item in manifest:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Create a minimal lexicon
with open(fixtures / "test_lexicon.txt", "w") as f:
    f.write("test\n")

print("Fixtures created.")
```

Run: `cd /data/MyProject/stt/tune/text-only && python tests/create_fixtures.py`

- [ ] **Step 2: Run eval on test fixture (baseline, alpha=1.0)**

```bash
cd /data/MyProject/stt/tune/text-only
python -m src.eval.run_eval \
  --model_name openai/whisper-tiny \
  --manifest tests/fixtures/test_manifest.jsonl \
  --alpha 1.0 \
  --lexicon tests/fixtures/test_lexicon.txt \
  --output_json tests/fixtures/baseline_metrics.json \
  --device cpu
```

Expected: `metrics.json` produced with CER, WER, domain_term_recall, hallucination_rate, timestamp_error, latency_ms fields.

- [ ] **Step 3: Run all unit tests**

Run: `cd /data/MyProject/stt/tune/text-only && python -m pytest tests/ -v --ignore=tests/create_fixtures.py`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
cd /data/MyProject/stt/tune/text-only
git add tests/create_fixtures.py tests/fixtures/
git commit -m "test: add integration smoke test with minimal fixtures"
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - [x] B initialization from domain audio (Task 4)
   - [x] B training with CE loss, encoder/decoder freeze (Task 5)
   - [x] Encoder output interpolation (Task 2, 6)
   - [x] Eval metrics: CER, WER, term_recall, hallucination (Task 1)
   - [x] Eval runner on manifest (Task 7)
   - [x] Two-stage score gate (viability + promotion) (Task 8)
   - [x] auto_policy.yaml (Task 10)
   - [x] File relocation (Task 9, 11)
   - [x] Plumbing validation in Phase 0 (Task 12 smoke test)
   - [x] VIABLE status in score gate (Task 8)
   - [x] Provisional thresholds noted (Task 10)

2. **Placeholder scan:** No TBD/TODO in code. Manifest format is marked TODO in spec but eval code handles jsonl.

3. **Type consistency:** `mix_encoder_output` signature consistent across Task 2 and Task 6. `compute_*` function names consistent across Task 1 and Task 7. Task 6 exports `load_model`, `load_prior`, `transcribe_single` which Task 7 imports.

4. **Code review fixes applied:**
   - [x] C1: dataset.py prefix token duplication removed (Task 3)
   - [x] C2: transcribe_with_prior split into load_model/load_prior/transcribe_single; prior=None handled (Task 6)
   - [x] C3: run_eval.py loads model/processor/prior once outside loop (Task 7)
   - [x] I1: score.sh checks hallucination as absolute value, not delta (Task 8)
   - [x] I2: score.sh dead code removed, clean rel_gain/rel_gain_higher functions (Task 8)
   - [x] I3: warmup_steps/eval_every removed from train_text_only.py (Task 5)
   - [x] I4: run_eval.py uses corpus-level jiwer_cer/jiwer_wer (Task 7)
   - [x] I5: HUMAN_REVIEW checked before PASS/REPRO_REQUIRED with all safety metrics (Task 8)
   - [x] M3: test count corrected to 7 (Task 1)

---

## Phase 전환: B 보간 → LoRA (2026-04-08)

### B 보간 실험 결과 요약

| run | 데이터 | 설정 | best CER (α=0.9) | baseline 대비 |
|-----|--------|------|-------------------|--------------|
| train_run_002 | 306줄 | CE, no EMA | 0.1577 | +22.7% |
| train_run_003 | 2,437줄 | CE, EMA 0.99 | 0.1519 | +18.2% |

- 데이터 8배 증가 + EMA 적용해도 baseline(0.1285) 미달
- α=0.7 이하에서 decoder 붕괴 (CER 2.3~2.5)
- step 50~5000까지 결과 거의 동일 → 구조적 한계

### 결론
단일 고정 B를 encoder_output에 보간하는 방식은 utterance-specific representation을 만들 수 없어 한계. 논문은 decoder도 함께 fine-tune한 것으로 판단되며, B-only training은 insufficient.

### 전환 방향: LoRA adapter
- decoder cross-attention에 LoRA adapter 추가 (rank=8, PEFT)
- B 보간 불필요 — LoRA decoder가 직접 도메인 패턴 학습
- 기존 harness/eval 인프라 재활용
- 주의: PEFT seq2seq의 음성 입력 미지원 → 오버라이딩 필요
