"""Run evaluation on a manifest of audio files using LoRA adapter.

Usage (with LoRA):
    python -m src.eval.run_eval_lora \
        --model_name openai/whisper-large-v3-turbo \
        --manifest data/processed/eval_v1/manifest.jsonl \
        --lora_path lora/run_001/step_400 \
        --lexicon data/lexicon/domain_terms.txt \
        --output_json experiments/outputs/lora_run_001/step_400/metrics.json

Usage (baseline, no LoRA):
    python -m src.eval.run_eval_lora \
        --model_name openai/whisper-large-v3-turbo \
        --manifest data/processed/eval_v1/manifest.jsonl \
        --output_json experiments/outputs/lora_baseline/metrics.json
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
from src.inference.hf_lora import (
    load_model_with_lora,
    load_model_base,
    transcribe_single,
)


def load_manifest(manifest_path: str) -> list[dict]:
    """Load manifest.jsonl. Resolves audio paths relative to manifest's parent dir."""
    manifest_dir = Path(manifest_path).parent
    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                entry["audio_path"] = str(manifest_dir / entry["audio"])
                items.append(entry)
    return items


def load_lexicon(lexicon_path: str) -> list[str]:
    with open(lexicon_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run_eval(
    model_name: str,
    manifest_path: str,
    lora_path: str | None,
    lexicon_path: str | None,
    output_json: str,
    language: str = "ko",
    device: str = "cuda",
):
    items = load_manifest(manifest_path)
    domain_terms = load_lexicon(lexicon_path) if lexicon_path else []

    # Load model ONCE (not per file)
    if lora_path:
        model, processor = load_model_with_lora(model_name, lora_path, device)
    else:
        model, processor = load_model_base(model_name, device)

    all_refs = []
    all_hyps = []
    latencies = []

    for item in items:
        audio_path = item["audio_path"]
        reference = item["text"]

        t0 = time.time()
        hypothesis = transcribe_single(
            audio_path, model, processor, language, device
        )
        elapsed_ms = (time.time() - t0) * 1000

        all_refs.append(reference)
        all_hyps.append(hypothesis)
        latencies.append(elapsed_ms)

    # Corpus-level CER/WER (total edit distance / total reference length)
    corpus_cer = jiwer_cer(all_refs, all_hyps)
    corpus_wer = jiwer_wer(all_refs, all_hyps)

    # Per-utterance average for term recall
    total_term_recall = sum(
        compute_term_recall(r, h, domain_terms)
        for r, h in zip(all_refs, all_hyps)
    ) / len(all_refs)

    # Hallucination: broadcast pattern detection only
    hall_rate = compute_hallucination_rate(all_hyps)
    avg_latency = sum(latencies) / len(latencies)

    metrics = {
        "cer": round(corpus_cer, 6),
        "wer": round(corpus_wer, 6),
        "domain_term_recall": round(total_term_recall, 6),
        "hallucination_rate": round(hall_rate, 6),
        "timestamp_error": 0.0,
        "latency_ms": round(avg_latency, 2),
        "n_samples": len(items),
        "model_name": model_name,
        "lora_path": lora_path or "none",
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lora_path", default=None,
                        help="Path to LoRA adapter dir (None = base model)")
    parser.add_argument("--lexicon", default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--language", default="ko")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_eval(
        args.model_name, args.manifest, args.lora_path,
        args.lexicon, args.output_json, args.language, args.device,
    )


if __name__ == "__main__":
    main()
