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
        audio_path = item["audio_path"]
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
