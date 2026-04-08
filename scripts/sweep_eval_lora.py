"""Sweep evaluation across all LoRA checkpoints.

Usage:
    python -m scripts.sweep_eval_lora \
        --ckpt_dir lora/run_001 \
        --step_interval 200 \
        --output_dir experiments/outputs/sweep_lora_run_001
"""
import argparse
import json
import re
from pathlib import Path

from src.eval.run_eval_lora import run_eval


def find_lora_checkpoints(ckpt_dir: str) -> list[tuple[int, str]]:
    """Find all step_N/ directories that contain adapter_model files.

    Looks for directories matching step_N/ containing either
    adapter_model.safetensors or adapter_model.bin.
    """
    ckpt_path = Path(ckpt_dir)
    checkpoints = []
    for d in ckpt_path.iterdir():
        if not d.is_dir():
            continue
        match = re.match(r"step_(\d+)$", d.name)
        if not match:
            continue
        # Verify adapter files exist
        has_safetensors = (d / "adapter_model.safetensors").exists()
        has_bin = (d / "adapter_model.bin").exists()
        if has_safetensors or has_bin:
            checkpoints.append((int(match.group(1)), str(d)))
    return sorted(checkpoints)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True,
                        help="Directory containing step_N/ LoRA checkpoints")
    parser.add_argument("--step_interval", type=int, default=200,
                        help="Evaluate every N steps (default: 200)")
    parser.add_argument("--model_name", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--manifest", default="data/processed/eval_v1/manifest.jsonl")
    parser.add_argument("--lexicon", default="data/lexicon/domain_terms.txt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--language", default="ko")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoints = find_lora_checkpoints(args.ckpt_dir)
    checkpoints = [(s, p) for s, p in checkpoints if s % args.step_interval == 0]
    if not checkpoints:
        print(f"No LoRA checkpoints found in {args.ckpt_dir}")
        return

    print(f"Found {len(checkpoints)} LoRA checkpoints: {[s for s, _ in checkpoints]}")
    print(f"Total eval runs: {len(checkpoints)}")

    all_results = []

    for step, lora_path in checkpoints:
        run_name = f"step_{step}"
        output_json = str(Path(args.output_dir) / f"{run_name}/metrics.json")

        print(f"\n{'='*60}")
        print(f"Evaluating: step={step}, lora={lora_path}")
        print(f"{'='*60}")

        metrics = run_eval(
            model_name=args.model_name,
            manifest_path=args.manifest,
            lora_path=lora_path,
            lexicon_path=args.lexicon,
            output_json=output_json,
            language=args.language,
            device=args.device,
        )
        metrics["step"] = step
        all_results.append(metrics)

    # Save summary
    summary_path = Path(args.output_dir) / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'step':>6} {'CER':>10} {'WER':>10} {'term_recall':>12} {'hall':>6} {'lat_ms':>8}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['step']:>6} {r['cer']:>10.6f} {r['wer']:>10.6f} "
              f"{r['domain_term_recall']:>12.6f} {r['hallucination_rate']:>6.3f} {r['latency_ms']:>8.1f}")


if __name__ == "__main__":
    main()
