"""Sweep evaluation across all checkpoints × alpha values.

Usage:
    python -m scripts.sweep_eval \
        --ckpt_dir priors/train_run_002 \
        --alphas 0.9 0.7 0.5 \
        --output_dir experiments/outputs/sweep_run_002
"""
import argparse
import json
import re
from pathlib import Path

from src.eval.run_eval import run_eval


def find_checkpoints(ckpt_dir: str) -> list[tuple[int, str]]:
    """Find all B_step*.pt files and return sorted (step, path) pairs."""
    ckpt_path = Path(ckpt_dir)
    checkpoints = []
    for f in ckpt_path.glob("B_step*.pt"):
        match = re.search(r"B_step(\d+)\.pt", f.name)
        if match:
            checkpoints.append((int(match.group(1)), str(f)))
    return sorted(checkpoints)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.9, 0.7, 0.5])
    parser.add_argument("--step_interval", type=int, default=100,
                        help="Evaluate every N steps (default: 100)")
    parser.add_argument("--model_name", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--manifest", default="data/processed/eval_v1/manifest.jsonl")
    parser.add_argument("--lexicon", default="data/lexicon/domain_terms.txt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--language", default="ko")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.ckpt_dir)
    checkpoints = [(s, p) for s, p in checkpoints if s % args.step_interval == 0]
    if not checkpoints:
        print(f"No checkpoints found in {args.ckpt_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints: {[s for s, _ in checkpoints]}")
    print(f"Alphas: {args.alphas}")
    print(f"Total eval runs: {len(checkpoints) * len(args.alphas)}")

    all_results = []

    for step, ckpt_path in checkpoints:
        for alpha in args.alphas:
            run_name = f"step{step}_alpha{alpha}"
            output_json = str(Path(args.output_dir) / f"{run_name}/metrics.json")

            print(f"\n{'='*60}")
            print(f"Evaluating: step={step}, alpha={alpha}, prior={ckpt_path}")
            print(f"{'='*60}")

            metrics = run_eval(
                model_name=args.model_name,
                manifest_path=args.manifest,
                prior_path=ckpt_path,
                alpha=alpha,
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
    print(f"{'step':>6} {'alpha':>6} {'CER':>10} {'WER':>10} {'term_recall':>12} {'hall':>6} {'lat_ms':>8}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['step']:>6} {r['alpha']:>6.1f} {r['cer']:>10.6f} {r['wer']:>10.6f} "
              f"{r['domain_term_recall']:>12.6f} {r['hallucination_rate']:>6.3f} {r['latency_ms']:>8.1f}")


if __name__ == "__main__":
    main()
