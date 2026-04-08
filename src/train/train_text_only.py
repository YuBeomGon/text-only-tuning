"""Train bias B with text-only data. Encoder and decoder frozen, B is sole trainable param.

Usage:
    python -m src.train.train_text_only --config configs/train_v1.yaml

    # or with CLI overrides:
    python -m src.train.train_text_only --config configs/train_v1.yaml --max_steps 500
"""
import argparse
import json
import math
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperTokenizer

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


def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _load_texts(text_files: list[str] | None, text_file: str | None) -> list[str]:
    """Load text lines from multiple files (text_files) or single file (text_file).

    text_files takes priority. text_file is kept for backward compatibility.
    """
    paths: list[str] = []
    if text_files:
        paths = list(text_files)
    elif text_file:
        paths = [text_file]
    else:
        raise ValueError("Either text_files or text_file must be provided")

    texts: list[str] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            texts.extend(line.strip() for line in f if line.strip())
    return texts


def _build_domain_token_set(
    lexicon_path: str, model_name: str, language: str,
) -> set[int]:
    """Read domain terms from lexicon file and return their token IDs."""
    tokenizer = WhisperTokenizer.from_pretrained(model_name)
    tokenizer.set_prefix_tokens(language=language, task="transcribe")

    token_ids: set[int] = set()
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for line in f:
            term = line.strip()
            if not term:
                continue
            ids = tokenizer.encode(term)
            # encode() includes prefix + EOT; strip them to get content tokens
            # prefix tokens are the first 4 ([SOT, lang, task, notimestamps])
            # last token is EOT
            content_ids = ids[4:-1]  # Always strip 4 prefix tokens + EOT
            if content_ids:
                token_ids.update(content_ids)
    return token_ids


def train(
    model_name: str,
    e_pretrained_path: str,
    output_dir: str,
    text_files: list[str] | None = None,
    text_file: str | None = None,
    max_steps: int = 1000,
    save_every: int = 50,
    log_every: int = 10,
    batch_size: int = 8,
    lr: float = 3e-5,
    warmup_steps: int = 50,
    scheduler: str = "cosine",
    grad_clip: float = 1.0,
    ema_decay: float = 1.0,
    kl_weight: float = 0.0,
    bd_weight: float = 1.0,
    lexicon_path: str | None = None,
    language: str = "ko",
    device: str = "cuda",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model in fp32 for stable training, freeze everything
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    # Load E_pretrained and create trainable B (fp32)
    e_pretrained = torch.load(e_pretrained_path, map_location=device)
    e_pretrained_sq = e_pretrained.clone().squeeze(0).float()  # (seq_len, d_model)
    B = nn.Parameter(e_pretrained_sq.clone())

    # EMA copy of B (not a Parameter, no grad)
    B_ema = e_pretrained_sq.clone()

    # Load text data (multi-file or single-file)
    texts = _load_texts(text_files, text_file)

    dataset = DomainTextDataset(texts, model_name=model_name, language=language)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW([B], lr=lr)
    lr_scheduler = None
    if scheduler == "cosine":
        lr_scheduler = get_cosine_schedule(optimizer, warmup_steps, max_steps)

    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Build domain token set for Bregman Divergence penalty
    domain_token_ids: set[int] | None = None
    if lexicon_path is not None:
        domain_token_ids = _build_domain_token_set(lexicon_path, model_name, language)
        print(f"Loaded {len(domain_token_ids)} domain token IDs from {lexicon_path}")

    # Precompute E_pretrained encoder output (frozen, reused every step for KL)
    e_pretrained_enc = e_pretrained_sq.detach()  # (seq_len, d_model)

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

        # ---- Forward with B (pred) ----
        encoder_output = B.unsqueeze(0).expand(bsz, -1, -1)
        decoder_out = model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_output,
        )
        logits = model.proj_out(decoder_out.last_hidden_state)

        # CE loss
        ce_loss = ce_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        # ---- KL divergence loss ----
        # Forward with E_pretrained (baseline, no grad)
        with torch.no_grad():
            baseline_enc = e_pretrained_enc.unsqueeze(0).expand(bsz, -1, -1)
            baseline_out = model.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=baseline_enc,
            )
            baseline_logits = model.proj_out(baseline_out.last_hidden_state)

        # KL(P_baseline || P_pred) = sum P_baseline * (log P_baseline - log P_pred)
        # Using F.kl_div which expects input=log_softmax(pred), target=softmax(baseline)
        kl_loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(baseline_logits, dim=-1),
            reduction="batchmean",
        ) / logits.size(1)  # normalize by seq_len for stable kl_weight tuning

        # ---- Bregman Divergence (domain term penalty) ----
        bd_loss = torch.tensor(0.0, device=device)
        if domain_token_ids is not None and len(domain_token_ids) > 0:
            # Build a mask over label positions that correspond to domain tokens
            # labels shape: (bsz, seq_len)
            domain_mask = torch.zeros_like(labels, dtype=torch.bool)
            for tid in domain_token_ids:
                domain_mask |= (labels == tid)
            # Ignore padding positions
            domain_mask &= (labels != -100)

            if domain_mask.any():
                # Per-token CE loss (no reduction)
                per_token_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                )  # (bsz * seq_len,)
                domain_mask_flat = domain_mask.view(-1)
                bd_loss = per_token_loss[domain_mask_flat].mean()

        # ---- Total loss ----
        loss = ce_loss + kl_weight * kl_loss + bd_weight * bd_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([B], grad_clip)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Update EMA of B
        with torch.no_grad():
            B_ema.mul_(ema_decay).add_(B.data, alpha=1.0 - ema_decay)

        step += 1

        if step % log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            ema_diff = (B.data - B_ema).norm().item()
            record = {
                "step": step,
                "loss": loss.item(),
                "ce_loss": ce_loss.item(),
                "kl_loss": kl_loss.item(),
                "bd_loss": bd_loss.item(),
                "lr": current_lr,
                "ema_diff_norm": ema_diff,
            }
            log_records.append(record)
            print(
                f"step={step} loss={loss.item():.4f} "
                f"ce={ce_loss.item():.4f} kl={kl_loss.item():.4f} "
                f"bd={bd_loss.item():.4f} lr={current_lr:.2e}"
            )

        if step % save_every == 0:
            ckpt_path = output_path / f"B_step{step}.pt"
            torch.save(B_ema.unsqueeze(0), ckpt_path)

    # Save final B (EMA version)
    final_path = output_path / "B_final.pt"
    torch.save(B_ema.unsqueeze(0), final_path)

    # Save training log
    log_path = output_path / "train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, indent=2)

    # Save config used
    config = {
        "model_name": model_name, "e_pretrained_path": e_pretrained_path,
        "text_files": text_files, "text_file": text_file,
        "output_dir": output_dir,
        "max_steps": max_steps, "save_every": save_every, "batch_size": batch_size,
        "lr": lr, "warmup_steps": warmup_steps, "scheduler": scheduler,
        "grad_clip": grad_clip, "ema_decay": ema_decay,
        "kl_weight": kl_weight, "bd_weight": bd_weight,
        "lexicon_path": lexicon_path, "language": language,
    }
    with open(output_path / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Training complete. B (EMA) saved to {final_path}")
    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--e_pretrained_path", default=None)
    parser.add_argument("--text_file", default=None, help="Single text file (backward compat)")
    parser.add_argument("--text_files", nargs="+", default=None, help="Multiple text files")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--kl_weight", type=float, default=None)
    parser.add_argument("--bd_weight", type=float, default=None)
    parser.add_argument("--lexicon_path", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Load config from YAML, then override with CLI args
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    for k, v in vars(args).items():
        if k != "config" and v is not None:
            config[k] = v

    train(**config)


if __name__ == "__main__":
    main()
