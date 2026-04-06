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


def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    model_name: str,
    e_pretrained_path: str,
    text_file: str,
    output_dir: str,
    max_steps: int = 1000,
    save_every: int = 50,
    log_every: int = 10,
    batch_size: int = 8,
    lr: float = 3e-5,
    warmup_steps: int = 50,
    scheduler: str = "cosine",
    grad_clip: float = 1.0,
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
    B = nn.Parameter(e_pretrained.clone().squeeze(0).float())  # (seq_len, d_model)

    # Load text data
    with open(text_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    dataset = DomainTextDataset(texts, model_name=model_name, language=language)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW([B], lr=lr)
    lr_scheduler = None
    if scheduler == "cosine":
        lr_scheduler = get_cosine_schedule(optimizer, warmup_steps, max_steps)

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
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([B], grad_clip)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        step += 1

        if step % log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            record = {"step": step, "loss": loss.item(), "lr": current_lr}
            log_records.append(record)
            print(f"step={step} loss={loss.item():.4f} lr={current_lr:.2e}")

        if step % save_every == 0:
            ckpt_path = output_path / f"B_step{step}.pt"
            torch.save(B.detach().unsqueeze(0), ckpt_path)

    # Save final B
    final_path = output_path / "B_final.pt"
    torch.save(B.detach().unsqueeze(0), final_path)

    # Save training log
    log_path = output_path / "train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, indent=2)

    # Save config used
    config = {
        "model_name": model_name, "e_pretrained_path": e_pretrained_path,
        "text_file": text_file, "output_dir": output_dir,
        "max_steps": max_steps, "save_every": save_every, "batch_size": batch_size,
        "lr": lr, "warmup_steps": warmup_steps, "scheduler": scheduler,
        "grad_clip": grad_clip, "language": language,
    }
    with open(output_path / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Training complete. B saved to {final_path}")
    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--e_pretrained_path", default=None)
    parser.add_argument("--text_file", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
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
