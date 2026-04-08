"""Joint B + LoRA decoder fine-tuning with text-only data.

Trains two parameter groups simultaneously:
  - B (nn.Parameter): encoder-output bias, initialised from E_pretrained.
  - LoRA adapters on decoder cross-attention (q_proj, v_proj).

The encoder is never run; B is expanded per-batch and injected as
encoder_outputs.  An EMA copy of B is maintained for checkpointing.

Usage:
    python -m src.train.train_lora --config configs/train_lora_v1.yaml

    # or with CLI overrides:
    python -m src.train.train_lora --config configs/train_lora_v1.yaml --max_steps 500
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
from transformers.modeling_outputs import BaseModelOutput
from peft import LoraConfig, get_peft_model

from src.train.dataset import DomainTextDataset


# ---------------------------------------------------------------------------
# Helpers (shared with train_text_only.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model_name: str,
    e_pretrained_path: str,
    output_dir: str,
    text_files: list[str] | None = None,
    text_file: str | None = None,
    max_steps: int = 1000,
    save_every: int = 200,
    log_every: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    warmup_steps: int = 50,
    scheduler: str = "cosine",
    grad_clip: float = 1.0,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    ema_decay: float = 1.0,
    language: str = "ko",
    device: str = "cuda",
):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load whisper model in fp32, freeze everything
    # ------------------------------------------------------------------
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # 2. Apply LoRA to decoder cross-attention layers
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 3. Create trainable B from E_pretrained + EMA copy
    # ------------------------------------------------------------------
    e_pretrained = torch.load(
        e_pretrained_path, map_location=device, weights_only=True,
    ).squeeze(0).float()  # (seq_len, d_model)
    B = nn.Parameter(e_pretrained.clone())
    B_ema = e_pretrained.clone()  # not a Parameter, no grad

    # ------------------------------------------------------------------
    # 4. Dataset & dataloader
    # ------------------------------------------------------------------
    texts = _load_texts(text_files, text_file)
    dataset = DomainTextDataset(texts, model_name=model_name, language=language)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # 5. Optimizer & scheduler (LoRA params + B)
    # ------------------------------------------------------------------
    lora_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params = lora_params + [B]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    lr_scheduler = None
    if scheduler == "cosine":
        lr_scheduler = get_cosine_schedule(optimizer, warmup_steps, max_steps)

    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    model.train()
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

        encoder_hidden_states = B.unsqueeze(0).expand(bsz, -1, -1)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        # Forward through the PEFT-wrapped model.
        # LoRA adapters are applied automatically to the decoder.
        # We pass encoder_outputs to skip the encoder forward pass
        # and inject B as the encoder hidden states.
        outputs = model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            labels=None,  # compute loss manually for clarity
        )
        logits = outputs.logits

        loss = ce_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Update EMA of B
        with torch.no_grad():
            B_ema.mul_(ema_decay).add_(B.data, alpha=1.0 - ema_decay)

        step += 1

        # Logging
        if step % log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            ema_diff = (B.data - B_ema).norm().item()
            record = {
                "step": step,
                "loss": loss.item(),
                "lr": current_lr,
                "ema_diff_norm": ema_diff,
            }
            log_records.append(record)
            print(
                f"step={step} loss={loss.item():.4f} lr={current_lr:.2e}"
                f" ema_diff={ema_diff:.4f}"
            )

        # Checkpoint (LoRA adapter + B)
        if step % save_every == 0:
            ckpt_dir = output_path / f"step_{step}"
            model.save_pretrained(ckpt_dir)
            torch.save(B_ema.unsqueeze(0), ckpt_dir / "B.pt")
            print(f"  checkpoint saved to {ckpt_dir}")

    # ------------------------------------------------------------------
    # 7. Save final adapter + B, log, and config
    # ------------------------------------------------------------------
    model.save_pretrained(output_path)
    torch.save(B_ema.unsqueeze(0), output_path / "B.pt")
    print(f"Final LoRA adapter + B saved to {output_path}")

    log_path = output_path / "train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, indent=2)

    config_record = {
        "model_name": model_name,
        "e_pretrained_path": e_pretrained_path,
        "text_files": text_files,
        "text_file": text_file,
        "output_dir": output_dir,
        "max_steps": max_steps,
        "save_every": save_every,
        "log_every": log_every,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "scheduler": scheduler,
        "grad_clip": grad_clip,
        "ema_decay": ema_decay,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "language": language,
        "device": device,
    }
    with open(output_path / "train_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    print("Training complete.")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LoRA decoder fine-tuning with text-only data"
    )
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
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--target_modules", nargs="+", default=None)
    parser.add_argument("--ema_decay", type=float, default=None)
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
