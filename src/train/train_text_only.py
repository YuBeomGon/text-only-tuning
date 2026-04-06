"""Train bias B with text-only data. Encoder and decoder frozen, B is sole trainable param.

Usage:
    python -m src.train.train_text_only \
        --model_name openai/whisper-base \
        --e_pretrained_path priors/e_pretrained.pt \
        --text_file data/domain_text.txt \
        --output_dir priors/ \
        --max_steps 100
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
            torch.save(B.detach().unsqueeze(0), ckpt_path)

    # Save final B
    final_path = output_path / "B_final.pt"
    torch.save(B.detach().unsqueeze(0), final_path)

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
