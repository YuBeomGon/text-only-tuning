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

    stacked = torch.cat(encoder_outputs, dim=0)
    e_pretrained = stacked.mean(dim=0, keepdim=True)
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
