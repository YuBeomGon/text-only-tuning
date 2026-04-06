import torch
from pathlib import Path

import numpy as np
import soundfile as sf
import scipy.signal
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
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo -> mono
    if sr != 16000:
        num_samples = int(len(audio) * 16000 / sr)
        audio = scipy.signal.resample(audio, num_samples)
        sr = 16000
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device=device, dtype=model.dtype)

    with torch.no_grad():
        encoder_output = model.get_encoder()(input_features).last_hidden_state

        if prior is not None and alpha < 1.0:
            prior_matched = prior.to(dtype=encoder_output.dtype)
            encoder_output = mix_encoder_output(encoder_output, prior_matched, alpha)

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
