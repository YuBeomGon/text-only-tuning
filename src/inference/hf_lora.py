"""LoRA-based Whisper inference (no B interpolation).

Loads base Whisper + optional LoRA adapter, merges weights,
and transcribes via standard generate().
"""
import torch
import numpy as np
import soundfile as sf
import scipy.signal
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def load_model_with_lora(
    model_name: str, lora_path: str, device: str = "cuda"
):
    """Load base Whisper model with LoRA adapter merged.

    The adapter is merged into base weights via merge_and_unload()
    so inference speed is identical to the original model.
    """
    from peft import PeftModel

    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model = model.to(device)
    model.eval()
    processor = WhisperProcessor.from_pretrained(model_name)
    return model, processor


def load_model_base(model_name: str, device: str = "cuda"):
    """Load base Whisper model without LoRA (for baseline comparison)."""
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    processor = WhisperProcessor.from_pretrained(model_name)
    return model, processor


def transcribe_single(
    audio_path: str,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    language: str = "ko",
    device: str = "cuda",
) -> str:
    """Transcribe a single audio file. No B interpolation needed.

    Uses soundfile + scipy (not librosa) to avoid numpy compat issues.
    """
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo -> mono
    if sr != 16000:
        num_samples = int(len(audio) * 16000 / sr)
        audio = scipy.signal.resample(audio, num_samples)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device=device, dtype=model.dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            language=language,
            task="transcribe",
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()
