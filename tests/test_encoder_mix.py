import pytest
import torch
from src.inference.hf_encoder_mix import mix_encoder_output


def test_alpha_one_returns_original():
    original = torch.randn(1, 10, 512)
    prior = torch.randn(1, 10, 512)
    result = mix_encoder_output(original, prior, alpha=1.0)
    assert torch.allclose(result, original)


def test_alpha_zero_returns_prior():
    original = torch.randn(1, 10, 512)
    prior = torch.randn(1, 10, 512)
    result = mix_encoder_output(original, prior, alpha=0.0)
    assert torch.allclose(result, prior)


def test_alpha_half_is_mean():
    original = torch.ones(1, 10, 512) * 2.0
    prior = torch.zeros(1, 10, 512)
    result = mix_encoder_output(original, prior, alpha=0.5)
    assert torch.allclose(result, torch.ones(1, 10, 512))


def test_shape_mismatch_raises():
    original = torch.randn(1, 10, 512)
    prior = torch.randn(1, 5, 512)
    with pytest.raises(ValueError, match="shape"):
        mix_encoder_output(original, prior, alpha=0.5)


def test_output_shape_matches_input():
    original = torch.randn(1, 10, 512)
    prior = torch.randn(1, 10, 512)
    result = mix_encoder_output(original, prior, alpha=0.7)
    assert result.shape == original.shape
