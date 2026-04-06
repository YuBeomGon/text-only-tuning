import torch


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
