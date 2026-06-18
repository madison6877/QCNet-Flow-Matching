from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class FiniteDifferenceGeometryLoss(nn.Module):
    def __init__(self, num_directions: int = 8, perturbation: float = 0.05, scale_weight: float = 0.1, output_dim: int = 2, eps: float = 1e-8) -> None:
        super().__init__()

        if num_directions < 2:
            raise ValueError("num_directions must be at least 2.")
        if perturbation <= 0:
            raise ValueError("perturbation must be positive.")
        if scale_weight < 0:
            raise ValueError("scale_weight must be non-negative.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        self.num_directions = int(num_directions)
        self.perturbation = float(perturbation)
        self.scale_weight = float(scale_weight)
        self.output_dim = int(output_dim)
        self.eps = float(eps)

    def forward(self, decoder_std: nn.Module, z_std: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z_std.ndim != 3:
            raise ValueError(f"z_std must have shape [B, num_intents, latent_dim], got {tuple(z_std.shape)}.")

        batch_size, num_intents, latent_dim = z_std.shape
        if batch_size == 0:
            zero = z_std.sum() * 0.0
            return {
                "loss": zero,
                "direction_loss": zero,
                "scale_loss": zero,
                "mean_sensitivity": zero,
                "min_sensitivity": zero,
                "max_sensitivity": zero,
            }

        flat_dim = num_intents * latent_dim
        z_flat = z_std.reshape(batch_size, flat_dim)

        # Random unit directions in the standardized latent space.
        directions = torch.randn(batch_size, self.num_directions, flat_dim, device=z_std.device, dtype=z_std.dtype)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(self.eps)

        delta = self.perturbation * directions
        z_plus = (z_flat[:, None, :] + delta).reshape(batch_size * self.num_directions, num_intents, latent_dim)
        z_minus = (z_flat[:, None, :] - delta).reshape(batch_size * self.num_directions, num_intents, latent_dim)

        # Central finite difference: J_D(z) u ≈ [D(z + εu) - D(z - εu)] / (2ε)
        trajectory_plus = decoder_std(z_plus)[..., :self.output_dim]
        trajectory_minus = decoder_std(z_minus)[..., :self.output_dim]

        directional_derivative = (trajectory_plus - trajectory_minus) / (2.0 * self.perturbation)
        directional_derivative = directional_derivative.reshape(batch_size, self.num_directions, directional_derivative.size(-2), self.output_dim)

        # Equal weighting across all T_future frames and x/y coordinates.
        directional_sensitivity = directional_derivative.pow(2).mean(dim=(-1, -2))
        log_sensitivity = torch.log(directional_sensitivity.clamp_min(self.eps))

        # For one latent point, all directions should have similar sensitivity.
        direction_loss = log_sensitivity.var(dim=1, unbiased=False).mean()

        # Different latent points should have similar overall local scale.
        per_sample_log_scale = log_sensitivity.mean(dim=1)
        scale_loss = per_sample_log_scale.var(unbiased=False)

        total_loss = direction_loss + self.scale_weight * scale_loss

        return {
            "loss": total_loss,
            "direction_loss": direction_loss,
            "scale_loss": scale_loss,
            "mean_sensitivity": directional_sensitivity.mean(),
            "min_sensitivity": directional_sensitivity.min(),
            "max_sensitivity": directional_sensitivity.max(),
        }