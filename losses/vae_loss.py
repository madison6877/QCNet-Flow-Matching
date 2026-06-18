from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    """VAE loss: reconstruction + beta * KL + gamma * orthogonality."""

    def __init__(self, beta: float = 1.0, gamma: float = 3.0, reduction: str = "mean") -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def reconstruction_loss(self, recon_x: torch.Tensor, target_x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Masked reconstruction loss with the same scale as the current code."""
        raw = F.mse_loss(recon_x, target_x, reduction="none")

        if mask is None:
            return raw.sum(dim=(1, 2)).mean()

        mask = mask.bool()
        valid_agent_mask = mask.any(dim=-1)
        if not valid_agent_mask.any():
            return recon_x.sum() * 0.0

        mask_f = mask.unsqueeze(-1).to(raw.dtype)
        per_agent_sum = (raw * mask_f).sum(dim=(1, 2))
        num_valid_coords = (mask.sum(dim=-1) * recon_x.size(-1)).clamp_min(1)
        per_agent_mse = per_agent_sum / num_valid_coords.to(per_agent_sum.dtype)

        # Preserve the existing 60 frames × 2 coordinates scaling.
        return per_agent_mse[valid_agent_mask].mean() * 120.0

    def forward(self, recon_x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, target_x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, beta: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        recon_loss = self.reconstruction_loss(recon_x, target_x, mask)

        if mask is not None:
            mask = mask.bool()
            valid_agent_mask = mask.any(dim=-1)
            num_valid_agents = valid_agent_mask.sum().clamp_min(1)

            kl_per_agent = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(0, 2))
            kl_loss = kl_per_agent[valid_agent_mask].sum() / num_valid_agents

            num_intents = mu.size(0)
            if num_intents > 1:
                mu_norm = F.normalize(mu, dim=-1)
                cosine = torch.einsum("inj,mnj->nim", mu_norm, mu_norm)
                off_diagonal = ~torch.eye(num_intents, dtype=torch.bool, device=mu.device)
                ortho_per_agent = cosine.abs()[:, off_diagonal].mean(dim=-1)
                ortho_loss = ortho_per_agent[valid_agent_mask].sum() / num_valid_agents
            else:
                ortho_loss = mu.sum() * 0.0
        else:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(0, 2)).mean()

            num_intents = mu.size(0)
            if num_intents > 1:
                mu_norm = F.normalize(mu, dim=-1)
                cosine = torch.einsum("inj,mnj->nim", mu_norm, mu_norm)
                off_diagonal = ~torch.eye(num_intents, dtype=torch.bool, device=mu.device)
                ortho_loss = cosine.abs()[:, off_diagonal].mean()
            else:
                ortho_loss = mu.sum() * 0.0

        beta_now = self.beta if beta is None else float(beta)
        total_loss = recon_loss + beta_now * kl_loss + self.gamma * ortho_loss

        return total_loss, {
            "loss_total": total_loss.detach(),
            "loss_recon": recon_loss.detach(),
            "loss_kl": kl_loss.detach(),
            "vae_beta": total_loss.new_tensor(beta_now),
            "ortho_aux": ortho_loss.detach(),
        }
