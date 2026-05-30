# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    """VAE training loss: reconstruction MSE + KL divergence + orthogonality penalty.

    The orthogonality penalty encourages the 3 intent tokens to learn
    disentangled representations by penalizing pairwise cosine similarity.

    Total loss = recon_loss + beta * kl_loss + gamma * ortho_loss
    """

    def __init__(self,
                 beta: float = 1.0,
                 gamma: float = 3.0,
                 reduction: str = 'mean') -> None:
        super(VAELoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self,
                recon_x: torch.Tensor,
                mu: torch.Tensor,
                logvar: torch.Tensor,
                target_x: torch.Tensor,
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
      
        # ---- Reconstruction loss (scale-aligned with KL) ----
        # Sum over trajectory steps and coordinate dims, mean over agents.
        recon_loss = F.mse_loss(recon_x, target_x, reduction='none')  # [N_a, T_f, D]

        if mask is not None:
            # mask: [N_a, T_f]  (1 for valid, 0 for padding)
            # Dynamic weight compensation: scale short trajectories up to match
            # the element count of full-length trajectories, keeping recon on par
            # with the constant per-agent KL (288 elements).
            valid_elements = mask.sum(dim=1) * recon_x.size(2)          # [N_a] — valid coords per agent
            per_elem_sum = (recon_loss * mask.unsqueeze(-1)).sum(dim=(1, 2))  # [N_a] — valid sum
            per_elem_mean = per_elem_sum / valid_elements.clamp(min=1)        # [N_a] — per-coord mean
            max_elements = float(recon_x.size(1) * recon_x.size(2))           # T_f * D
            recon_loss = (per_elem_mean * max_elements).mean()                # unified-scale batch mean
        else:
            recon_loss = recon_loss.sum(dim=(1, 2)).mean()                    # original behaviour

        # ---- KL divergence (standard VAE reduction) ----
        # Sum over intent dim (0) and feature dim (2), mean over agents (1).
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),
                                   dim=(0, 2)).mean()                  # scalar

        # ---- Orthogonality penalty (intent disentanglement) ----
        # Compute pairwise cosine similarity among the 3 intent vectors.
        mu_norm = F.normalize(mu, dim=-1)                               # [3, N_a, hidden_dim]
        cos_sim = torch.einsum('inj,mnj->nim', mu_norm, mu_norm)        # [N_a, 3, 3]
        mask = ~torch.eye(3, dtype=torch.bool, device=mu.device)        # exclude diagonal
        ortho_loss = cos_sim.abs()[:, mask].mean()                      # scalar

        # ---- Total loss ----
        total_loss = recon_loss + self.beta * kl_loss + self.gamma * ortho_loss

        return total_loss, {
            'loss_total': total_loss.detach(),
            'loss_recon': recon_loss.detach(),
            'loss_kl': kl_loss.detach(),
            'ortho_aux': ortho_loss.detach(),
        }