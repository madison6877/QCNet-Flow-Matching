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
from typing import Tuple, Union

import torch
import torch.nn as nn

from layers.VAE import VAE
from utils import weight_init


class LatentSpaceEncoder(nn.Module):
    """Latent Space Encoder module for trajectory encoding.

    Single forward() with a mode switch:
      - return_latent=False (default, training):  returns (recon_x, mu, logvar) for VAE loss.
      - return_latent=True  (inference):           returns (mu, logvar) for downstream use.
    """

    def __init__(self,
                 hidden_dim: int,
                 latent_dim: int = 16,
                 input_dim: int = 2,
                 num_future_steps: int = 60,
                 num_intents: int = 4,
                 num_freq_bands: int = 48,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(LatentSpaceEncoder, self).__init__()
        self.vae = VAE(
            hidden_dim=hidden_dim,
            latent_dim = latent_dim,
            input_dim=input_dim,
            num_future_steps=num_future_steps,
            num_intents=num_intents,
            num_freq_bands=num_freq_bands,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.apply(weight_init)

    def forward(self, x: torch.Tensor, return_latent: bool = False, predict_mask: torch.Tensor = None) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        """Encode future trajectory, optionally decode for reconstruction.

        Returns:
            如果 return_latent=False (VAE 训练): 返回 recon_x, mu, logvar
            如果 return_latent=True  (流匹配训练): 返回 z, mu, logvar
        """

        mu, logvar, z = self.vae.encode(x, predict_mask=predict_mask)
        if return_latent:
            return z, mu, logvar
        recon_x = self.vae.decode(z)
        return recon_x, mu, logvar

    @torch.no_grad()
    def encode(self, x: torch.Tensor, predict_mask: torch.Tensor = None) -> torch.Tensor:
        """Encode trajectory → latent z (no gradient, mode preserved).

        Returns:
            z: [N_a, 3, H]  batch-first latent vectors
        """
        mu, logvar, z = self.vae.encode(x, predict_mask=predict_mask)
        return mu.transpose(0, 1).contiguous()  # [3, N_a, H] → [N_a, 3, H]
