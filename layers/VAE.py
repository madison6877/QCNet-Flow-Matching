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
from typing import Tuple

import torch
import torch.nn as nn

from layers.fourier_embedding import FourierEmbedding
from utils import weight_init


class VAE(nn.Module):
    """Variational Auto-Encoder for trajectory latent space encoding.

    Encoder pipeline:
      1. FourierEmbedding per timestep
      2. Temporal self-attention block (60 steps)
      3. Intent-to-intent self-attention block (3 intents communicate first)
      4. Intent×trajectory cross-attention block
      5. Reparameterization into latent variables Z

    Decoder pipeline:
      1. Time-query × latent cross-attention block
      2. Shared MLP mapping to (x, y) coordinates

    Every attention block follows Pre-Norm → Attention → Residual → Pre-Norm → FFN → Residual.
    """

    def __init__(self,
                 hidden_dim: int,
                 input_dim: int = 2,
                 num_future_steps: int = 60,
                 num_intents: int = 3,
                 num_freq_bands: int = 64,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(VAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_future_steps = num_future_steps
        self.num_intents = num_intents

        # ---- Encoder: Fourier embedding ----
        self.fourier_emb = FourierEmbedding(input_dim=input_dim, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)

        # ---- Encoder: Temporal self-attention block ----
        self.temporal_sa_norm1 = nn.LayerNorm(hidden_dim)
        self.temporal_sa = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                                  dropout=dropout, batch_first=False)
        self.temporal_sa_norm2 = nn.LayerNorm(hidden_dim)
        self.temporal_sa_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # ---- Encoder: Intent self-attention block (3 intents communicate first) ----
        self.intent_queries = nn.Parameter(torch.randn(num_intents, 1, hidden_dim))

        self.intent_sa_norm1 = nn.LayerNorm(hidden_dim)
        self.intent_sa = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                                dropout=dropout, batch_first=False)
        self.intent_sa_norm2 = nn.LayerNorm(hidden_dim)
        self.intent_sa_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # ---- Encoder: Intent×trajectory cross-attention block ----
        self.intent_ca_norm1_q = nn.LayerNorm(hidden_dim)
        self.intent_ca_norm1_kv = nn.LayerNorm(hidden_dim)
        self.intent_ca = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                                dropout=dropout, batch_first=False)
        self.intent_ca_norm2 = nn.LayerNorm(hidden_dim)
        self.intent_ca_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # ---- Reparameterization (shared MLP) ----
        self.reparam_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

        # ---- Decoder: Time-query×latent cross-attention block ----
        self.time_queries = nn.Parameter(torch.randn(num_future_steps, 1, hidden_dim))

        self.decoder_ca_norm1_q = nn.LayerNorm(hidden_dim)
        self.decoder_ca_norm1_kv = nn.LayerNorm(hidden_dim)
        self.decoder_ca = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                                 dropout=dropout, batch_first=False)
        self.decoder_ca_norm2 = nn.LayerNorm(hidden_dim)
        self.decoder_ca_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # ---- Decoder: Coordinate mapping ----
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, input_dim),
        )

        self.apply(weight_init)

    # ------------------------------------------------------------------
    #  Helper: self-attention transformer block (no bipartite pre-norms)
    # ------------------------------------------------------------------
    def _self_attn_block(self, x: torch.Tensor,
                         norm1: nn.LayerNorm,
                         attn: nn.MultiheadAttention,
                         norm2: nn.LayerNorm,
                         ffn: nn.Sequential) -> torch.Tensor:
        """Pre-Norm → Self-Attention → Residual → Pre-Norm → FFN → Residual."""
        x_norm = norm1(x)
        attn_out, _ = attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + ffn(norm2(x))
        return x

    # ------------------------------------------------------------------
    #  Helper: cross-attention transformer block
    # ------------------------------------------------------------------
    def _cross_attn_block(self,
                          q: torch.Tensor,
                          kv: torch.Tensor,
                          norm_q: nn.LayerNorm,
                          norm_kv: nn.LayerNorm,
                          attn: nn.MultiheadAttention,
                          norm2: nn.LayerNorm,
                          ffn: nn.Sequential) -> torch.Tensor:
        """Pre-Norm → Cross-Attention → Residual → Pre-Norm → FFN → Residual."""
        q_norm = norm_q(q)
        kv_norm = norm_kv(kv)
        attn_out, _ = attn(q_norm, kv_norm, kv_norm)
        q = q + attn_out
        q = q + ffn(norm2(q))
        return q

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode future trajectories into latent distribution parameters.

        Args:
            x: [N_a, T_f, 2] future trajectory coordinates (agent-centric).

        Returns:
            mu:     [3, N_a, hidden_dim]
            logvar: [3, N_a, hidden_dim]
            z:      [3, N_a, hidden_dim] sampled latent variables.
        """
        N_a, T_f, _ = x.shape

        # 1. FourierEmbedding per timestep
        x_flat = x.view(N_a * T_f, self.input_dim)
        x_emb = self.fourier_emb(continuous_inputs=x_flat, categorical_embs=None)
        x_emb = x_emb.view(N_a, T_f, self.hidden_dim)
        x_seq = x_emb.transpose(0, 1)  # [T_f, N_a, hidden_dim]

        # 2. Temporal self-attention block
        x_seq = self._self_attn_block(
            x_seq,
            self.temporal_sa_norm1, self.temporal_sa,
            self.temporal_sa_norm2, self.temporal_sa_ffn,
        )  # [T_f, N_a, hidden_dim]

        # 3. Intent self-attention block (3 intents communicate first)
        intents = self.intent_queries.expand(-1, N_a, -1)  # [3, N_a, hidden_dim]
        intents = self._self_attn_block(
            intents,
            self.intent_sa_norm1, self.intent_sa,
            self.intent_sa_norm2, self.intent_sa_ffn,
        )  # [3, N_a, hidden_dim]

        # 4. Intent×trajectory cross-attention block
        intents = self._cross_attn_block(
            q=intents, kv=x_seq,
            norm_q=self.intent_ca_norm1_q, norm_kv=self.intent_ca_norm1_kv,
            attn=self.intent_ca,
            norm2=self.intent_ca_norm2, ffn=self.intent_ca_ffn,
        )  # [3, N_a, hidden_dim]

        # 5. Reparameterization
        params = self.reparam_mlp(intents)  # [3, N_a, hidden_dim * 2]
        mu, logvar = torch.chunk(params, 2, dim=-1)  # each [3, N_a, hidden_dim]

        # Sample z via reparameterization trick
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)

        return mu, logvar, z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent variables back to trajectory coordinates.

        Args:
            z: [3, N_a, hidden_dim] latent variables.

        Returns:
            recon_x: [N_a, T_f, 2] reconstructed trajectories.
        """
        N_a = z.size(1)

        # Cross-attention: 60 time queries × 3 latent intents
        time_q = self.time_queries.expand(-1, N_a, -1)  # [T_f, N_a, hidden_dim]
        decoded = self._cross_attn_block(
            q=time_q, kv=z,
            norm_q=self.decoder_ca_norm1_q, norm_kv=self.decoder_ca_norm1_kv,
            attn=self.decoder_ca,
            norm2=self.decoder_ca_norm2, ffn=self.decoder_ca_ffn,
        )  # [T_f, N_a, hidden_dim]

        # MLP → coordinates
        recon_seq = self.decoder_mlp(decoded)  # [T_f, N_a, 2]
        recon_x = recon_seq.transpose(0, 1)    # [N_a, T_f, 2]

        return recon_x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward pass.

        Args:
            x: [N_a, T_f, 2] future trajectory coordinates.

        Returns:
            recon_x: [N_a, T_f, 2] reconstructed trajectories.
            mu:      [3, N_a, hidden_dim] latent mean.
            logvar:  [3, N_a, hidden_dim] latent log-variance.
        """
        mu, logvar, z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, mu, logvar