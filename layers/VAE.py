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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.fourier_embedding import FourierEmbedding
from utils import weight_init

class LearnableEndpointGate(nn.Module):
    def __init__(self, 
                 initial_bias: float = 0.70, 
                 initial_sharpness: float = 15.0, 
                 min_bias: float = 0.50, 
                 max_bias: float = 0.90, 
                 min_sharpness: float = 1.0, 
                 eps: float = 1e-6) -> None:
        super().__init__()

        self.min_bias = float(min_bias)
        self.max_bias = float(max_bias)
        self.min_sharpness = float(min_sharpness)
        self.eps = float(eps)

        bias_ratio = (initial_bias - min_bias) / (max_bias - min_bias)
        bias_ratio = min(max(bias_ratio, 1e-4), 1.0 - 1e-4)

        self.bias_raw = nn.Parameter(torch.tensor(math.log(bias_ratio / (1.0 - bias_ratio)), dtype=torch.float32))

        sharpness_target = max(initial_sharpness - min_sharpness, 1e-4)
        self.sharpness_raw = nn.Parameter(torch.tensor(math.log(math.exp(sharpness_target) - 1.0), dtype=torch.float32))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # b 始终限制在 [0.50, 0.90]
        bias = self.min_bias + (self.max_bias - self.min_bias) * torch.sigmoid(self.bias_raw)

        # k 始终为正
        sharpness = self.min_sharpness + F.softplus(self.sharpness_raw)

        value = torch.sigmoid(sharpness * (s - bias))
        value_at_zero = torch.sigmoid(-sharpness * bias)
        value_at_one = torch.sigmoid(sharpness * (1.0 - bias))

        gate = (value - value_at_zero) / (value_at_one - value_at_zero).clamp_min(self.eps)

        return gate.clamp(0.0, 1.0)


class VAEEncoderBlock(nn.Module):
    """A single encoder block: Temporal SA → Intent SA → Intent×Trajectory CA.

    Stackable via nn.ModuleList to form a deep encoder.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_intents: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(VAEEncoderBlock, self).__init__()
        self.num_intents = num_intents

        # ---- Temporal self-attention block ----
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

        # ---- Intent self-attention block ----
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

        # ---- Intent×trajectory cross-attention block ----
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

    # ------------------------------------------------------------------
    #  Helper: self-attention transformer block
    # ------------------------------------------------------------------
    def _self_attn_block(self, x: torch.Tensor,
                         norm1: nn.LayerNorm,
                         attn: nn.MultiheadAttention,
                         norm2: nn.LayerNorm,
                         ffn: nn.Sequential,
                         key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Pre-Norm → Self-Attention → Residual → Pre-Norm → FFN → Residual."""
        x_norm = norm1(x)
        attn_out, _ = attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + ffn(norm2(x))
        return x

    # ------------------------------------------------------------------
    #  Helper: cross-attention transformer block
    # ------------------------------------------------------------------
    def _cross_attn_block(self, q: torch.Tensor,
                          kv: torch.Tensor,
                          norm_q: nn.LayerNorm,
                          norm_kv: nn.LayerNorm,
                          attn: nn.MultiheadAttention,
                          norm2: nn.LayerNorm,
                          ffn: nn.Sequential,
                          key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Pre-Norm → Cross-Attention → Residual → Pre-Norm → FFN → Residual."""
        q_norm = norm_q(q)
        kv_norm = norm_kv(kv)
        attn_out, _ = attn(q_norm, kv_norm, kv_norm, key_padding_mask=key_padding_mask)
        q = q + attn_out
        q = q + ffn(norm2(q))
        return q

    def forward(self, x_seq: torch.Tensor, intents: torch.Tensor, key_padding_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Temporal self-attention
        x_seq = self._self_attn_block(
            x_seq,
            self.temporal_sa_norm1, self.temporal_sa,
            self.temporal_sa_norm2, self.temporal_sa_ffn,
            key_padding_mask=key_padding_mask
        )

        # 2. Intent self-attention
        if self.num_intents > 1:
            intents = self._self_attn_block(
                intents,
                self.intent_sa_norm1, self.intent_sa,
                self.intent_sa_norm2, self.intent_sa_ffn,
            )

        # 3. Intent×trajectory cross-attention
        intents = self._cross_attn_block(
            q=intents, kv=x_seq,
            norm_q=self.intent_ca_norm1_q, norm_kv=self.intent_ca_norm1_kv,
            attn=self.intent_ca,
            norm2=self.intent_ca_norm2, ffn=self.intent_ca_ffn,
            key_padding_mask=key_padding_mask
        )

        return x_seq, intents


class VAEDecoderBlock(nn.Module):
    """A single decoder block: Time SA → Time×Latent CA.

    Stackable via nn.ModuleList to form a deep decoder.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(VAEDecoderBlock, self).__init__()

        # ---- Time self-attention block ----
        self.sa_norm1 = nn.LayerNorm(hidden_dim)
        self.sa = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                         dropout=dropout, batch_first=False)
        self.sa_norm2 = nn.LayerNorm(hidden_dim)
        self.sa_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # ---- Time×latent cross-attention block ----
        self.ca_norm1_q = nn.LayerNorm(hidden_dim)
        self.ca_norm1_kv = nn.LayerNorm(hidden_dim)
        self.ca = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                         dropout=dropout, batch_first=False)
        self.ca_norm2 = nn.LayerNorm(hidden_dim)
        self.ca_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.last_attn_weights = None

    # ------------------------------------------------------------------
    #  Helper: self-attention transformer block
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
    def _cross_attn_block(self,q: torch.Tensor,
                          kv: torch.Tensor,
                          norm_q: nn.LayerNorm,
                          norm_kv: nn.LayerNorm,
                          attn: nn.MultiheadAttention,
                          norm2: nn.LayerNorm,
                          ffn: nn.Sequential) -> torch.Tensor:
        """Pre-Norm → Cross-Attention → Residual → Pre-Norm → FFN → Residual."""
        q_norm = norm_q(q)
        kv_norm = norm_kv(kv)
        attn_out, attn_weights = attn(q_norm, kv_norm, kv_norm)
        self.last_attn_weights = attn_weights
        q = q + attn_out
        q = q + ffn(norm2(q))
        return q

    def forward(self, time_q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # 1. Time self-attention
        time_q = self._self_attn_block(
            time_q,
            self.sa_norm1, self.sa,
            self.sa_norm2, self.sa_ffn,
        )

        # 2. Time×latent cross-attention
        time_q = self._cross_attn_block(
            q=time_q, kv=z,
            norm_q=self.ca_norm1_q, norm_kv=self.ca_norm1_kv,
            attn=self.ca,
            norm2=self.ca_norm2, ffn=self.ca_ffn,
        )

        return time_q


class VAE(nn.Module):
    """Variational Auto-Encoder for trajectory latent space encoding.

    Encoder pipeline:
      1. FourierEmbedding per timestep
      2. N stacked VAEEncoderBlock (each: Temporal SA → Intent SA → Intent×Trajectory CA)
      3. Reparameterization into latent variables Z

    Decoder pipeline:
      1. N stacked VAEDecoderBlock (each: Time SA → Time×Latent CA)
      2. Shared MLP mapping to (x, y) coordinates

    Every attention block follows Pre-Norm → Attention → Residual → Pre-Norm → FFN → Residual.
    """

    def __init__(self,
                 hidden_dim: int,
                 latent_dim: int = 16,
                 input_dim: int = 2,
                 num_future_steps: int = 60,
                 num_intents: int = 4,
                 num_encoder_blocks: int = 2,
                 num_decoder_blocks: int = 2,
                 num_freq_bands: int = 64,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(VAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_future_steps = num_future_steps
        self.num_intents = num_intents
        self.latent_dim = latent_dim

        # ---- Encoder: Fourier embedding ----
        self.fourier_emb = FourierEmbedding(input_dim=input_dim*2, hidden_dim=hidden_dim,
                                             num_freq_bands=num_freq_bands)

        # ---- Encoder: Stacked VAEEncoderBlocks ----
        self.encoder_blocks = nn.ModuleList([
            VAEEncoderBlock(hidden_dim=hidden_dim, num_intents=num_intents,
                            num_heads=num_heads, dropout=dropout)
            for _ in range(num_encoder_blocks)
        ])

        pe = torch.zeros(num_future_steps, hidden_dim)
        position = torch.arange(0, num_future_steps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1) # [T_f, 1, hidden_dim]
        self.register_buffer('time_pe', pe) # 注册 buffer，使其随模型保存但不参与梯度更新
        
        # 2. 可学习的时间查询基座 (保持不变)
        self.time_queries_base = nn.Parameter(torch.randn(num_future_steps, 1, hidden_dim) * 0.1)
        self.intent_queries = nn.Parameter(torch.randn(num_intents, 1, hidden_dim) * 0.1)

        # ---- Reparameterization (shared MLP, after all encoder blocks) ----
        self.reparam_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim * 2),
        )

        # ---- Decoder: Stacked VAEDecoderBlocks ----
        self.z_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            VAEDecoderBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_decoder_blocks)
        ])

        # ---- Decoder: Coordinate mapping ----
        self.decoder_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, input_dim),
        )

        self.endpoint_delta_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.input_dim),
        )
        
        self.endpoint_gate = LearnableEndpointGate()

        self.apply(weight_init)
        nn.init.zeros_(self.endpoint_delta_head[-1].weight)
        nn.init.zeros_(self.endpoint_delta_head[-1].bias)

    def encode(self, x: torch.Tensor, predict_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode future trajectories into latent distribution parameters.

        Args:
            x: [N_a, T_f, 2] future trajectory coordinates (agent-centric).

        Returns:
            mu:     [3, N_a, hidden_dim]
            logvar: [3, N_a, hidden_dim]
            z:      [3, N_a, hidden_dim] sampled latent variables.
        """
        N_a, T_f, D = x.shape

        if predict_mask is None:
            mask = torch.ones(N_a,T_f,dtype=torch.bool,device=x.device)
        else:
            mask = predict_mask.bool()
        # 无效位置先清零
        x_masked = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        vel = torch.zeros_like(x_masked)
        # 只有相邻两个时间点都有效时，速度才有效
        pair_valid = mask[:, 1:] & mask[:, :-1]
        vel[:, 1:] = (x_masked[:, 1:] - x_masked[:, :-1]) * pair_valid.unsqueeze(-1)
        x_augmented = torch.cat([x_masked, vel],dim=-1,)

        # 1. FourierEmbedding per timestep
        x_flat = x_augmented.view(N_a * T_f, self.input_dim * 2)
        x_emb = self.fourier_emb(continuous_inputs=x_flat, categorical_embs=None)
        x_emb = x_emb.view(N_a, T_f, self.hidden_dim)
        x_seq = x_emb.transpose(0, 1)  # [T_f, N_a, hidden_dim]
        x_seq = x_seq + self.time_pe[:T_f].to(device=x_seq.device,dtype=x_seq.dtype,)

        intents = self.intent_queries.expand(-1, N_a, -1)  # [3, N_a, hidden_dim]

        key_padding_mask = None
        if predict_mask is not None:
            valid_agent_mask = predict_mask.any(dim=1)
            mask_bool = ~predict_mask.bool()
            key_padding_mask = mask_bool
            key_padding_mask[~valid_agent_mask] = False

        # 2. Stacked encoder blocks (x_seq flows through all blocks)
        for enc_block in self.encoder_blocks:
            x_seq, intents = enc_block(x_seq, intents, key_padding_mask=key_padding_mask)

        # 3. Reparameterization (shared MLP, after all encoder blocks)
        params = self.reparam_mlp(intents)  # [3, N_a, hidden_dim * 2]
        mu, logvar = torch.chunk(params, 2, dim=-1)  # each [3, N_a, hidden_dim]
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)

        # Sample z via reparameterization trick
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
        else:
            z = mu

        return mu, logvar, z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent variables back to trajectory coordinates.

        Args:
            z: [3, N_a, hidden_dim] latent variables.

        Returns:
            recon_x: [N_a, T_f, 2] reconstructed trajectories.
        """
        N_a = z.size(1)
        z = self.z_proj(z)  # [3, N_a, hidden_dim] → projected to hidden_dim for cross-attention

        # Expand learnable time queries (shared across all decoder blocks)
        time_q = (self.time_queries_base + self.time_pe).expand(-1, N_a, -1)  # [T_f, N_a, hidden_dim]

        # 1. Stacked decoder blocks (Time SA → Time×Latent CA per block)
        for dec_block in self.decoder_blocks:
            time_q = dec_block(time_q, z)

        # 2. MLP → coordinates
        raw_trajectory = self.decoder_mlp(time_q).transpose(0, 1)   # [N_a, T_f, 2]

        endpoint_feature = time_q[-1]
        endpoint_delta = self.endpoint_delta_head(endpoint_feature) 
        endpoint = raw_trajectory[:, -1, :] + endpoint_delta
        s = torch.arange(1, self.num_future_steps + 1, device=raw_trajectory.device, dtype=raw_trajectory.dtype,) / float(self.num_future_steps)
        gate = self.endpoint_gate(s).view(1, self.num_future_steps, 1)
        corrected_trajectory = raw_trajectory + gate * endpoint_delta.unsqueeze(1)
        corrected_trajectory = torch.cat([corrected_trajectory[:, :-1], endpoint.unsqueeze(1)], dim=1)

        return corrected_trajectory

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