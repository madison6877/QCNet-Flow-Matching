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
from typing import Optional

import torch
import torch.nn as nn

from utils import weight_init


class TransformerLayer(nn.Module):
    """Standard Pre-norm Transformer block built on nn.MultiheadAttention.

    Supports optional context injection: when ``context`` is given, it is
    added to the query and key just before the attention computation, while
    the value stays as the original input.  This replaces the former
    ``GatedMHA`` pattern with a clean, canonical block that includes
    residual connections and a feed-forward sub-layer.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float) -> None:
        super(TransformerLayer, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_prenorm = nn.LayerNorm(hidden_dim)
        self.ffn_prenorm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.apply(weight_init)

    def forward(self,
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[batch, seq_len, hidden_dim]``.
            context: Optional conditioning tensor broadcastable to
                ``[1, seq_len, hidden_dim]``.  When provided, it is
                injected into query and key: ``q = k = x + context``
                while ``v = x``.

        Returns:
            Output tensor with the same shape as ``x``.
        """
        # ---- Pre-norm Attention ----
        x_norm = self.attn_prenorm(x)
        if context is None:
            attn_out, _ = self.attn(
                x_norm, x_norm, x_norm, need_weights=False
            )
        else:
            # context is broadcastable to [1, seq_len, hidden_dim]
            qk = x_norm + context
            attn_out, _ = self.attn(
                qk, qk, x_norm, need_weights=False
            )
        x = x + attn_out

        # ---- Pre-norm FFN ----
        x = x + self.ffn(self.ffn_prenorm(x))

        return x