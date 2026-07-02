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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from utils import weight_init

# === 新增这行 ===
from torch.utils.checkpoint import checkpoint

import torch.nn.functional as F


class AttentionLayer(MessagePassing):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 bipartite: bool,
                 has_pos_emb: bool,
                 **kwargs) -> None:
        super(AttentionLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.has_pos_emb = has_pos_emb
        self.scale = head_dim ** -0.5
        self.debug_forward_count = 0
        self.last_attn_ratio = None
        self.last_ff_ratio = None
        self.debug_name = "unnamed"

        self.to_q = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_k = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.to_v = nn.Linear(hidden_dim, head_dim * num_heads)
        if has_pos_emb:
            self.to_k_r = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
            self.to_v_r = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_s = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_g = nn.Linear(head_dim * num_heads + hidden_dim, head_dim * num_heads)
        self.to_out = nn.Linear(head_dim * num_heads, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.ff_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        if bipartite:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim)
        else:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = self.attn_prenorm_x_src
        if has_pos_emb:
            self.attn_prenorm_r = nn.LayerNorm(hidden_dim)
        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)

    def forward(self,
                x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                r: Optional[torch.Tensor],
                edge_index: torch.Tensor,
                edge_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x_src = x_dst = self.attn_prenorm_x_src(x)
        else:
            x_src, x_dst = x
            x_src = self.attn_prenorm_x_src(x_src)
            x_dst = self.attn_prenorm_x_dst(x_dst)
            x = x[1]
        if self.has_pos_emb and r is not None:
            r = self.attn_prenorm_r(r)
        #  x = x + self.attn_postnorm(self._attn_block(x_src, x_dst, r, edge_index, edge_gate))
        #  x = x + self.ff_postnorm(self._ff_block(self.ff_prenorm(x)))

        x = x + self._attn_block(x_src, x_dst, r, edge_index, edge_gate)
        x = x + self._ff_block(self.ff_prenorm(x))

        return x

    # def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], r: Optional[torch.Tensor], edge_index: torch.Tensor, edge_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
    #     if isinstance(x, torch.Tensor):
    #         x_src = x_dst = self.attn_prenorm_x_src(x)
    #     else:
    #         x_src, x_dst = x
    #         x_src = self.attn_prenorm_x_src(x_src)
    #         x_dst = self.attn_prenorm_x_dst(x_dst)
    #         x = x[1]

    #     if self.has_pos_emb and r is not None:
    #         r = self.attn_prenorm_r(r)

    #     # ============================================================
    #     # 1. Attention 残差分支
    #     # ============================================================
    #     x_before_attn = x

    #     attn_out = self._attn_block(x_src=x_src, x_dst=x_dst, r=r, edge_index=edge_index, edge_gate=edge_gate)

    #     with torch.no_grad():
    #         x_attn_debug = x_before_attn.detach().float()
    #         attn_out_debug = attn_out.detach().float()

    #         x_attn_norm = x_attn_debug.norm(dim=-1)
    #         attn_out_norm = attn_out_debug.norm(dim=-1)

    #         # 每个节点的相对残差强度
    #         attn_ratio_per_node = attn_out_norm / x_attn_norm.clamp_min(1e-6)

    #         attn_ratio_mean = attn_ratio_per_node.mean()
    #         attn_ratio_median = attn_ratio_per_node.median()
    #         attn_ratio_p90 = torch.quantile(attn_ratio_per_node, 0.90)

    #         # 整个张量的 RMS 能量比例
    #         attn_ratio_rms = attn_out_debug.pow(2).mean().sqrt() / x_attn_debug.pow(2).mean().sqrt().clamp_min(1e-6)

    #         # 加入注意力残差后，主干特征范数的变化
    #         x_after_attn_debug = x_attn_debug + attn_out_debug
    #         attn_after_ratio = x_after_attn_debug.norm(dim=-1).mean() / x_attn_norm.mean().clamp_min(1e-6)

    #         # 注意力更新与原主干方向的余弦相似度
    #         attn_cosine = F.cosine_similarity(x_attn_debug, attn_out_debug, dim=-1, eps=1e-6).mean()

    #         self.last_attn_ratio = attn_ratio_mean

    #     x_after_attn = x_before_attn + attn_out

    #     # ============================================================
    #     # 2. FFN 残差分支
    #     # ============================================================
    #     x_before_ff = x_after_attn

    #     ff_in = self.ff_prenorm(x_before_ff)
    #     ff_out = self._ff_block(ff_in)

    #     with torch.no_grad():
    #         x_ff_debug = x_before_ff.detach().float()
    #         ff_out_debug = ff_out.detach().float()

    #         x_ff_norm = x_ff_debug.norm(dim=-1)
    #         ff_out_norm = ff_out_debug.norm(dim=-1)

    #         # 每个节点的相对 FFN 残差强度
    #         ff_ratio_per_node = ff_out_norm / x_ff_norm.clamp_min(1e-6)

    #         ff_ratio_mean = ff_ratio_per_node.mean()
    #         ff_ratio_median = ff_ratio_per_node.median()
    #         ff_ratio_p90 = torch.quantile(ff_ratio_per_node, 0.90)

    #         # 整个张量的 RMS 能量比例
    #         ff_ratio_rms = ff_out_debug.pow(2).mean().sqrt() / x_ff_debug.pow(2).mean().sqrt().clamp_min(1e-6)

    #         # 加入 FFN 残差后，主干特征范数的变化
    #         x_after_ff_debug = x_ff_debug + ff_out_debug
    #         ff_after_ratio = x_after_ff_debug.norm(dim=-1).mean() / x_ff_norm.mean().clamp_min(1e-6)

    #         # FFN 更新与进入 FFN 前主干方向的余弦相似度
    #         ff_cosine = F.cosine_similarity(x_ff_debug, ff_out_debug, dim=-1, eps=1e-6).mean()

    #         self.last_ff_ratio = ff_ratio_mean

    #     x = x_before_ff + ff_out

    #     # ============================================================
    #     # 3. 调试输出
    #     # ============================================================
    #     if self.debug_forward_count % 100 == 0:
    #         print(f"\n[{self.debug_name}] forward_count={self.debug_forward_count}\n  Attention: mean={attn_ratio_mean.item():.3f}, median={attn_ratio_median.item():.3f}, p90={attn_ratio_p90.item():.3f}, rms={attn_ratio_rms.item():.3f}, after/before={attn_after_ratio.item():.3f}, cos={attn_cosine.item():.3f}\n  FFN:       mean={ff_ratio_mean.item():.3f}, median={ff_ratio_median.item():.3f}, p90={ff_ratio_p90.item():.3f}, rms={ff_ratio_rms.item():.3f}, after/before={ff_after_ratio.item():.3f}, cos={ff_cosine.item():.3f}")

    #     self.debug_forward_count += 1

    #     return x

    def message(self,
                q_i: torch.Tensor,
                k_j: torch.Tensor,
                v_j: torch.Tensor,
                r: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: Optional[torch.Tensor],
                edge_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.has_pos_emb and r is not None:
            k_j = k_j + self.to_k_r(r).view(-1, self.num_heads, self.head_dim)
            v_j = v_j + self.to_v_r(r).view(-1, self.num_heads, self.head_dim)
        sim = (q_i * k_j).sum(dim=-1) * self.scale
        if edge_gate is not None:
            bias = torch.log(edge_gate + 1e-6) # [E]
            sim = sim + bias.unsqueeze(-1)
        attn = softmax(sim, index, ptr)
        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x_dst: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.num_heads * self.head_dim)
        g = torch.sigmoid(self.to_g(torch.cat([inputs, x_dst], dim=-1)))
        return inputs + g * (self.to_s(x_dst) - inputs)

    def _attn_block(self,
                    x_src: torch.Tensor,
                    x_dst: torch.Tensor,
                    r: Optional[torch.Tensor],
                    edge_index: torch.Tensor,
                    edge_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim)
        k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)
        v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)
        agg = self.propagate(edge_index=edge_index, x_dst=x_dst, q=q, k=k, v=v, r=r, edge_gate=edge_gate)
        return self.to_out(agg)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_mlp(x)
