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
import math
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from layers import AttentionLayer
from layers import FourierEmbedding
from layers import TransformerLayer
from utils import angle_between_2d_vectors
from utils import bipartite_dense_to_sparse
from utils import weight_init
from utils import wrap_angle


class TrajectoryScorer(nn.Module):
    def __init__(self, hidden_dim: int, num_future_steps: int, output_dim: int) -> None:
        super(TrajectoryScorer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_future_steps = num_future_steps
        self.output_dim = output_dim

        self.traj_proj = nn.Linear(output_dim, hidden_dim)
        self.traj_encoder = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                                   batch_first=False, dropout=0.0, bidirectional=False)
        self.traj_encoder_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weight_init)

    def forward(self,
                agent_context: torch.Tensor,
                trajectories: torch.Tensor) -> torch.Tensor:
        N_a, K, T_f, D = trajectories.shape

        # Encode trajectories with GRU (following QCNetDecoder.traj_emb pattern)
        traj = self.traj_proj(trajectories)               # [N_a, K, T_f, hidden_dim]
        traj = traj.reshape(N_a * K, T_f, self.hidden_dim).transpose(0, 1).contiguous()
        h0 = self.traj_encoder_h0.unsqueeze(1).expand(1, traj.size(1), self.hidden_dim).contiguous()
        traj_feat = self.traj_encoder(traj, h0)[1].squeeze(0).reshape(N_a, K, self.hidden_dim)

        # Expand agent context to match K modes
        ctx_exp = agent_context.unsqueeze(1).expand(-1, K, -1)  # [N_a, K, hidden_dim]

        # Concatenate and score (Linear supports high-dimensional input directly)
        combined = torch.cat([ctx_exp, traj_feat], dim=-1)      # [N_a, K, 2*hidden_dim]
        logits = self.scorer(combined).squeeze(-1)              # [N_a, K]

        return logits


class QCNetDiTBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(QCNetDiTBlock, self).__init__()
        self.hidden_dim = hidden_dim

        # ========================================================================
        # 优化点 1 (算子融合): 将 5 个独立的 adaLN 融合成一个巨大的 Linear，
        # 大幅减少 CUDA Kernel Launch，特别适合极宽的 5090
        # ========================================================================
        self.adaLN_all = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 10))

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        self.norm5 = nn.LayerNorm(hidden_dim)

        self.t2a_attn = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                       dropout=dropout, bipartite=True, has_pos_emb=True)
        self.pl2a_attn = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                        dropout=dropout, bipartite=True, has_pos_emb=True)
        self.a2a_attn = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                       dropout=dropout, bipartite=False, has_pos_emb=True)
        self.seg_attn = TransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        self.apply(weight_init)

    @staticmethod
    def _expand_edge_index(edge_index: torch.Tensor, K: int, bipartite: bool) -> torch.Tensor:
        if edge_index is None:
            return None
        E = edge_index.size(1)

        offsets = torch.arange(K, device=edge_index.device, dtype=edge_index.dtype).repeat_interleave(E)

        dst_base = edge_index[1].repeat(K) * K
        dst = dst_base + offsets

        if bipartite:
            src = edge_index[0].repeat(K)
        else:
            src_base = edge_index[0].repeat(K) * K
            src = src_base + offsets

        return torch.stack([src, dst], dim=0)

    def forward(self,
                x: torch.Tensor,
                t_emb_s: torch.Tensor,
                x_t: torch.Tensor,
                freq_pos_emb: torch.Tensor,
                r_t2a_exp: torch.Tensor,
                edge_index_t2a_exp: torch.Tensor,
                x_pl: torch.Tensor,
                r_pl2a_exp: torch.Tensor,
                edge_index_pl2a_exp: torch.Tensor,
                r_a2a_exp: torch.Tensor,
                edge_index_a2a_exp: torch.Tensor,
                K: int) -> torch.Tensor:
        N_a = x.size(0)

        x_flat = x.reshape(N_a * K, self.hidden_dim)                # [N_a*K, H]
        t_emb_flat = t_emb_s.reshape(N_a * K, self.hidden_dim)      # [N_a*K, H]

        shift_scale_all = self.adaLN_all(t_emb_flat)
        ss1, ss2, ss3, ss4, ss5 = shift_scale_all.chunk(5, dim=-1)

        # Step 1: Frequency token self-attention (add learnable freq_pos_emb, then self-attend)
        shift1, scale1 = ss1.chunk(2, dim=-1)
        x_mod = self.norm1(x_flat) * (1.0 + scale1) + shift1
        x_seg = x_mod.reshape(N_a, K, self.hidden_dim)
        seg_out = self.seg_attn(x=x_seg, context=freq_pos_emb)
        x_flat = x_flat + seg_out.reshape(N_a * K, self.hidden_dim)

        # Step 2: Temporal cross-attention (t2a)
        shift2, scale2 = ss2.chunk(2, dim=-1)
        x_mod = self.norm2(x_flat) * (1.0 + scale2) + shift2
        x_flat = x_flat + self.t2a_attn((x_t, x_mod), r_t2a_exp, edge_index_t2a_exp)

        # Step 3: Map cross-attention (pl2a)
        shift3, scale3 = ss3.chunk(2, dim=-1)
        x_mod = self.norm3(x_flat) * (1.0 + scale3) + shift3
        x_flat = x_flat + self.pl2a_attn((x_pl, x_mod), r_pl2a_exp, edge_index_pl2a_exp)

        # Step 4: Agent self-attention (a2a)
        shift4, scale4 = ss4.chunk(2, dim=-1)
        x_mod = self.norm4(x_flat) * (1.0 + scale4) + shift4
        x_flat = x_flat + self.a2a_attn(x_mod, r_a2a_exp, edge_index_a2a_exp)

        # Step 5: Feed-Forward Network
        shift5, scale5 = ss5.chunk(2, dim=-1)
        x_mod = self.norm5(x_flat) * (1.0 + scale5) + shift5
        x_flat = x_flat + self.ffn(x_mod)

        # Reshape back to [N_a, K, H]
        x = x_flat.reshape(N_a, K, self.hidden_dim)

        return x


class SymmetricVelocityHead(nn.Module):
    """Symmetric velocity prediction heads for orthogonal frequency bands.
    All bands use an identical 3-layer MLP with LayerNorm + SiLU for ODE stability.
    """
    def __init__(self, hidden_dim: int, output_dim: int, num_tokens: int = 3) -> None:
        super(SymmetricVelocityHead, self).__init__()
        
        self.heads = nn.ModuleList([
            self._build_head(hidden_dim, output_dim) for _ in range(num_tokens)
        ])

    def _build_head(self, hidden_dim: int, output_dim: int) -> nn.Sequential:
 
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),

            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),

            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N_a, K, H] -> velocity: [N_a, K, H]"""
        out_tokens = []
        for i in range(x.size(1)):
            out_tokens.append(self.heads[i](x[:, i]))
        return torch.stack(out_tokens, dim=1)


class QCNetFMDecoder(nn.Module):
    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(QCNetFMDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_r_t = 4
        input_dim_r_pl2a = 3
        input_dim_r_a2a = 3

        self.num_intents = 3  # K frequency tokens: low, mid, high

        self.freq_pos_emb = nn.Parameter(torch.randn(self.num_intents, hidden_dim))

        self.t_emb = FourierEmbedding(input_dim=1, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)

        self.r_t2a_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.r_pl2a_emb = FourierEmbedding(input_dim=input_dim_r_pl2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)

        self.blocks = nn.ModuleList(
            [QCNetDiTBlock(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout)
             for _ in range(num_layers)]
        )

        self.to_vel = SymmetricVelocityHead(hidden_dim=hidden_dim, output_dim=hidden_dim)

        # Trajectory scorer for multi-modal ranking
        self.scorer = TrajectoryScorer(
            hidden_dim=hidden_dim,
            num_future_steps=num_future_steps,
            output_dim=output_dim,
        )

        self.apply(weight_init)

    def _build_graph_context(self,
                             data: HeteroData,
                             scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # extract target agent state at t=0 (current position)
        pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim]
        head_m = data['agent']['heading'][:, self.num_historical_steps - 1]
        head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)

        # mask definitions
        mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True)

        # temporal: history steps -> target agents (t2a)
        pos_t = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
        head_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)
        edge_index_t2a = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))
        theta = data['agent']['heading'][:, self.num_historical_steps - 1]
        rel_pos_t2a = pos_t[edge_index_t2a[0]] - pos_m[edge_index_t2a[1]]
        rel_head_t2a = wrap_angle(head_t[edge_index_t2a[0]] - head_m[edge_index_t2a[1]])
        r_t2a = torch.stack(
            [torch.norm(rel_pos_t2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2a[1]], nbr_vector=rel_pos_t2a[:, :2]),
             rel_head_t2a,
             (edge_index_t2a[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
        r_t2a = self.r_t2a_emb(continuous_inputs=r_t2a, categorical_embs=None)

        # map: polygons -> target agents (pl2a)
        pos_pl = data['map_polygon']['position'][:, :self.input_dim]
        orient_pl = data['map_polygon']['orientation']
        edge_index_pl2a = radius(
            x=pos_m[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2m_radius,
            batch_x=data['agent']['batch'] if isinstance(data, Batch) else None,
            batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
            max_num_neighbors=300)
        edge_index_pl2a = edge_index_pl2a[:, mask_dst[edge_index_pl2a[1], 0]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_m[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_m[edge_index_pl2a[1]])
        r_pl2a = torch.stack(
            [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
             rel_orient_pl2a], dim=-1)
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)

        # agent -> agent (a2a)
        edge_index_a2a = radius_graph(
            x=pos_m[:, :2],
            r=self.a2m_radius,
            batch=data['agent']['batch'] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=300)
        edge_index_a2a = edge_index_a2a[:, mask_src[:, -1][edge_index_a2a[0]] & mask_dst[edge_index_a2a[1], 0]]
        rel_pos_a2a = pos_m[edge_index_a2a[0]] - pos_m[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_m[edge_index_a2a[0]] - head_m[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
             rel_head_a2a], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        # prepare context features for attention
        x_t_hist = scene_enc['x_a'].reshape(-1, self.hidden_dim)
        x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1]
        agent_batch = data['agent']['batch']

        # pre-expand spatial edges and relations for K frequency tokens (cached, not recomputed per layer/step)
        K = self.num_intents
        edge_index_t2a_exp = QCNetDiTBlock._expand_edge_index(edge_index_t2a, K, bipartite=True)
        edge_index_pl2a_exp = QCNetDiTBlock._expand_edge_index(edge_index_pl2a, K, bipartite=True)
        edge_index_a2a_exp = QCNetDiTBlock._expand_edge_index(edge_index_a2a, K, bipartite=False)
        r_t2a_exp = r_t2a.unsqueeze(0).expand(K, -1, -1).reshape(-1, r_t2a.size(-1)) if r_t2a is not None else None
        r_pl2a_exp = r_pl2a.unsqueeze(0).expand(K, -1, -1).reshape(-1, r_pl2a.size(-1)) if r_pl2a is not None else None
        r_a2a_exp = r_a2a.unsqueeze(0).expand(K, -1, -1).reshape(-1, r_a2a.size(-1)) if r_a2a is not None else None

        return {
            'pos_m': pos_m,
            'head_m': head_m,
            'head_vector_m': head_vector_m,
            'r_t2a_exp': r_t2a_exp,
            'edge_index_t2a_exp': edge_index_t2a_exp,
            'r_pl2a_exp': r_pl2a_exp,
            'edge_index_pl2a_exp': edge_index_pl2a_exp,
            'r_a2a_exp': r_a2a_exp,
            'edge_index_a2a_exp': edge_index_a2a_exp,
            'x_t_hist': x_t_hist,
            'x_pl': x_pl,
            'agent_batch': agent_batch,
        }

    def _forward_core(self,
                      ctx: Dict[str, torch.Tensor],
                      x_t: torch.Tensor,
                      t: torch.Tensor) -> torch.Tensor:
        """Latent-space velocity field prediction.

        Args:
            ctx:  pre-computed graph context from _build_graph_context
            x_t:  [N_a, K, H] latent frequency tokens (noised)
            t:    [N_a] per-agent diffusion timestep

        Returns:
            v_theta: [N_a, K, H] predicted velocity for each frequency band
        """
        N_a, K, H = x_t.shape
        device = x_t.device

        # ---- Step 1: Time embedding (per-agent, shared across K tokens) ----
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_emb = self.t_emb(continuous_inputs=t.unsqueeze(-1), categorical_embs=None)
        t_emb = t_emb[ctx['agent_batch']]                           # [N_a, H]
        t_emb_s = t_emb.unsqueeze(1).expand(N_a, K, H)              # [N_a, K, H]

        # ---- Step 2: DiT blocks (K frequency tokens in parallel) ----
        x = x_t
        for block in self.blocks:
            x = block(
                x=x, t_emb_s=t_emb_s,
                freq_pos_emb=self.freq_pos_emb, K=K,
                x_t=ctx['x_t_hist'],
                r_t2a_exp=ctx['r_t2a_exp'], edge_index_t2a_exp=ctx['edge_index_t2a_exp'],
                x_pl=ctx['x_pl'],
                r_pl2a_exp=ctx['r_pl2a_exp'], edge_index_pl2a_exp=ctx['edge_index_pl2a_exp'],
                r_a2a_exp=ctx['r_a2a_exp'], edge_index_a2a_exp=ctx['edge_index_a2a_exp'],
            )

        # ---- Step 3: Asymmetric velocity output (3 separate heads) ----
        v_theta = self.to_vel(x)  # [N_a, K, H]

        return v_theta

    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor],
                x_t: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        ctx = self._build_graph_context(data, scene_enc)
        return self._forward_core(ctx, x_t, t)

    @torch.no_grad()
    def sample(self,
               data: HeteroData,
               scene_enc: Mapping[str, torch.Tensor],
               num_modes: int = 6,
               num_steps: int = 10,
               latent_decoder: Optional[nn.Module] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample trajectories via latent flow matching, optionally decode to physical space.

        Args:
            data:            HeteroData graph for the scene
            scene_enc:       encoded scene features from QCNetEncoder
            num_modes:       number of trajectory modes to sample
            num_steps:       ODE integration steps
            latent_decoder:  optional VAE decoder (LatentSpaceDecoder) to map latent → physical

        Returns:
            trajectories: [N_a, num_modes, ...]  physical trajectories (if decoder given)
                                                   or latent tokens [N_a, num_modes, K, H]
            pi:          [N_a, num_modes]          selection probabilities
        """
        ctx = self._build_graph_context(data, scene_enc)
        N_a = ctx['pos_m'].size(0)
        device = ctx['pos_m'].device

        # time discretisation: t ∈ [0, 1)
        dt = 1.0 / num_steps
        t_grid = torch.linspace(0.0, 1.0 - dt, num_steps, device=device)

        B = int(ctx['agent_batch'].max().item()) + 1 if ctx['agent_batch'].numel() > 0 else 1

        all_outputs: List[torch.Tensor] = []
        t_cur_tensor = torch.empty(B, device=device)
        t_next_tensor = torch.empty(B, device=device)

        for _ in range(num_modes):
            # initial noise in latent space: x_0 ~ N(0, I)  → [N_a, K, H]
            x_t = torch.randn(N_a, self.num_intents, self.hidden_dim, device=device)
            for t_val in t_grid:
                t_cur_tensor.fill_(t_val)
                t_next_tensor.fill_(t_val + dt)

                # Step 1: Velocity at current position
                v1 = self._forward_core(ctx, x_t, t_cur_tensor)
                # Step 2: Euler probe
                x_euler = x_t + v1 * dt
                # Step 3: Velocity at probe position
                v2 = self._forward_core(ctx, x_euler, t_next_tensor)
                # Step 4: Heun correction
                v_heun = 0.5 * (v1 + v2)
                x_t = x_t + v_heun * dt

            # Decode to physical space if decoder is provided
            if latent_decoder is not None:
                output = latent_decoder(x_t)  # [N_a, T_f, D]
            else:
                output = x_t  # [N_a, K, H] (keep latent for debugging)
            all_outputs.append(output)

        trajectories = torch.stack(all_outputs, dim=1)  # [N_a, num_modes, ...]

        # Score requires physical trajectories; skip scoring if in latent space
        if latent_decoder is not None:
            agent_context = scene_enc['x_a'][:, -1, :]
            logits = self.scorer(agent_context, trajectories)
            pi = F.softmax(logits, dim=-1)
        else:
            pi = torch.ones(N_a, num_modes, device=device) / num_modes

        return trajectories, pi

    def compute_scorer_loss(self,
                            trajectories: torch.Tensor,
                            target: torch.Tensor,
                            predict_mask: torch.Tensor,
                            agent_context: torch.Tensor) -> torch.Tensor:

        # 1. 获取打分器的预测 logits (未经 softmax 的原始得分)
        logits = self.scorer(agent_context, trajectories)  # [N_a, K]

        # 2. 寻找每个智能体未来轨迹的最后一个有效时间步 (FDE 计算点)
        # 即使 predict_mask 中间有稀疏的 False，flip + argmax 也能准确找到最右侧的 True
        T_f = predict_mask.size(1)
        last_valid_idx = T_f - 1 - predict_mask.flip(dims=[-1]).long().argmax(dim=-1)  # [N_a]
        last_valid_idx = last_valid_idx.clamp(min=0)  # 安全保护机制

        N_a = trajectories.size(0)
        batch_idx = torch.arange(N_a, device=trajectories.device)

        # 3. 提取各个智能体在最后有效时刻的预测坐标和真实坐标
        # 使用 [batch_idx, :, last_valid_idx, :] 确保提取出的形状是绝对干净的 [N_a, K, D]
        traj_end = trajectories[batch_idx, :, last_valid_idx, :]   # [N_a, K, D]
        target_end = target[batch_idx, last_valid_idx, :]          # [N_a, D]

        # 4. 计算终点误差 (FDE: Final Displacement Error)
        fde = torch.norm(traj_end - target_end.unsqueeze(1), dim=-1)  # [N_a, K]
        fde = fde * 10.0

        # 5. 掩码过滤：只对未来有有效运动数据 (非全 0 padding) 的智能体计算 Loss
        valid = predict_mask.any(dim=-1)  # [N_a]
        if not valid.any():
            # 如果整个 Batch 都没有有效目标，返回带梯度的 0 防治报错
            return torch.tensor(0.0, device=trajectories.device, requires_grad=True)

        # 提取有效智能体的 FDE 和 Logits
        valid_fde = fde[valid]       # [有效N_a, K]
        valid_logits = logits[valid] # [有效N_a, K]

        # 引入温度系数 (Temperature, alpha)，控制惩罚的严厉程度
        # alpha 越大，越接近 hard label; alpha 越小，越平滑。通常取 1.0 ~ 2.0
        alpha = 1.5

        # 把距离 (FDE) 转换成目标概率分布 (Target Probabilities)
        # 距离越小，取负数后越大，Softmax 后的概率就越高！
        target_probs = F.softmax(-alpha * valid_fde, dim=-1)  # [有效N_a, K]

        # 7. 计算交叉熵损失
        # 此时 logits 是 [有效N_a, K]，positive_idx 是纯粹的一维整数索引 [有效N_a]
        loss = F.cross_entropy(valid_logits, target_probs)

        return loss