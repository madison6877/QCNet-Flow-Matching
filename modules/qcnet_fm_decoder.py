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

        self.adaLN_t2a = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.adaLN_pl2a = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.adaLN_a2a = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.adaLN_seg = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.xm_cond_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)

        self.t2a_attn = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                       dropout=dropout, bipartite=True, has_pos_emb=True)
        self.pl2a_attn = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                        dropout=dropout, bipartite=True, has_pos_emb=True)
        self.a2a_attn = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                       dropout=dropout, bipartite=False, has_pos_emb=True)
        self.seg_attn = TransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

        self.a2a_dynamic_align = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.alpha_bias=nn.Parameter(torch.tensor(0.3))
        self.alpha_scale=nn.Parameter(torch.tensor(10.0))

        self.apply(weight_init)

    def _init_adaln(self):
        for ada_layer in [self.adaLN_t2a, self.adaLN_pl2a, self.adaLN_a2a, self.adaLN_seg]:
            nn.init.zeros_(ada_layer[-1].weight)
            nn.init.zeros_(ada_layer[-1].bias)

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
                t_scalar: torch.Tensor, 
                x_t: torch.Tensor,
                freq_pos_emb: torch.Tensor,
                r_t2a_exp: torch.Tensor,
                edge_index_t2a_exp: torch.Tensor,
                x_pl: torch.Tensor,
                r_pl2a_exp: torch.Tensor,
                edge_index_pl2a_exp: torch.Tensor,
                r_a2a_exp: torch.Tensor,
                edge_index_a2a_exp: torch.Tensor,
                edge_threat_exp: torch.Tensor,
                edge_map_exp: torch.Tensor,
                x_m: torch.Tensor,
                K: int,
                use_xm:bool = True,
                use_history: bool = True, 
                use_map: bool = True,     
                use_agent: bool = True    
                ) -> torch.Tensor:
        
        N_a = x.size(0)
        x_flat = x.reshape(N_a * K, self.hidden_dim)                
        t_emb_flat = t_emb_s.reshape(N_a * K, self.hidden_dim)

        t_scalar_exp = t_scalar.unsqueeze(1).expand(-1, K).reshape(N_a * K, 1) 

        scale = F.softplus(self.alpha_bias) + 1e-4
        bias = torch.sigmoid(self.alpha_bias)
        alpha=torch.sigmoid(scale * (t_scalar_exp - bias))

        freq_emb_flat = freq_pos_emb.unsqueeze(0).expand(N_a, K, self.hidden_dim).reshape(N_a * K, self.hidden_dim)
        x_m_flat = x_m.unsqueeze(1).expand(-1, K, -1).reshape(N_a * K, self.hidden_dim)
        
        cond1 = t_emb_flat
        if use_xm:
            cond1 = cond1 + self.xm_cond_proj(x_m_flat)
        if K > 1:
            cond2 = cond1 + freq_emb_flat

        if use_history:
            ssg_t2a = self.adaLN_t2a(cond1).chunk(6, dim=-1)
            shift1_a, scale1_a, gate1_a = ssg_t2a[0], ssg_t2a[1], ssg_t2a[2]
            shift1_f, scale1_f, gate1_f = ssg_t2a[3], ssg_t2a[4], ssg_t2a[5]
            x_mod = self.norm1(x_flat) * (1.0 + scale1_a) + shift1_a
            x_src_norm = self.t2a_attn.attn_prenorm_x_src(x_t)
            r_norm = self.t2a_attn.attn_prenorm_r(r_t2a_exp) if (self.t2a_attn.has_pos_emb and r_t2a_exp is not None) else None
            attn_out = self.t2a_attn._attn_block(x_src=x_src_norm, x_dst=x_mod, r=r_norm, edge_index=edge_index_t2a_exp)
            x_flat = x_flat + gate1_a * attn_out
            ff_in = self.t2a_attn.ff_prenorm(x_flat) * (1.0 + scale1_f) + shift1_f
            ff_out = self.t2a_attn._ff_block(ff_in)
            x_flat = x_flat + gate1_f * ff_out

        if use_map:
            ssg_pl2a = self.adaLN_pl2a(cond1).chunk(6, dim=-1)
            shift2_a, scale2_a, gate2_a = ssg_pl2a[0], ssg_pl2a[1], ssg_pl2a[2]
            shift2_f, scale2_f, gate2_f = ssg_pl2a[3], ssg_pl2a[4], ssg_pl2a[5]
            x_mod = self.norm2(x_flat) * (1.0 + scale2_a) + shift2_a
            x_src_norm = self.pl2a_attn.attn_prenorm_x_src(x_pl)
            r_norm = self.pl2a_attn.attn_prenorm_r(r_pl2a_exp) if (self.pl2a_attn.has_pos_emb and r_pl2a_exp is not None) else None
            attn_out = self.pl2a_attn._attn_block(x_src=x_src_norm, x_dst=x_mod, r=r_norm, edge_index=edge_index_pl2a_exp, edge_gate=edge_map_exp)
            x_flat = x_flat + gate2_a * attn_out
            ff_in = self.pl2a_attn.ff_prenorm(x_flat) * (1.0 + scale2_f) + shift2_f
            ff_out = self.pl2a_attn._ff_block(ff_in)
            x_flat = x_flat + gate2_f * ff_out

        if use_agent:
            ss3_a2a = self.adaLN_a2a(cond1).chunk(6, dim=-1)
            shift3_a, scale3_a, gate3_a = ss3_a2a[0], ss3_a2a[1], ss3_a2a[2]
            shift3_f, scale3_f, gate3_f = ss3_a2a[3], ss3_a2a[4], ss3_a2a[5]
            x_mod = self.norm3(x_flat) * (1.0 + scale3_a) + shift3_a
            r_norm = self.a2a_attn.attn_prenorm_r(r_a2a_exp) if (self.a2a_attn.has_pos_emb and r_a2a_exp is not None) else None
            x_m_normed = self.norm3(x_m_flat) * (1.0 + scale3_a) + shift3_a
            x_src = (1 - alpha) * self.a2a_dynamic_align(x_m_normed)  + alpha * x_mod
            #x_src = self.a2a_attn.attn_prenorm_x_src(x_m_flat)
            attn_out = self.a2a_attn._attn_block(x_src=x_src, x_dst=x_mod, r=r_norm, edge_index=edge_index_a2a_exp, edge_gate=edge_threat_exp)
            x_flat = x_flat + gate3_a * attn_out
            ff_in = self.a2a_attn.ff_prenorm(x_flat) * (1.0 + scale3_f) + shift3_f  
            ff_out = self.a2a_attn._ff_block(ff_in)
            x_flat = x_flat + gate3_f * ff_out

        if K > 1:
            ssg_ffn = self.adaLN_seg(cond2).chunk(6, dim=-1)
            shift4_a, scale4_a, gate4_a = ssg_ffn[0], ssg_ffn[1], ssg_ffn[2]
            shift4_f, scale4_f, gate4_f = ssg_ffn[3], ssg_ffn[4], ssg_ffn[5]
            x_mod = self.norm4(x_flat) * (1.0 + scale4_a) + shift4_a
            x_mod_seg = x_mod.reshape(N_a, K, self.hidden_dim)
            attn_out, _ = self.seg_attn.attn(query=x_mod_seg, key=x_mod_seg, value=x_mod_seg, need_weights=False)
            x_flat = x_flat + gate4_a * attn_out.reshape(N_a * K, self.hidden_dim)
            ff_in = self.seg_attn.ffn_prenorm(x_flat) * (1.0 + scale4_f) + shift4_f
            ff_out = self.seg_attn.ffn(ff_in)
            x_flat = x_flat + gate4_f * ff_out

        return x_flat.reshape(N_a, K, self.hidden_dim)

class AsymmetricVelocityHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, num_intents: int = 3) -> None:
        super(AsymmetricVelocityHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_intents = num_intents
        self.shortcut = nn.Linear(hidden_dim, output_dim)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.apply(weight_init)

    def _init_weights(self):
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.shortcut.weight)
        nn.init.zeros_(self.shortcut.bias)
        nn.init.zeros_(self.time_proj[-1].weight)
        nn.init.zeros_(self.time_proj[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, x_m: torch.Tensor) -> torch.Tensor:
        N_a, K, _ = x.shape
        x_m_exp = x_m.unsqueeze(1).expand(-1, K, -1)
        t_cond = self.time_proj(t_emb+x_m_exp)        # [N_a, K, hidden_dim * 2]
        shift, scale = t_cond.chunk(2, dim=-1)  # [N_a, K, hidden_dim]
        x_modulated = self.norm(x) * (1.0 + scale) + shift
        base_vel = self.shortcut(x) 
        res_vel = self.residual(x_modulated)
        return base_vel + res_vel

class QCNetFMDecoder(nn.Module):
    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
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
                 dropout: float,
                 vae_num_intents: int = 3) -> None:
        super(QCNetFMDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
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

        self.num_intents = vae_num_intents  # K frequency tokens: configurable via hyperparameter

        self.x_proj_in = nn.Linear(self.latent_dim, self.hidden_dim)

        self.freq_pos_emb = nn.Parameter(0.5 * torch.randn(self.num_intents, hidden_dim))

        self.t_emb = FourierEmbedding(input_dim=1, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)

        self.r_t2a_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)

        self.r_pl2a_emb = FourierEmbedding(input_dim=input_dim_r_pl2a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)

        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)

        #self.threat_fourier = FourierEmbedding(input_dim=5, hidden_dim=hidden_dim,  num_freq_bands=num_freq_bands)
        
        #self.map_fourier = FourierEmbedding(input_dim=3, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)

        # self.threat_net = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.LayerNorm(hidden_dim // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim // 2, 1),
        #     nn.Sigmoid()
        # )
        
        # self.map_net = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.LayerNorm(hidden_dim // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim // 2, 1),
        #     nn.Sigmoid()
        # )

        self.blocks = nn.ModuleList(
            [QCNetDiTBlock(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout)
             for _ in range(num_layers)]
        )

        self.to_vel = AsymmetricVelocityHead(hidden_dim=hidden_dim, output_dim=latent_dim,
                                             num_intents=self.num_intents)

        # Trajectory scorer for multi-modal ranking
        self.scorer = TrajectoryScorer(
            hidden_dim=hidden_dim,
            num_future_steps=num_future_steps,
            output_dim=output_dim,
        )

        self.apply(weight_init)
        for block in self.blocks: 
            block._init_adaln()
            nn.init.eye_(block.a2a_dynamic_align.weight)
        self.to_vel._init_weights()

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
        edge_threat_exp = None
        edge_map_exp = None
        pinn_loss = 0
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
            'edge_threat_exp': edge_threat_exp,
            'edge_map_exp': edge_map_exp,
            'pinn_loss': pinn_loss
        }

    # def _build_graph_context(self,
    #                          data: HeteroData,
    #                          scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     # extract target agent state at t=0 (current position)
    #     pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim]
    #     head_m = data['agent']['heading'][:, self.num_historical_steps - 1]
    #     head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)

    #     # mask definitions
    #     mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
    #     mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
    #     mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True)

    #     # temporal: history steps -> target agents (t2a)
    #     pos_t = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
    #     head_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)
    #     edge_index_t2a = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))
    #     theta = data['agent']['heading'][:, self.num_historical_steps - 1]
    #     rel_pos_t2a = pos_t[edge_index_t2a[0]] - pos_m[edge_index_t2a[1]]
    #     rel_head_t2a = wrap_angle(head_t[edge_index_t2a[0]] - head_m[edge_index_t2a[1]])
    #     r_t2a = torch.stack(
    #         [torch.norm(rel_pos_t2a[:, :2], p=2, dim=-1),
    #          angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2a[1]], nbr_vector=rel_pos_t2a[:, :2]),
    #          rel_head_t2a,
    #          (edge_index_t2a[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
    #     r_t2a = self.r_t2a_emb(continuous_inputs=r_t2a, categorical_embs=None)

    #     # map: polygons -> target agents (pl2a)
    #     pos_pl = data['map_polygon']['position'][:, :self.input_dim]
    #     orient_pl = data['map_polygon']['orientation']
    #     edge_index_pl2a = radius(
    #         x=pos_m[:, :2],
    #         y=pos_pl[:, :2],
    #         r=self.pl2m_radius,
    #         batch_x=data['agent']['batch'] if isinstance(data, Batch) else None,
    #         batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
    #         max_num_neighbors=300)
    #     edge_index_pl2a = edge_index_pl2a[:, mask_dst[edge_index_pl2a[1], 0]]
    #     src_pl, dst_a = edge_index_pl2a[0], edge_index_pl2a[1]
    #     rel_pos_pl2a = pos_pl[src_pl] - pos_m[dst_a]
    #     rel_orient_pl2a = wrap_angle(orient_pl[src_pl] - head_m[dst_a])
    #     r_pl2a = torch.stack(
    #         [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
    #          angle_between_2d_vectors(ctr_vector=head_vector_m[dst_a], nbr_vector=rel_pos_pl2a[:, :2]),
    #          rel_orient_pl2a], dim=-1)
    #     r_pl2a_emb_feat = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
    #     if rel_pos_pl2a.size(0) > 0:
    #         # ----------------------------------------------------
    #         # 🌟 地图门控双路融合 (Map Dual-Stream Gating)
    #         # ----------------------------------------------------
    #         # 路一：物理极坐标 (距离 + 夹角)
    #         dist_pl2a = torch.norm(rel_pos_pl2a, p=2, dim=-1, keepdim=True)
    #         azimuth_pl2a = wrap_angle(torch.atan2(rel_pos_pl2a[:, 1], rel_pos_pl2a[:, 0]) - head_m[dst_a]).unsqueeze(-1)
            
    #         map_phys_raw = torch.cat([dist_pl2a, azimuth_pl2a, rel_orient_pl2a.unsqueeze(-1)], dim=-1)
    #         # 使用一个新的 map_fourier 网络将 3维物理量 升维
    #         map_phys_emb = self.map_fourier(continuous_inputs=map_phys_raw, categorical_embs=None)

    #         # 路二：深层意图对齐
    #         # x_pl [N_pl, hidden_dim] 是地图特征
    #         x_src_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1][src_pl] 
    #         x_dst_a = scene_enc['x_a'][:, -1, :][dst_a]
            
    #         # 使用相对编码 r_pl2a_emb_feat 对齐坐标系
    #         aligned_x_src_pl = x_src_pl + r_pl2a_emb_feat
    #         map_deep_feat = aligned_x_src_pl - x_dst_a
            
    #         # 融合并打分 (分数越高代表该地图元素越重要)
    #         map_combined_feat = map_phys_emb + map_deep_feat
    #         map_importance_scores = self.map_net(map_combined_feat).squeeze(-1) 
            
    #         # 【可选】硬截断：只保留对主车最重要的 Top-K_map 个地图元素
    #         K_map_max = 16
    #         K_map_min = 4
    #         map_threshold = 0.01
    #         sort_key_map = dst_a.double() * 100.0 - map_importance_scores.double()
    #         sorted_indices_map = torch.argsort(sort_key_map)
    #         sorted_dst_map = dst_a[sorted_indices_map]
            
    #         _, counts_map = torch.unique_consecutive(sorted_dst_map, return_counts=True)
    #         local_ranks_map = torch.cat([torch.arange(c, device=pos_m.device) for c in counts_map])

    #         mask_min_map = local_ranks_map < K_map_min
    #         mask_threat_map = map_importance_scores[sorted_indices_map] > map_threshold
    #         mask_max_map = local_ranks_map < K_map_max
            
    #         keep_mask_map = (mask_min_map | mask_threat_map) & mask_max_map
    #         keep_indices_map = sorted_indices_map[keep_mask_map]
            
    #         edge_index_pl2a = edge_index_pl2a[:, keep_indices_map]
    #         r_pl2a = r_pl2a_emb_feat[keep_indices_map]
    #         edge_map_scores = map_importance_scores[keep_indices_map]
            
    #         # 软门控稀释原始相对编码
    #         r_pl2a = r_pl2a * edge_map_scores.unsqueeze(-1)
    #     else:
    #         r_pl2a = r_pl2a_emb_feat
    #         edge_map_scores = torch.empty(0, device=pos_m.device)
    #     # ==========================================================

    #     # agent -> agent (a2a)
    #     edge_index_a2a = radius_graph(
    #         x=pos_m[:, :2],
    #         r=self.a2m_radius,
    #         batch=data['agent']['batch'] if isinstance(data, Batch) else None,
    #         loop=False,
    #         max_num_neighbors=300)
    #     edge_index_a2a = edge_index_a2a[:, mask_src[:, -1][edge_index_a2a[0]] & mask_dst[edge_index_a2a[1], 0]]
    #     rel_pos_a2a = pos_m[edge_index_a2a[0]] - pos_m[edge_index_a2a[1]]
    #     rel_head_a2a = wrap_angle(head_m[edge_index_a2a[0]] - head_m[edge_index_a2a[1]])
    #     r_a2a = torch.stack(
    #         [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
    #          angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
    #          rel_head_a2a], dim=-1)
    #     r_a2a_emb_feat = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

    #    # ==========================================================
    #     # 🌟 插件模块：极坐标 PINN 威胁网络 + 节点级门控计算
    #     # ==========================================================
    #     src_a, dst_a = edge_index_a2a[0], edge_index_a2a[1]
        
    #     # 提取速度特征
    #     vel_m = data['agent']['velocity'][:, self.num_historical_steps - 1, :2]
    #     rel_vel_a2a = vel_m[src_a] - vel_m[dst_a]
    #     rel_pos_a2a = pos_m[src_a] - pos_m[dst_a]
    #     rel_head_a2a = wrap_angle(head_m[src_a] - head_m[dst_a])

    #     if rel_pos_a2a.size(0) > 0:

    #         # 1. 位置转化为：距离 (Distance) 和 相对主车车头的方位角 (Azimuth)
    #         dist_a2a = torch.norm(rel_pos_a2a, p=2, dim=-1, keepdim=True)
    #         azimuth_a2a = wrap_angle(torch.atan2(rel_pos_a2a[:, 1], rel_pos_a2a[:, 0]) - head_m[dst_a]).unsqueeze(-1)
            
    #         # 2. 速度转化为：相对速率 (Speed) 和 速度向量方位角 (Velocity Angle)
    #         speed_a2a = torch.norm(rel_vel_a2a, p=2, dim=-1, keepdim=True)
    #         vel_angle_a2a = wrap_angle(torch.atan2(rel_vel_a2a[:, 1], rel_vel_a2a[:, 0]) - head_m[dst_a]).unsqueeze(-1)
            
    #         # 组装 5 维极坐标特征！(维度依然是 5，无需修改初始化网络)
    #         phys_feat_raw = torch.cat([dist_a2a, azimuth_a2a, speed_a2a, vel_angle_a2a, rel_head_a2a.unsqueeze(-1)], dim=-1)
    #         phys_emb = self.threat_fourier(continuous_inputs=phys_feat_raw, categorical_embs=None) # [E, hidden_dim]
    #         x_src_a = scene_enc['x_a'][:, -1, :][src_a]  # 处于 src 的局部坐标系
    #         x_dst_a = scene_enc['x_a'][:, -1, :][dst_a]  # 处于 dst 的局部坐标系
    #         aligned_x_src = x_src_a + r_a2a_emb_feat
    #         deep_feat = aligned_x_src - x_dst_a
    #         combined_feat = phys_emb + deep_feat
    #         threat_scores = self.threat_net(combined_feat).squeeze(-1)  # [E]
            
    #         # 物理公式运算 (用于计算 Loss，不受极坐标影响，依然用向量点积最快)
    #         with torch.no_grad():
    #             dist_val = dist_a2a.squeeze(-1)
    #             approach_speed = - (rel_vel_a2a * rel_pos_a2a).sum(dim=-1) / (dist_val + 1e-5)
    #             ttc = dist_val / (F.relu(approach_speed) + 1e-5)
    #             physics_danger = (ttc < 3.5) & (dist_val < 20.0)
            
    #         # 计算 PINN 损失
    #         miss_penalty = physics_danger.float() * (1.0 - threat_scores).pow(2)
    #         sparsity_penalty = (~physics_danger).float() * threat_scores.abs()
    #         pinn_loss = miss_penalty.mean() + 0.1 * sparsity_penalty.mean()
            
    #         K_neighbors = 16
    #         sort_key = dst_a.double() * 100.0 - threat_scores.double()
    #         sorted_indices = torch.argsort(sort_key)
    #         sorted_dst = dst_a[sorted_indices]
    #         _, counts = torch.unique_consecutive(sorted_dst, return_counts=True)
    #         local_ranks = torch.cat([torch.arange(c, device=pos_m.device) for c in counts])
    #         keep_mask = local_ranks < K_neighbors
    #         keep_indices = sorted_indices[keep_mask]
            
    #         edge_index_a2a = edge_index_a2a[:, keep_indices]
    #         r_a2a = r_a2a_emb_feat[keep_indices]

    #         edge_threat_scores = threat_scores[keep_indices]
    #     # ==========================================================

    #     # prepare context features for attention
    #     x_t_hist = scene_enc['x_a'].reshape(-1, self.hidden_dim)
    #     x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1]
    #     agent_batch = data['agent']['batch']

    #     # pre-expand spatial edges and relations for K frequency tokens (cached, not recomputed per layer/step)
    #     K = self.num_intents
    #     edge_index_t2a_exp = QCNetDiTBlock._expand_edge_index(edge_index_t2a, K, bipartite=True)
    #     edge_index_pl2a_exp = QCNetDiTBlock._expand_edge_index(edge_index_pl2a, K, bipartite=True)
    #     edge_index_a2a_exp = QCNetDiTBlock._expand_edge_index(edge_index_a2a, K, bipartite=False)
    #     r_t2a_exp = r_t2a.unsqueeze(0).expand(K, -1, -1).reshape(-1, r_t2a.size(-1)) if r_t2a is not None else None
    #     r_pl2a_exp = r_pl2a.unsqueeze(0).expand(K, -1, -1).reshape(-1, r_pl2a.size(-1)) if r_pl2a is not None else None
    #     r_a2a_exp = r_a2a.unsqueeze(0).expand(K, -1, -1).reshape(-1, r_a2a.size(-1)) if r_a2a is not None else None

    #     edge_threat_exp = edge_threat_scores.unsqueeze(0).expand(K, -1).reshape(-1) if 'edge_threat_scores' in locals() else None
    #     edge_map_exp = edge_map_scores.unsqueeze(0).expand(K, -1).reshape(-1) if 'edge_map_scores' in locals() else None

    #     return {
    #         'pos_m': pos_m,
    #         'head_m': head_m,
    #         'head_vector_m': head_vector_m,
    #         'r_t2a_exp': r_t2a_exp,
    #         'edge_index_t2a_exp': edge_index_t2a_exp,
    #         'r_pl2a_exp': r_pl2a_exp,
    #         'edge_index_pl2a_exp': edge_index_pl2a_exp,
    #         'r_a2a_exp': r_a2a_exp,
    #         'edge_index_a2a_exp': edge_index_a2a_exp,
    #         'x_t_hist': x_t_hist,
    #         'x_pl': x_pl,
    #         'agent_batch': agent_batch,
    #         'edge_threat_exp': edge_threat_exp,
    #         'edge_map_exp': edge_map_exp,
    #         'pinn_loss': pinn_loss
    #     }

    def _forward_core(self,
                      ctx: Dict[str, torch.Tensor],
                      x_t: torch.Tensor,
                      t: torch.Tensor) -> torch.Tensor:
        
        N_a, K, latent_dim = x_t.shape
        H = self.hidden_dim
        device = x_t.device

        # ---- Step 1: Time embedding ----
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        t_emb = self.t_emb(continuous_inputs=t.unsqueeze(-1), categorical_embs=None)
        t_emb_s = t_emb.unsqueeze(1).expand(N_a, K, H)         
        
        x = self.x_proj_in(x_t)
        x_t_hist_unflat = ctx['x_t_hist'].view(N_a, self.num_historical_steps, H)
        x_m = x_t_hist_unflat[:, -1, :]

        # ---- Step 2: DiT blocks (K frequency tokens in parallel) ----
        for layer_idx, block in enumerate(self.blocks):
            if layer_idx == 0:
                use_xm = False
                use_history = True
                use_map = True
                use_agent = False
            elif layer_idx == 1:
                use_xm = True
                use_history = True
                use_map = False
                use_agent = True

            elif layer_idx == 2:
                use_xm = True
                use_history = False
                use_map = False
                use_agent = True

            x = block(
                x=x, t_emb_s=t_emb_s,
                t_scalar=t, # 🌟 将最原始的 0~1 时间标量传进去！
                freq_pos_emb=self.freq_pos_emb, K=K,
                x_t=ctx['x_t_hist'],
                r_t2a_exp=ctx['r_t2a_exp'], edge_index_t2a_exp=ctx['edge_index_t2a_exp'],
                x_pl=ctx['x_pl'],
                r_pl2a_exp=ctx['r_pl2a_exp'], edge_index_pl2a_exp=ctx['edge_index_pl2a_exp'],
                r_a2a_exp=ctx['r_a2a_exp'], edge_index_a2a_exp=ctx['edge_index_a2a_exp'],
                edge_threat_exp=ctx['edge_threat_exp'],
                edge_map_exp=ctx['edge_map_exp'],
                x_m=x_m,
                use_xm = use_xm,
                use_history=use_history,
                use_map=use_map,
                use_agent=use_agent
            )

        # ---- Step 3: Asymmetric velocity output (K separate heads) ----
        v_theta = self.to_vel(x, t_emb_s,x_m)  

        return v_theta

    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor],
                x_t: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        ctx = self._build_graph_context(data, scene_enc)
        v_theta = self._forward_core(ctx, x_t, t)
        pinn_loss = ctx.get('pinn_loss', torch.tensor(0.0, device=x_t.device))
        return v_theta, pinn_loss

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

        all_outputs: List[torch.Tensor] = []
        t_cur_tensor = torch.empty(N_a, device=device)
        t_next_tensor = torch.empty(N_a, device=device)

        for _ in range(num_modes):
            # initial noise in latent space: x_0 ~ N(0, I)  → [N_a, K, H]
            x_t = torch.randn(N_a, self.num_intents, self.latent_dim, device=device)
            scale = torch.tensor([0.9458926320075989, 0.6536939144134521, 1.17396879196167, 0.5958303809165955, 0.6562486886978149],device=device).reshape(1, 1, -1)
            x_t = x_t * scale
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