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


class StructuredSceneAggregator(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.history_query = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.map_query = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.agent_query = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))

        self.condition_type_embedding = nn.Parameter(torch.zeros(3, hidden_dim))

        self.history_attn = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                           dropout=dropout, bipartite=True, has_pos_emb=True)
        self.map_attn = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                       dropout=dropout, bipartite=True, has_pos_emb=True)
        self.agent_attn = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                         dropout=dropout, bipartite=True, has_pos_emb=True)
        
        self.history_out_norm = nn.LayerNorm(hidden_dim)
        self.map_out_norm = nn.LayerNorm(hidden_dim)
        self.agent_out_norm = nn.LayerNorm(hidden_dim)

        self.apply(weight_init)
        nn.init.normal_(self.condition_type_embedding, mean=0.0, std=0.02)

    @staticmethod
    def _aggregate(attn: AttentionLayer, source: torch.Tensor, query: torch.Tensor,
                   relation: Optional[torch.Tensor], edge_index: Optional[torch.Tensor],
                   edge_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return query
        
        return attn((source, query), relation, edge_index, edge_gate=edge_gate)

    def forward(self, x_m: torch.Tensor, x_t_hist: torch.Tensor, x_pl: torch.Tensor,
                r_t2a: Optional[torch.Tensor], edge_index_t2a: Optional[torch.Tensor],
                r_pl2a: Optional[torch.Tensor], edge_index_pl2a: Optional[torch.Tensor],
                r_a2a: Optional[torch.Tensor], edge_index_a2a: Optional[torch.Tensor],
                edge_map: Optional[torch.Tensor] = None,
                edge_threat: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if x_m.ndim != 2 or x_m.size(-1) != self.hidden_dim:
            raise ValueError(f"x_m 应为 [N_a,{self.hidden_dim}]，实际为 {tuple(x_m.shape)}")
        
        q_hist = self.history_query(x_m) + self.condition_type_embedding[0]
        q_map = self.map_query(x_m) + self.condition_type_embedding[1]
        q_agent = self.agent_query(x_m) + self.condition_type_embedding[2]
        c_hist = self._aggregate(self.history_attn, x_t_hist, q_hist, r_t2a, edge_index_t2a)
        c_map = self._aggregate(self.map_attn, x_pl, q_map, r_pl2a, edge_index_pl2a, edge_map)
        c_agent = self._aggregate(self.agent_attn, x_m, q_agent, r_a2a, edge_index_a2a, edge_threat)

        return {"c_hist": self.history_out_norm(c_hist), "c_map": self.map_out_norm(c_map),"c_agent": self.agent_out_norm(c_agent)}


class LatentAdaLNBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.time_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.history_mod = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(),
                                         nn.Linear(hidden_dim, hidden_dim * 6))
        self.map_mod = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim * 6))
        self.agent_mod = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(),
                                       nn.Linear(hidden_dim, hidden_dim * 6))
        
        self.latent_mixer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU(),
                                          nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        
        self.apply(weight_init)
        self._init_adaln_zero()

    def _init_adaln_zero(self) -> None:
        for module in (self.time_mod, self.history_mod, self.map_mod, self.agent_mod):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_hist: torch.Tensor,
                c_map: torch.Tensor, c_agent: torch.Tensor) -> torch.Tensor:
        modulation = self.time_mod(t_emb) + self.history_mod(c_hist) + self.map_mod(c_map) + self.agent_mod(c_agent)
        shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)

        h1 = self.norm1(x) * (1.0 + scale1) + shift1
        x = x + gate1 * self.latent_mixer(h1)
        h2 = self.norm2(x) * (1.0 + scale2) + shift2
        x = x + gate2 * self.ffn(h2)

        return x
    

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
        self.num_intents = vae_num_intents

        self.x_proj_in = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

        self.intent_embedding = nn.Parameter(torch.zeros(self.num_intents, hidden_dim))
        self.t_emb = FourierEmbedding(input_dim=1, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t2a_emb = FourierEmbedding(input_dim=4, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2a_emb = FourierEmbedding(input_dim=3, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=3, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)

        self.scene_aggregator = StructuredSceneAggregator(hidden_dim, num_heads, head_dim, dropout)
        
        self.blocks = nn.ModuleList([LatentAdaLNBlock(hidden_dim, dropout) for _ in range(num_layers)])

        self.final_norm = nn.LayerNorm(hidden_dim)

        self.to_vel = nn.Linear(hidden_dim, latent_dim)

        self.scorer = TrajectoryScorer(hidden_dim, num_future_steps, output_dim)

        self.apply(weight_init)
        if self.num_intents > 1:
            nn.init.normal_(self.intent_embedding, mean=0.0, std=0.02)
        nn.init.zeros_(self.to_vel.weight)
        nn.init.zeros_(self.to_vel.bias)
        for block in self.blocks:
            block._init_adaln_zero()


    @staticmethod
    def _expand_edge_index(edge_index: Optional[torch.Tensor], k: int, bipartite: bool) -> Optional[torch.Tensor]:
        if edge_index is None:
            return None
        e = edge_index.size(1)
        offsets = torch.arange(k, device=edge_index.device, dtype=edge_index.dtype).repeat_interleave(e)
        dst = edge_index[1].repeat(k) * k + offsets
        src = edge_index[0].repeat(k) if bipartite else edge_index[0].repeat(k) * k + offsets
        return torch.stack([src, dst], dim=0)


    @staticmethod
    def _expand_relation(relation: Optional[torch.Tensor], k: int) -> Optional[torch.Tensor]:
        if relation is None:
            return None
        return relation.unsqueeze(0).expand(k, -1, -1).reshape(-1, relation.size(-1))


    def _build_graph_context(self, data: HeteroData,
                             scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pos_m = data["agent"]["position"][:, self.num_historical_steps - 1, :self.input_dim]
        head_m = data["agent"]["heading"][:, self.num_historical_steps - 1]
        head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)
        mask_src = data["agent"]["valid_mask"][:, :self.num_historical_steps].contiguous()
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        mask_dst = data["agent"]["predict_mask"].any(dim=-1, keepdim=True)

        pos_t = data["agent"]["position"][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
        head_t = data["agent"]["heading"][:, :self.num_historical_steps].reshape(-1)
        edge_index_t2a = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))
        rel_pos_t2a = pos_t[edge_index_t2a[0]] - pos_m[edge_index_t2a[1]]
        rel_head_t2a = wrap_angle(head_t[edge_index_t2a[0]] - head_m[edge_index_t2a[1]])
        r_t2a_raw = torch.stack([torch.norm(rel_pos_t2a[:, :2], p=2, dim=-1),
                                 angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2a[1]],
                                                          nbr_vector=rel_pos_t2a[:, :2]),
                                 rel_head_t2a,
                                 (edge_index_t2a[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
        r_t2a = self.r_t2a_emb(continuous_inputs=r_t2a_raw, categorical_embs=None)

        pos_pl = data["map_polygon"]["position"][:, :self.input_dim]
        orient_pl = data["map_polygon"]["orientation"]
        edge_index_pl2a = radius(x=pos_m[:, :2], y=pos_pl[:, :2], r=self.pl2m_radius,
                                  batch_x=data["agent"]["batch"] if isinstance(data, Batch) else None,
                                  batch_y=data["map_polygon"]["batch"] if isinstance(data, Batch) else None,
                                  max_num_neighbors=300)
        edge_index_pl2a = edge_index_pl2a[:, mask_dst[edge_index_pl2a[1], 0]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_m[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_m[edge_index_pl2a[1]])
        r_pl2a_raw = torch.stack([torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                                  angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2a[1]],
                                                           nbr_vector=rel_pos_pl2a[:, :2]),
                                  rel_orient_pl2a], dim=-1)
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a_raw, categorical_embs=None)

        edge_index_a2a = radius_graph(x=pos_m[:, :2], r=self.a2m_radius,
                                      batch=data["agent"]["batch"] if isinstance(data, Batch) else None,
                                      loop=False, max_num_neighbors=300)
        edge_index_a2a = edge_index_a2a[:, mask_src[:, -1][edge_index_a2a[0]] & mask_dst[edge_index_a2a[1], 0]]
        rel_pos_a2a = pos_m[edge_index_a2a[0]] - pos_m[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_m[edge_index_a2a[0]] - head_m[edge_index_a2a[1]])
        r_a2a_raw = torch.stack([torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                                 angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2a[1]],
                                                          nbr_vector=rel_pos_a2a[:, :2]),
                                 rel_head_a2a], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a_raw, categorical_embs=None)

        x_t_hist = scene_enc["x_a"].reshape(-1, self.hidden_dim)
        x_m = scene_enc["x_a"][:, self.num_historical_steps - 1, :]
        x_pl = scene_enc["x_pl"][:, self.num_historical_steps - 1]
        agent_batch = data["agent"]["batch"]
        edge_map = None
        edge_threat = None
        k = self.num_intents

        return {
            "pos_m": pos_m, "head_m": head_m, "head_vector_m": head_vector_m,
            "x_t_hist": x_t_hist, "x_m": x_m, "x_pl": x_pl, "agent_batch": agent_batch,
            "r_t2a": r_t2a, "edge_index_t2a": edge_index_t2a,
            "r_pl2a": r_pl2a, "edge_index_pl2a": edge_index_pl2a,
            "r_a2a": r_a2a, "edge_index_a2a": edge_index_a2a,
            "edge_map": edge_map, "edge_threat": edge_threat,
            "r_t2a_exp": self._expand_relation(r_t2a, k),
            "edge_index_t2a_exp": self._expand_edge_index(edge_index_t2a, k, bipartite=True),
            "r_pl2a_exp": self._expand_relation(r_pl2a, k),
            "edge_index_pl2a_exp": self._expand_edge_index(edge_index_pl2a, k, bipartite=True),
            "r_a2a_exp": self._expand_relation(r_a2a, k),
            "edge_index_a2a_exp": self._expand_edge_index(edge_index_a2a, k, bipartite=False),
            "edge_map_exp": None, "edge_threat_exp": None, "pinn_loss": 0,
        }


    def _aggregate_scene(self, ctx: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.scene_aggregator(x_m=ctx["x_m"], x_t_hist=ctx["x_t_hist"], x_pl=ctx["x_pl"],
                                     r_t2a=ctx["r_t2a"], edge_index_t2a=ctx["edge_index_t2a"],
                                     r_pl2a=ctx["r_pl2a"], edge_index_pl2a=ctx["edge_index_pl2a"],
                                     r_a2a=ctx["r_a2a"], edge_index_a2a=ctx["edge_index_a2a"],
                                     edge_map=ctx.get("edge_map"), edge_threat=ctx.get("edge_threat"))


    def _forward_core(self, scene_conditions: Mapping[str, torch.Tensor],
                      x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        n_a, k, _ = x_t.shape
        if k != self.num_intents:
            raise ValueError(f"x_t 的 K={k}，但 decoder.num_intents={self.num_intents}")
        if t.dim() == 0:
            t = t.expand(n_a)
        elif t.dim() == 1 and t.numel() == 1:
            t = t.expand(n_a)
        if t.dim() != 1 or t.size(0) != n_a:
            raise ValueError(f"t 应为标量或 [N_a]，实际为 {tuple(t.shape)}")
        
        t_emb = self.t_emb(continuous_inputs=t.unsqueeze(-1), categorical_embs=None)
        t_emb = t_emb.unsqueeze(1).expand(-1, k, -1)
        c_hist = scene_conditions["c_hist"].unsqueeze(1).expand(-1, k, -1)
        c_map = scene_conditions["c_map"].unsqueeze(1).expand(-1, k, -1)
        c_agent = scene_conditions["c_agent"].unsqueeze(1).expand(-1, k, -1)

        x = self.x_proj_in(x_t) + self.intent_embedding.unsqueeze(0)
        for block in self.blocks:
            x = block(x, t_emb, c_hist, c_map, c_agent)

        return self.to_vel(self.final_norm(x))

    def forward(self, data: HeteroData, scene_enc: Mapping[str, torch.Tensor], x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ctx = self._build_graph_context(data, scene_enc)
        scene_conditions = self._aggregate_scene(ctx)
        v_theta = self._forward_core(scene_conditions, x_t, t)

        return v_theta

    @torch.no_grad()
    def sample(self, data: HeteroData, scene_enc: Mapping[str, torch.Tensor], 
               num_modes: int = 6,num_steps: int = 10, latent_decoder: Optional[nn.Module] = None, 
               latent_std: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx = self._build_graph_context(data, scene_enc)
        scene_conditions = self._aggregate_scene(ctx)

        N_a = ctx['pos_m'].size(0)
        device = ctx['pos_m'].device

        # time discretisation: t ∈ [0, 1)
        dt = 1.0 / num_steps
        t_grid = torch.linspace(0.0, 1.0 - dt, num_steps, device=device)
        outputs: List[torch.Tensor] = []
        t_cur = torch.empty(N_a, device=device)
        t_next = torch.empty(N_a, device=device)

        if latent_std is None:
            raise ValueError("centered-raw FM 采样必须显式传入 latent_std，""以保证训练与推理使用相同的先验尺度。")
        latent_std = latent_std.to(device=device, dtype=ctx['pos_m'].dtype)
        if latent_std.shape[-1] != self.latent_dim:
            raise ValueError(
                "latent_std 最后一维必须等于 latent_dim："f"std_shape={tuple(latent_std.shape)}, latent_dim={self.latent_dim}")

        for _ in range(num_modes):
            # initial noise in latent space: x_0 ~ N(0, I)  → [N_a, K, H]
            x_t = torch.randn(N_a, self.num_intents, self.latent_dim, device=device)
            x_t = x_t = x_t * latent_std.to(dtype=x_t.dtype)
            for t_val in t_grid:
                t_cur.fill_(t_val)
                t_next.fill_(t_val + dt)
                v1 = self._forward_core(scene_conditions, x_t, t_cur)
                x_euler = x_t + v1 * dt
                v2 = self._forward_core(scene_conditions, x_euler, t_next)
                v_heun = 0.5 * (v1 + v2)
                x_t = x_t + v_heun * dt
            outputs.append(latent_decoder(x_t) if latent_decoder is not None else x_t)
        trajectories = torch.stack(outputs, dim=1)
        
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