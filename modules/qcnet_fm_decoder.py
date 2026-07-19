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
        self.traj_encoder = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False)
        self.traj_encoder_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.scorer = nn.Sequential( nn.Linear(2 * hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.apply(weight_init)

    def forward(self, agent_context: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
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


class SceneTokenSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.apply(weight_init)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(tokens)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens

class MultiTokenStructuredSceneAggregator(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dropout: float, num_hist_tokens: int = 2, num_map_tokens: int = 4, num_agent_tokens: int = 2, num_modality_refiner_layers: int = 1, num_scene_refiner_layers: int = 1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_hist_tokens = num_hist_tokens
        self.num_map_tokens = num_map_tokens
        self.num_agent_tokens = num_agent_tokens
        self.num_modality_refiner_layers = max(0, int(num_modality_refiner_layers))
        self.num_scene_refiner_layers = max(0, int(num_scene_refiner_layers))
        self.history_query = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * num_hist_tokens))
        self.map_query = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * num_map_tokens))
        self.agent_query = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * num_agent_tokens))
        self.history_token_embedding = nn.Parameter(torch.zeros(num_hist_tokens, hidden_dim))
        self.map_token_embedding = nn.Parameter(torch.zeros(num_map_tokens, hidden_dim))
        self.agent_token_embedding = nn.Parameter(torch.zeros(num_agent_tokens, hidden_dim))
        self.condition_type_embedding = nn.Parameter(torch.zeros(3, hidden_dim))
        self.history_attn = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout, bipartite=True, has_pos_emb=True)
        self.map_attn = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout, bipartite=True, has_pos_emb=True)
        self.agent_attn = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout, bipartite=True, has_pos_emb=True)
        self.history_out_norm = nn.LayerNorm(hidden_dim)
        self.map_out_norm = nn.LayerNorm(hidden_dim)
        self.agent_out_norm = nn.LayerNorm(hidden_dim)
        # 先使用三套参数完全独立的 Transformer 对同一模态内部 tokens 做角色分化，
        # 再保留原有联合 Transformer 建模 history/map/agent 之间的跨模态关系。
        self.history_token_refiner = nn.ModuleList([SceneTokenSelfAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(self.num_modality_refiner_layers)])
        self.map_token_refiner = nn.ModuleList([SceneTokenSelfAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(self.num_modality_refiner_layers)])
        self.agent_token_refiner = nn.ModuleList([SceneTokenSelfAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(self.num_modality_refiner_layers)])
        self.scene_token_refiner = nn.ModuleList([SceneTokenSelfAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(self.num_scene_refiner_layers)])
        self.history_refined_norm = nn.LayerNorm(hidden_dim)
        self.map_refined_norm = nn.LayerNorm(hidden_dim)
        self.agent_refined_norm = nn.LayerNorm(hidden_dim)
        self.history_pool_norm = nn.LayerNorm(hidden_dim)
        self.map_pool_norm = nn.LayerNorm(hidden_dim)
        self.agent_pool_norm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)
        nn.init.normal_(self.history_token_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.map_token_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.agent_token_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.condition_type_embedding, mean=0.0, std=0.02)

    @staticmethod
    def _expand_target_edges(edge_index: Optional[torch.Tensor], relation: Optional[torch.Tensor], edge_gate: Optional[torch.Tensor], num_tokens: int):
        if edge_index is None or edge_index.numel() == 0:
            return edge_index, relation, edge_gate
        src = edge_index[0]
        dst = edge_index[1]
        device = edge_index.device
        token_id = torch.arange(num_tokens, device=device)
        src_new = src[:, None].expand(-1, num_tokens).reshape(-1)
        dst_new = (dst[:, None] * num_tokens + token_id[None, :]).reshape(-1)
        edge_index_new = torch.stack([src_new, dst_new], dim=0)
        relation_new = None
        if relation is not None:
            relation_new = relation[:, None].expand(-1, num_tokens, *relation.shape[1:]).reshape( relation.size(0) * num_tokens, *relation.shape[1:] )
        edge_gate_new = None
        if edge_gate is not None:
            edge_gate_new = edge_gate[:, None].expand(-1, num_tokens, *edge_gate.shape[1:]).reshape( edge_gate.size(0) * num_tokens, *edge_gate.shape[1:] )
        return edge_index_new, relation_new, edge_gate_new

    @staticmethod
    def _aggregate_multi(attn: AttentionLayer, source: torch.Tensor, query_tokens: torch.Tensor,
                         relation: Optional[torch.Tensor], edge_index: Optional[torch.Tensor],
                         edge_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_agents, num_tokens, hidden_dim = query_tokens.shape
        query_flat = query_tokens.reshape(num_agents * num_tokens, hidden_dim)
        if edge_index is None or edge_index.numel() == 0:
            return query_tokens
        edge_index_exp, relation_exp, edge_gate_exp = MultiTokenStructuredSceneAggregator._expand_target_edges(
            edge_index=edge_index,
            relation=relation,
            edge_gate=edge_gate,
            num_tokens=num_tokens,
        )
        out = attn((source, query_flat), relation_exp,edge_index_exp, edge_gate=edge_gate_exp)
        return out.reshape(num_agents, num_tokens, hidden_dim)

    def forward(self, x_m: torch.Tensor, x_t_hist: torch.Tensor, x_pl: torch.Tensor,
                r_t2a: Optional[torch.Tensor], edge_index_t2a: Optional[torch.Tensor],
                r_pl2a: Optional[torch.Tensor], edge_index_pl2a: Optional[torch.Tensor],
                r_a2a: Optional[torch.Tensor], edge_index_a2a: Optional[torch.Tensor],
                edge_map: Optional[torch.Tensor] = None,
                edge_threat: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if x_m.ndim != 2 or x_m.size(-1) != self.hidden_dim:
            raise ValueError(f"x_m 应为 [N_a,{self.hidden_dim}]，实际为 {tuple(x_m.shape)}")
        num_agents = x_m.size(0)
        q_hist = self.history_query(x_m).view(num_agents, self.num_hist_tokens, self.hidden_dim)
        q_map = self.map_query(x_m).view(num_agents, self.num_map_tokens, self.hidden_dim)
        q_agent = self.agent_query(x_m).view(num_agents, self.num_agent_tokens, self.hidden_dim)
        q_hist = q_hist + self.history_token_embedding[None] + self.condition_type_embedding[0][None, None]
        q_map = q_map + self.map_token_embedding[None] + self.condition_type_embedding[1][None, None]
        q_agent = q_agent + self.agent_token_embedding[None] + self.condition_type_embedding[2][None, None]
        c_hist_tokens = self._aggregate_multi(self.history_attn, x_t_hist, q_hist, r_t2a, edge_index_t2a)
        c_map_tokens = self._aggregate_multi(self.map_attn, x_pl, q_map, r_pl2a, edge_index_pl2a, edge_map)
        c_agent_tokens = self._aggregate_multi(self.agent_attn, x_m, q_agent, r_a2a, edge_index_a2a, edge_threat)
        c_hist_tokens = self.history_out_norm(c_hist_tokens)
        c_map_tokens = self.map_out_norm(c_map_tokens)
        c_agent_tokens = self.agent_out_norm(c_agent_tokens)
        # 组内 refinement：三组 tokens 使用不同 Transformer 参数，先形成各自的角色分工。
        for block in self.history_token_refiner:
            c_hist_tokens = block(c_hist_tokens)
        for block in self.map_token_refiner:
            c_map_tokens = block(c_map_tokens)
        for block in self.agent_token_refiner:
            c_agent_tokens = block(c_agent_tokens)
        # 跨模态 refinement：保留原有联合 self-attention，让三类场景证据继续交互。
        scene_tokens = torch.cat([c_hist_tokens, c_map_tokens, c_agent_tokens], dim=1)
        for block in self.scene_token_refiner:
            scene_tokens = block(scene_tokens)
        h_end = self.num_hist_tokens
        m_end = h_end + self.num_map_tokens
        c_hist_tokens = self.history_refined_norm(scene_tokens[:, :h_end])
        c_map_tokens = self.map_refined_norm(scene_tokens[:, h_end:m_end])
        c_agent_tokens = self.agent_refined_norm(scene_tokens[:, m_end:])
        c_hist = self.history_pool_norm(c_hist_tokens.mean(dim=1))
        c_map = self.map_pool_norm(c_map_tokens.mean(dim=1))
        c_agent = self.agent_pool_norm(c_agent_tokens.mean(dim=1))
        return { "c_hist": c_hist, "c_map": c_map, "c_agent": c_agent, "c_hist_tokens": c_hist_tokens, "c_map_tokens": c_map_tokens, "c_agent_tokens": c_agent_tokens}

class DecomposedMultiTokenConditionCrossAttn(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.q_norm_hist = nn.LayerNorm(hidden_dim)
        self.kv_norm_hist = nn.LayerNorm(hidden_dim)
        self.hist_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.hist_ffn = nn.Sequential( nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.q_norm_map = nn.LayerNorm(hidden_dim)
        self.kv_norm_map = nn.LayerNorm(hidden_dim)
        self.map_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.map_ffn = nn.Sequential( nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.q_norm_agent = nn.LayerNorm(hidden_dim)
        self.kv_norm_agent = nn.LayerNorm(hidden_dim)
        self.agent_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.agent_ffn = nn.Sequential( nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.apply(weight_init)

    @staticmethod
    def _ensure_3d(x: torch.Tensor):
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True
        return x, squeeze

    def _cross_attend(self, x: torch.Tensor, tokens: torch.Tensor, q_norm: nn.Module, kv_norm: nn.Module, attn: nn.MultiheadAttention, ffn: nn.Module) -> torch.Tensor:
        q = q_norm(x)
        kv = kv_norm(tokens)
        out, _ = attn(query=q, key=kv, value=kv, need_weights=False)
        x = x + out
        x = x + ffn(x)
        return x

    def forward(self, x: torch.Tensor, c_hist_tokens: torch.Tensor, c_map_tokens: torch.Tensor, c_agent_tokens: torch.Tensor) -> torch.Tensor:
        x, squeeze = self._ensure_3d(x)
        x_in = x
        x = self._cross_attend(x=x, tokens=c_hist_tokens, q_norm=self.q_norm_hist, kv_norm=self.kv_norm_hist, attn=self.hist_attn, ffn=self.hist_ffn)
        x = self._cross_attend(x=x, tokens=c_map_tokens, q_norm=self.q_norm_map, kv_norm=self.kv_norm_map, attn=self.map_attn, ffn=self.map_ffn)
        x = self._cross_attend(x=x, tokens=c_agent_tokens, q_norm=self.q_norm_agent, kv_norm=self.kv_norm_agent, attn=self.agent_attn, ffn=self.agent_ffn)
        delta =  x - x_in
        if squeeze:
            delta = delta.squeeze(1)
        return delta


class PrototypeConditionedResidualCenterHead(nn.Module):
    """Predict a deterministic, scene-conditioned residual center for each prototype.

    The head uses an independent prototype geometry/ID embedding and repeatedly lets
    the prototype query read history, map and agent tokens. It does not use time,
    source noise, AdaLN modulation or an additional outer FFN.
    """
    def __init__(self, hidden_dim: int, latent_dim: int, num_heads: int, dropout: float, num_layers: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = max(1, int(num_layers))
        self.prototype_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prototype_id_embedding: Optional[nn.Embedding] = None
        self.prototype_id_gate = nn.Parameter(torch.tensor(0.1))
        self.layers = nn.ModuleList([
            DecomposedMultiTokenConditionCrossAttn(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(self.num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.to_center = nn.Linear(hidden_dim, latent_dim)
        self.apply(weight_init)
        # Start from the original residual-FM model: center=0 at initialization.
        nn.init.zeros_(self.to_center.weight)
        nn.init.zeros_(self.to_center.bias)

    def init_prototype_id_embedding(self, num_prototypes: int) -> None:
        self.prototype_id_embedding = nn.Embedding(int(num_prototypes), self.hidden_dim)
        nn.init.normal_(self.prototype_id_embedding.weight, mean=0.0, std=0.02)

    def _add_prototype_id(self, proto_emb: torch.Tensor, prototype_index: Optional[torch.Tensor]) -> torch.Tensor:
        if prototype_index is None or self.prototype_id_embedding is None:
            return proto_emb
        n, m, k, _ = proto_emb.shape
        num_prototypes = int(self.prototype_id_embedding.num_embeddings)
        index = prototype_index.to(device=proto_emb.device, dtype=torch.long)
        if index.ndim == 1:
            index = index.unsqueeze(1)
        if index.ndim != 2 or index.size(0) != n or index.size(1) != m:
            raise ValueError(
                "center prototype_index 应为 [N,M] 或 [N]："
                f"index={tuple(index.shape)}, prototype={tuple(proto_emb.shape)}"
            )
        invalid = index.lt(0) | index.ge(num_prototypes)
        safe_index = index.clamp(min=0, max=max(num_prototypes - 1, 0))
        id_emb = self.prototype_id_embedding(safe_index).to(dtype=proto_emb.dtype)
        id_emb = id_emb.unsqueeze(2).expand(-1, -1, k, -1)
        id_emb = id_emb.masked_fill(invalid.unsqueeze(-1).unsqueeze(-1), 0.0)
        return proto_emb + self.prototype_id_gate * id_emb

    def forward(
        self,
        scene_conditions: Mapping[str, torch.Tensor],
        prototype_latent: torch.Tensor,
        prototype_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        squeeze_mode = False
        if prototype_latent.ndim == 3:
            prototype_latent = prototype_latent.unsqueeze(1)
            squeeze_mode = True
        if prototype_latent.ndim != 4:
            raise ValueError(
                "center prototype_latent 应为 [N,M,K,D] 或 [N,K,D]："
                f"实际 {tuple(prototype_latent.shape)}"
            )
        n, m, k, d = prototype_latent.shape
        if d != self.latent_dim:
            raise ValueError(f"center latent_dim 不一致：input={d}, expected={self.latent_dim}")
        proto_emb = self.prototype_proj(prototype_latent)
        proto_emb = self._add_prototype_id(proto_emb, prototype_index)
        x = proto_emb.reshape(n, m * k, self.hidden_dim)
        for layer in self.layers:
            x = x + layer(
                x=x,
                c_hist_tokens=scene_conditions["c_hist_tokens"],
                c_map_tokens=scene_conditions["c_map_tokens"],
                c_agent_tokens=scene_conditions["c_agent_tokens"],
            )
        center = self.to_center(self.final_norm(x)).reshape(n, m, k, self.latent_dim)
        return center[:, 0] if squeeze_mode else center

class LatentAdaLNBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, num_hist_tokens: int = 2, num_map_tokens: int = 4, num_agent_tokens: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.time_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 3))
        self.history_mod = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 3))
        self.map_mod = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 3))
        self.agent_mod = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 3))
        self.prototype_mod = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 3))
        self.cond_cross_attn = DecomposedMultiTokenConditionCrossAttn(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.apply(weight_init)
        self._init_adaln_zero()

    def _init_adaln_zero(self) -> None:
        for module in (self.time_mod, self.history_mod, self.map_mod, self.agent_mod, self.prototype_mod):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
        h = self.hidden_dim
        with torch.no_grad():
            # chunk(3): shift [0:h], scale [h:2h], gate [2h:3h]
            self.time_mod[-1].bias[2*h:3*h].fill_(0.05)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                c_hist: torch.Tensor, c_map: torch.Tensor, c_agent: torch.Tensor,
                c_hist_tokens: torch.Tensor, c_map_tokens: torch.Tensor,
                c_agent_tokens: torch.Tensor,
                proto_emb: torch.Tensor) -> torch.Tensor:
        modulation = (self.time_mod(t_emb) + self.history_mod(c_hist) + self.map_mod(c_map) + self.agent_mod(c_agent) + self.prototype_mod(proto_emb))
        shift, scale, gate = modulation.chunk(3, dim=-1)
        if x.ndim == 3 and shift.ndim == 2:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
            gate = gate.unsqueeze(1)
        h = self.norm(x) * (1.0 + scale) + shift
        x = x + gate * self.cond_cross_attn(h, c_hist_tokens, c_map_tokens, c_agent_tokens)
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
                 num_hist_tokens: int,
                 num_map_tokens: int,
                 num_agent_tokens: int,
                 head_dim: int,
                 dropout: float,
                 vae_num_intents: int = 3,
                 num_modality_token_refiner_layers: int = 1,
                 num_scene_token_refiner_layers: int = 1,
                 num_center_layers: int = 2) -> None:
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
        self.num_hist_tokens = num_hist_tokens
        self.num_map_tokens = num_map_tokens
        self.num_agent_tokens = num_agent_tokens
        self.head_dim = head_dim
        self.dropout = dropout
        self.num_intents = vae_num_intents
        self.num_modality_token_refiner_layers = max(0, int(num_modality_token_refiner_layers))
        self.num_scene_token_refiner_layers = max(0, int(num_scene_token_refiner_layers))
        self.num_center_layers = max(1, int(num_center_layers))
        self.x_proj_in = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.prototype_proj = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.intent_embedding = nn.Parameter(torch.zeros(self.num_intents, hidden_dim))
        self.t_emb = FourierEmbedding(input_dim=1, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t2a_emb = FourierEmbedding(input_dim=4, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2a_emb = FourierEmbedding(input_dim=3, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=3, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.scene_aggregator = MultiTokenStructuredSceneAggregator(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            num_hist_tokens=self.num_hist_tokens,
            num_map_tokens=self.num_map_tokens,
            num_agent_tokens=self.num_agent_tokens,
            num_modality_refiner_layers=self.num_modality_token_refiner_layers,
            num_scene_refiner_layers=self.num_scene_token_refiner_layers,
        )
        self.residual_center_head = PrototypeConditionedResidualCenterHead(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=self.num_center_layers,
        )
        # Prototype-conditioned CFG 的“无场景”条件。
        # 这里仅移除 scene，prototype latent / prototype ID / prototype modulation 始终保留。
        self.null_c_hist = nn.Parameter(torch.zeros(1, hidden_dim))
        self.null_c_map = nn.Parameter(torch.zeros(1, hidden_dim))
        self.null_c_agent = nn.Parameter(torch.zeros(1, hidden_dim))
        self.null_hist_tokens = nn.Parameter(torch.zeros(1, self.num_hist_tokens, hidden_dim))
        self.null_map_tokens = nn.Parameter(torch.zeros(1, self.num_map_tokens, hidden_dim))
        self.null_agent_tokens = nn.Parameter(torch.zeros(1, self.num_agent_tokens, hidden_dim))
        self.blocks = nn.ModuleList([
            LatentAdaLNBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_hist_tokens=self.num_hist_tokens,
                num_map_tokens=self.num_map_tokens,
                num_agent_tokens=self.num_agent_tokens,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.to_vel = nn.Linear(hidden_dim, latent_dim)
        self.scorer = TrajectoryScorer(hidden_dim, num_future_steps, output_dim)
        self.apply(weight_init)
        if self.num_intents > 1:
            nn.init.normal_(self.intent_embedding, mean=0.0, std=0.02)
        nn.init.zeros_(self.to_vel.weight)
        nn.init.zeros_(self.to_vel.bias)
        nn.init.zeros_(self.residual_center_head.to_center.weight)
        nn.init.zeros_(self.residual_center_head.to_center.bias)
        for block in self.blocks:
            block._init_adaln_zero()

    def init_prototype_id_embedding(self, num_prototypes: int) -> None:
        """Create a learnable discrete prototype ID embedding used together with prototype geometry."""
        self.num_prototypes = int(num_prototypes)
        self.prototype_id_embedding = nn.Embedding(self.num_prototypes, self.hidden_dim)
        self.prototype_id_gate = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.prototype_id_embedding.weight, mean=0.0, std=0.02)
        self.residual_center_head.init_prototype_id_embedding(self.num_prototypes)

    def predict_residual_center(
        self,
        scene_conditions: Mapping[str, torch.Tensor],
        prototype_latent: torch.Tensor,
        prototype_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.residual_center_head(
            scene_conditions=scene_conditions,
            prototype_latent=prototype_latent,
            prototype_index=prototype_index,
        )

    def make_null_scene_conditions( self, scene_conditions: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build the prototype-only condition used by CFG.

        All scene pooled features and scene tokens are replaced by learnable null
        embeddings. Prototype geometry, prototype ID and prototype AdaLN modulation
        are not touched because they are passed separately to ``_forward_core``.
        """
        required = ( "c_hist", "c_map", "c_agent", "c_hist_tokens", "c_map_tokens", "c_agent_tokens")
        missing = [key for key in required if key not in scene_conditions]
        if missing:
            raise KeyError(f"scene_conditions 缺少字段：{missing}")
        ref = scene_conditions["c_hist"]
        if ref.ndim != 2:
            raise ValueError(f"c_hist 应为 [N,H]，实际 {tuple(ref.shape)}")
        n = ref.size(0)
        dtype = ref.dtype
        # 保留可能存在的额外字段，只替换 scene 条件本身。
        null_conditions = dict(scene_conditions)
        null_conditions.update({
            "c_hist": self.null_c_hist.to(dtype=dtype).expand(n, -1),
            "c_map": self.null_c_map.to(dtype=dtype).expand(n, -1),
            "c_agent": self.null_c_agent.to(dtype=dtype).expand(n, -1),
            "c_hist_tokens": self.null_hist_tokens.to(dtype=dtype).expand(n, -1, -1),
            "c_map_tokens": self.null_map_tokens.to(dtype=dtype).expand(n, -1, -1),
            "c_agent_tokens": self.null_agent_tokens.to(dtype=dtype).expand(n, -1, -1),
        })
        return null_conditions

    def apply_scene_condition_dropout( self, scene_conditions: Mapping[str, torch.Tensor], drop_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Replace scene conditions by null embeddings for selected agents.

        ``drop_mask[i] = True`` means agent ``i`` is trained through the
        prototype-only branch. The prototype condition remains unchanged.
        """
        n = scene_conditions["c_hist"].size(0)
        drop_mask = drop_mask.to( device=scene_conditions["c_hist"].device, dtype=torch.bool)
        if drop_mask.ndim != 1 or drop_mask.size(0) != n:
            raise ValueError( f"drop_mask 应为 [N]，实际 {tuple(drop_mask.shape)}, N={n}" )
        if not drop_mask.any():
            return dict(scene_conditions)
        null_conditions = self.make_null_scene_conditions(scene_conditions)
        mixed = dict(scene_conditions)
        pooled_mask = drop_mask.view(n, 1)
        token_mask = drop_mask.view(n, 1, 1)
        for key in ("c_hist", "c_map", "c_agent"):
            mixed[key] = torch.where( pooled_mask, null_conditions[key], scene_conditions[key])
        for key in ("c_hist_tokens", "c_map_tokens", "c_agent_tokens"):
            mixed[key] = torch.where( token_mask, null_conditions[key], scene_conditions[key])
        return mixed

    def _prototype_cfg_velocity(
            self,
            scene_conditions: Mapping[str, torch.Tensor],
            null_scene_conditions: Mapping[str, torch.Tensor],
            x_t: torch.Tensor,
            t: torch.Tensor,
            prototype_latent: torch.Tensor,
            prototype_index: Optional[torch.Tensor],
            guidance_scale: float,
    ) -> torch.Tensor:
        """Prototype-conditioned classifier-free guidance velocity.

        v_guided = v(scene=null, prototype)
                   + w * [v(scene=full, prototype) - v(scene=null, prototype)]

        w=0 gives the prototype-only branch, w=1 gives the ordinary conditional
        branch, and w>1 amplifies the scene-specific velocity correction.
        """
        w = float(guidance_scale)
        if w < 0.0:
            raise ValueError(f"cfg_guidance_scale 必须 >= 0，实际 {w}")
        if abs(w - 1.0) < 1e-8:
            return self._forward_core( scene_conditions, x_t, t, prototype_latent=prototype_latent, prototype_index=prototype_index)
        v_proto = self._forward_core( null_scene_conditions, x_t, t, prototype_latent=prototype_latent, prototype_index=prototype_index)
        if w == 0.0:
            return v_proto
        v_cond = self._forward_core( scene_conditions, x_t, t, prototype_latent=prototype_latent, prototype_index=prototype_index)
        return v_proto + w * (v_cond - v_proto)

    def _make_prototype_embedding( self, prototype_latent: torch.Tensor, prototype_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fuse continuous prototype geometry with optional discrete prototype ID.

        The index path is guarded so -1 or out-of-range cached assignments do not
        trigger CUDA device-side asserts. Invalid indices simply receive no ID
        embedding, while the continuous prototype geometry is still used.
        """
        proto_emb = self.prototype_proj(prototype_latent)
        if ( prototype_index is not None and hasattr(self, "prototype_id_embedding") and self.prototype_id_embedding is not None ):
            num_proto = int(self.prototype_id_embedding.num_embeddings)
            prototype_index = prototype_index.to(device=proto_emb.device, dtype=torch.long)
            invalid = prototype_index.lt(0) | prototype_index.ge(num_proto)
            safe_index = prototype_index.clamp(min=0, max=max(num_proto - 1, 0))
            if safe_index.ndim == 1:
                id_emb = self.prototype_id_embedding(safe_index)
                id_emb = id_emb.unsqueeze(1).expand(-1, proto_emb.size(1), -1)
                invalid = invalid.unsqueeze(1).expand(-1, proto_emb.size(1))
            elif safe_index.ndim == 2:
                id_emb = self.prototype_id_embedding(safe_index)
                if id_emb.size(1) == 1 and proto_emb.size(1) > 1:
                    id_emb = id_emb.expand(-1, proto_emb.size(1), -1)
                    invalid = invalid.expand(-1, proto_emb.size(1))
                elif id_emb.size(1) != proto_emb.size(1):
                    raise ValueError( "prototype_index 的 K 维与 prototype_latent 不一致：" f"index={tuple(prototype_index.shape)}, prototype={tuple(prototype_latent.shape)}" )
            else:
                raise ValueError( f"prototype_index 应为 [N] 或 [N,K]，实际 {tuple(prototype_index.shape)}" )
            id_emb = id_emb.to(dtype=proto_emb.dtype)
            id_emb = id_emb.masked_fill(invalid.unsqueeze(-1), 0.0)
            proto_emb = proto_emb + self.prototype_id_gate * id_emb
        return proto_emb

    def _build_graph_context(self, data: HeteroData, scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        edge_index_a2a = radius_graph(x=pos_m[:, :2], r=self.a2m_radius, batch=data["agent"]["batch"] if isinstance(data, Batch) else None, loop=False, max_num_neighbors=300)
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
        return {
            "pos_m": pos_m, "head_m": head_m, "head_vector_m": head_vector_m,
            "x_t_hist": x_t_hist, "x_m": x_m, "x_pl": x_pl, "agent_batch": agent_batch,
            "r_t2a": r_t2a, "edge_index_t2a": edge_index_t2a,
            "r_pl2a": r_pl2a, "edge_index_pl2a": edge_index_pl2a,
            "r_a2a": r_a2a, "edge_index_a2a": edge_index_a2a,
            "edge_map": edge_map, "edge_threat": edge_threat
        }

    def _aggregate_scene(self, ctx: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.scene_aggregator(x_m=ctx["x_m"], x_t_hist=ctx["x_t_hist"], x_pl=ctx["x_pl"],
                                     r_t2a=ctx["r_t2a"], edge_index_t2a=ctx["edge_index_t2a"],
                                     r_pl2a=ctx["r_pl2a"], edge_index_pl2a=ctx["edge_index_pl2a"],
                                     r_a2a=ctx["r_a2a"], edge_index_a2a=ctx["edge_index_a2a"],
                                     edge_map=ctx.get("edge_map"), edge_threat=ctx.get("edge_threat"))

    def _forward_core(self, scene_conditions: Mapping[str, torch.Tensor],
                      x_t: torch.Tensor, t: torch.Tensor,
                      prototype_latent: Optional[torch.Tensor] = None,
                      prototype_index: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        c_hist_tokens = scene_conditions["c_hist_tokens"]
        c_map_tokens = scene_conditions["c_map_tokens"]
        c_agent_tokens = scene_conditions["c_agent_tokens"]
        x = self.x_proj_in(x_t) + self.intent_embedding.unsqueeze(0)
        proto_emb = torch.zeros( n_a, k, self.hidden_dim, device=x_t.device, dtype=x_t.dtype)
        if prototype_latent is not None:
            if prototype_latent.ndim == 2:
                prototype_latent = prototype_latent.unsqueeze(1)
            if prototype_latent.shape != x_t.shape:
                raise ValueError( "prototype_latent 的形状必须与 x_t 一致：" f"prototype_shape={tuple(prototype_latent.shape)}, x_t_shape={tuple(x_t.shape)}" )
            prototype_latent = prototype_latent.to(device=x_t.device, dtype=x_t.dtype)
            proto_emb = self._make_prototype_embedding( prototype_latent=prototype_latent, prototype_index=prototype_index)
            x = x + proto_emb
        for block in self.blocks:
            x = block(
                x=x,
                t_emb=t_emb,
                c_hist=c_hist,
                c_map=c_map,
                c_agent=c_agent,
                c_hist_tokens=c_hist_tokens,
                c_map_tokens=c_map_tokens,
                c_agent_tokens=c_agent_tokens,
                proto_emb=proto_emb,
            )
        return self.to_vel(self.final_norm(x))

    def build_scene_conditions(self, data: HeteroData, scene_enc: Mapping[str, torch.Tensor] ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Build graph context and multi-token scene conditions once.

        The returned scene_conditions are shared by the FM decoder and the
        prototype selector, so the expensive scene aggregation is not repeated.
        """
        ctx = self._build_graph_context(data, scene_enc)
        scene_conditions = self._aggregate_scene(ctx)
        return ctx, scene_conditions

    def forward(self, data: HeteroData, scene_enc: Mapping[str, torch.Tensor], x_t: torch.Tensor,
                t: torch.Tensor, prototype_latent: Optional[torch.Tensor] = None,
                prototype_index: Optional[torch.Tensor] = None,
                scene_conditions: Optional[Mapping[str, torch.Tensor]] = None) -> torch.Tensor:
        if scene_conditions is None:
            _, scene_conditions = self.build_scene_conditions(data, scene_enc)
        v_theta = self._forward_core( scene_conditions, x_t, t, prototype_latent=prototype_latent, prototype_index=prototype_index)
        return v_theta

    @torch.no_grad()
    def sample(self, data: HeteroData, scene_enc: Mapping[str, torch.Tensor],
               num_modes: int = 6, num_steps: int = 10,
               latent_decoder: Optional[nn.Module] = None,
               latent_std: Optional[torch.Tensor] = None,
               prototype_latents: Optional[torch.Tensor] = None,
               prototype_local_std: Optional[torch.Tensor] = None,
               prototype_pi: Optional[torch.Tensor] = None,
               prototype_indices: Optional[torch.Tensor] = None,
               scene_conditions: Optional[Mapping[str, torch.Tensor]] = None,
               cfg_guidance_scale: float = 1.0,
               residual_source_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        if scene_conditions is None:
            _, scene_conditions = self.build_scene_conditions(data, scene_enc)
        N_a = scene_conditions["c_hist"].size(0)
        device = scene_conditions["c_hist"].device
        dtype = scene_conditions["c_hist"].dtype
        # time discretisation: t ∈ [0, 1)
        dt = 1.0 / num_steps
        t_grid = torch.linspace(0.0, 1.0 - dt, num_steps, device=device)
        outputs: List[torch.Tensor] = []
        t_cur = torch.empty(N_a, device=device)
        t_next = torch.empty(N_a, device=device)
        residual_mode = prototype_latents is not None
        cfg_guidance_scale = float(cfg_guidance_scale)
        residual_source_scale = float(residual_source_scale)
        if residual_source_scale <= 0.0:
            raise ValueError(f"residual_source_scale 必须 > 0，实际 {residual_source_scale}")
        if cfg_guidance_scale < 0.0:
            raise ValueError( f"cfg_guidance_scale 必须 >= 0，实际 {cfg_guidance_scale}" )
        if not residual_mode and abs(cfg_guidance_scale - 1.0) > 1e-8:
            raise ValueError( "当前 CFG 只定义为 prototype-conditioned CFG；" "非 residual/prototype 模式下 cfg_guidance_scale 必须为 1.0。" )
        use_cfg = residual_mode and abs(cfg_guidance_scale - 1.0) > 1e-8
        null_scene_conditions = ( self.make_null_scene_conditions(scene_conditions) if use_cfg else None )
        if residual_mode:
            prototype_latents = prototype_latents.to(device=device, dtype=dtype)
            if prototype_latents.ndim == 3:
                prototype_latents = prototype_latents.unsqueeze(2)
            if prototype_latents.ndim != 4:
                raise ValueError( "prototype_latents 应为 [N_a,M,K,D] 或 [N_a,M,D]：" f"实际 {tuple(prototype_latents.shape)}" )
            if prototype_latents.size(0) != N_a or prototype_latents.size(2) != self.num_intents or prototype_latents.size(3) != self.latent_dim:
                raise ValueError( "prototype_latents 形状与 decoder 不匹配：" f"prototype={tuple(prototype_latents.shape)}, N_a={N_a}, K={self.num_intents}, D={self.latent_dim}" )
            if prototype_local_std is None:
                prototype_local_std = torch.ones_like(prototype_latents)
            else:
                prototype_local_std = prototype_local_std.to(device=device, dtype=dtype)
                if prototype_local_std.ndim == 3:
                    prototype_local_std = prototype_local_std.unsqueeze(2)
                prototype_local_std = prototype_local_std.clamp_min(1e-4)
            if prototype_indices is not None:
                prototype_indices = prototype_indices.to(device=device, dtype=torch.long)
                if prototype_indices.ndim == 1:
                    prototype_indices = prototype_indices.unsqueeze(1)
                if prototype_indices.ndim != 2:
                    raise ValueError( f"prototype_indices 应为 [N_a,M] 或 [N_a]，实际 {tuple(prototype_indices.shape)}" )
                if prototype_indices.size(0) != N_a or prototype_indices.size(1) < num_modes:
                    raise ValueError( "prototype_indices 与采样 mode 数不匹配：" f"indices={tuple(prototype_indices.shape)}, N_a={N_a}, num_modes={num_modes}" )
            residual_centers = self.predict_residual_center(
                scene_conditions=scene_conditions,
                prototype_latent=prototype_latents,
                prototype_index=prototype_indices,
            )
        else:
            if latent_std is None:
                raise ValueError("centered-raw FM 采样必须显式传入 latent_std，以保证训练与推理使用相同的先验尺度。")
            latent_std = latent_std.to(device=device, dtype=dtype)
            if latent_std.shape[-1] != self.latent_dim:
                raise ValueError( "latent_std 最后一维必须等于 latent_dim：" f"std_shape={tuple(latent_std.shape)}, latent_dim={self.latent_dim}" )
        for mode_idx in range(num_modes):
            if residual_mode:
                p = prototype_latents[:, mode_idx]
                idx_i = prototype_indices[:, mode_idx] if prototype_indices is not None else None
                std = prototype_local_std[:, mode_idx]
                center_i = residual_centers[:, mode_idx]
                x_t = torch.randn_like(p) * std.to(dtype=dtype) * residual_source_scale
            else:
                p = None
                idx_i = None
                x_t = torch.randn(N_a, self.num_intents, self.latent_dim, device=device, dtype=dtype)
                x_t = x_t * latent_std.to(dtype=dtype)
            for t_val in t_grid:
                t_cur.fill_(t_val)
                t_next.fill_(t_val + dt)
                if use_cfg:
                    v1 = self._prototype_cfg_velocity(
                        scene_conditions=scene_conditions,
                        null_scene_conditions=null_scene_conditions,
                        x_t=x_t,
                        t=t_cur,
                        prototype_latent=p,
                        prototype_index=idx_i,
                        guidance_scale=cfg_guidance_scale,
                    )
                else:
                    v1 = self._forward_core( scene_conditions, x_t, t_cur, prototype_latent=p, prototype_index=idx_i)
                x_euler = x_t + v1 * dt
                if use_cfg:
                    v2 = self._prototype_cfg_velocity(
                        scene_conditions=scene_conditions,
                        null_scene_conditions=null_scene_conditions,
                        x_t=x_euler,
                        t=t_next,
                        prototype_latent=p,
                        prototype_index=idx_i,
                        guidance_scale=cfg_guidance_scale,
                    )
                else:
                    v2 = self._forward_core( scene_conditions, x_euler, t_next, prototype_latent=p, prototype_index=idx_i)
                v_heun = 0.5 * (v1 + v2)
                x_t = x_t + v_heun * dt
            z_final = p + center_i + x_t if residual_mode else x_t
            decoded = latent_decoder(z_final) if latent_decoder is not None else z_final
            # Most current VAE settings use one latent token per trajectory.
            # Squeeze this token dimension so downstream scorer/metrics receive [N,M,T,D].
            if decoded.ndim == 4 and decoded.size(1) == 1:
                decoded = decoded.squeeze(1)
            outputs.append(decoded)
        trajectories = torch.stack(outputs, dim=1)
        if prototype_pi is not None:
            pi = prototype_pi[:, :num_modes].to(device=device, dtype=dtype)
        else:
            pi = torch.ones(N_a, num_modes, device=device, dtype=dtype) / max(num_modes, 1)
        return trajectories, pi

    def compute_scorer_loss(self, trajectories: torch.Tensor, target: torch.Tensor, predict_mask: torch.Tensor, agent_context: torch.Tensor) -> torch.Tensor:
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