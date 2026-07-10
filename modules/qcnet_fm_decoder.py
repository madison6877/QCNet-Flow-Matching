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


class MultiTokenStructuredSceneAggregator(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dropout: float,
                 num_hist_tokens: int = 2, num_map_tokens: int = 4, num_agent_tokens: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_hist_tokens = num_hist_tokens
        self.num_map_tokens = num_map_tokens
        self.num_agent_tokens = num_agent_tokens

        self.history_query = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * num_hist_tokens))
        self.map_query = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * num_map_tokens))
        self.agent_query = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * num_agent_tokens))

        self.history_token_embedding = nn.Parameter(torch.zeros(num_hist_tokens, hidden_dim))
        self.map_token_embedding = nn.Parameter(torch.zeros(num_map_tokens, hidden_dim))
        self.agent_token_embedding = nn.Parameter(torch.zeros(num_agent_tokens, hidden_dim))
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

        self.history_pool_norm = nn.LayerNorm(hidden_dim)
        self.map_pool_norm = nn.LayerNorm(hidden_dim)
        self.agent_pool_norm = nn.LayerNorm(hidden_dim)
        self.history_detail_norm = nn.LayerNorm(hidden_dim)
        self.map_detail_norm = nn.LayerNorm(hidden_dim)
        self.agent_detail_norm = nn.LayerNorm(hidden_dim)

        self.apply(weight_init)
        nn.init.normal_(self.history_token_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.map_token_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.agent_token_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.condition_type_embedding, mean=0.0, std=0.02)


    @staticmethod
    def _expand_target_edges(edge_index: Optional[torch.Tensor],
                             relation: Optional[torch.Tensor],
                             edge_gate: Optional[torch.Tensor],
                             num_tokens: int):
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
            relation_new = relation[:, None].expand(-1, num_tokens, *relation.shape[1:]).reshape(
                relation.size(0) * num_tokens, *relation.shape[1:]
            )

        edge_gate_new = None
        if edge_gate is not None:
            edge_gate_new = edge_gate[:, None].expand(-1, num_tokens, *edge_gate.shape[1:]).reshape(
                edge_gate.size(0) * num_tokens, *edge_gate.shape[1:]
            )

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

        c_hist = self.history_pool_norm(c_hist_tokens.mean(dim=1))
        c_map = self.map_pool_norm(c_map_tokens.mean(dim=1))
        c_agent = self.agent_pool_norm(c_agent_tokens.mean(dim=1))

        return {
            "c_hist": c_hist,
            "c_map": c_map,
            "c_agent": c_agent,
            "c_hist_tokens": c_hist_tokens,
            "c_map_tokens": c_map_tokens,
            "c_agent_tokens": c_agent_tokens,
        }

class LatentAdaLNBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float,
                 num_hist_tokens: int = 2,
                 num_map_tokens: int = 4,
                 num_agent_tokens: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # -------- AdaLN query/input norms --------
        self.norm_hist_attn = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm_hist_ffn = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.norm_map_attn = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm_map_ffn = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.norm_agent_attn = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm_agent_ffn = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # -------- key/value norms for condition tokens --------
        self.kv_norm_hist = nn.LayerNorm(hidden_dim)
        self.kv_norm_map = nn.LayerNorm(hidden_dim)
        self.kv_norm_agent = nn.LayerNorm(hidden_dim)

        # -------- decomposed condition attentions --------
        self.hist_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.map_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.agent_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # -------- separate FFNs --------
        self.hist_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.map_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.agent_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        mod_dim = hidden_dim * 18

        self.time_mod = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, mod_dim))
        self.history_mod = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, mod_dim))
        self.map_mod = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, mod_dim))
        self.agent_mod = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, mod_dim))
        self.prototype_mod = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, mod_dim))

        self.apply(weight_init)
        self._init_adaln_zero()

    def _init_adaln_zero(self) -> None:
        for module in (self.time_mod, self.history_mod, self.map_mod, self.agent_mod, self.prototype_mod):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)

        h = self.hidden_dim
        with torch.no_grad():
            gate_chunks = [2, 5, 8, 11, 14, 17]
            for idx in gate_chunks:
                module = self.time_mod[-1]
                module.bias[idx * h:(idx + 1) * h].fill_(0.05)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale) + shift

    @staticmethod
    def _ensure_3d(x: torch.Tensor):
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True
        return x, squeeze

    def _attend(self,
                x: torch.Tensor,
                tokens: torch.Tensor,
                kv_norm: nn.Module,
                attn: nn.MultiheadAttention) -> torch.Tensor:
        x_3d, squeeze = self._ensure_3d(x)

        kv = kv_norm(tokens)

        out, _ = attn(query=x_3d, key=kv, value=kv, need_weights=False)

        if squeeze:
            out = out.squeeze(1)
        return out

    def _apply_attn_sublayer(self,
                             x: torch.Tensor,
                             tokens: torch.Tensor,
                             norm: nn.Module,
                             kv_norm: nn.Module,
                             attn: nn.MultiheadAttention,
                             shift: torch.Tensor,
                             scale: torch.Tensor,
                             gate: torch.Tensor) -> torch.Tensor:
        h = self._modulate(norm(x), shift, scale)
        delta = self._attend(x=h, tokens=tokens, kv_norm=kv_norm, attn=attn)
        return x + gate * delta

    def _apply_ffn_sublayer(self,
                            x: torch.Tensor,
                            norm: nn.Module,
                            ffn: nn.Module,
                            shift: torch.Tensor,
                            scale: torch.Tensor,
                            gate: torch.Tensor) -> torch.Tensor:
        h = self._modulate(norm(x), shift, scale)
        return x + gate * ffn(h)

    def forward(self,
                x: torch.Tensor,
                t_emb: torch.Tensor,
                c_hist: torch.Tensor,
                c_map: torch.Tensor,
                c_agent: torch.Tensor,
                c_hist_tokens: torch.Tensor,
                c_map_tokens: torch.Tensor,
                c_agent_tokens: torch.Tensor,
                proto_emb: torch.Tensor) -> torch.Tensor:
        modulation = (self.time_mod(t_emb) + self.history_mod(c_hist) + self.map_mod(c_map) + self.agent_mod(c_agent) + self.prototype_mod(proto_emb))
        (shift_hist_attn, scale_hist_attn, gate_hist_attn, shift_hist_ffn, scale_hist_ffn, gate_hist_ffn,
         shift_map_attn, scale_map_attn, gate_map_attn, shift_map_ffn, scale_map_ffn, gate_map_ffn,
         shift_agent_attn, scale_agent_attn, gate_agent_attn, shift_agent_ffn, scale_agent_ffn, gate_agent_ffn) = modulation.chunk(18, dim=-1)

        if x.ndim == 3 and shift_hist_attn.ndim == 2:
            chunks = [shift_hist_attn, scale_hist_attn, gate_hist_attn,
                      shift_hist_ffn, scale_hist_ffn, gate_hist_ffn,
                      shift_map_attn, scale_map_attn, gate_map_attn,
                      shift_map_ffn, scale_map_ffn, gate_map_ffn,
                      shift_agent_attn, scale_agent_attn, gate_agent_attn,
                      shift_agent_ffn, scale_agent_ffn, gate_agent_ffn]
            chunks = [c.unsqueeze(1) for c in chunks]

            (shift_hist_attn, scale_hist_attn, gate_hist_attn, shift_hist_ffn, scale_hist_ffn, gate_hist_ffn,
             shift_map_attn, scale_map_attn, gate_map_attn, shift_map_ffn, scale_map_ffn, gate_map_ffn,
             shift_agent_attn, scale_agent_attn, gate_agent_attn, shift_agent_ffn, scale_agent_ffn, gate_agent_ffn)= chunks

        x = self._apply_attn_sublayer(x=x, tokens=c_hist_tokens, norm=self.norm_hist_attn, kv_norm=self.kv_norm_hist,
                                      attn=self.hist_attn, shift=shift_hist_attn, scale=scale_hist_attn, gate=gate_hist_attn)

        x = self._apply_ffn_sublayer(x=x, norm=self.norm_hist_ffn, ffn=self.hist_ffn,
                                     shift=shift_hist_ffn, scale=scale_hist_ffn,gate=gate_hist_ffn)

        x = self._apply_attn_sublayer(x=x, tokens=c_map_tokens, norm=self.norm_map_attn, kv_norm=self.kv_norm_map,
                                      attn=self.map_attn, shift=shift_map_attn, scale=scale_map_attn, gate=gate_map_attn)

        x = self._apply_ffn_sublayer(x=x,norm=self.norm_map_ffn, ffn=self.map_ffn, 
                                     shift=shift_map_ffn, scale=scale_map_ffn, gate=gate_map_ffn)

        x = self._apply_attn_sublayer(x=x, tokens=c_agent_tokens, norm=self.norm_agent_attn, kv_norm=self.kv_norm_agent,
                                      attn=self.agent_attn, shift=shift_agent_attn, scale=scale_agent_attn, gate=gate_agent_attn)

        x = self._apply_ffn_sublayer(x=x, norm=self.norm_agent_ffn, ffn=self.agent_ffn, 
                                     shift=shift_agent_ffn, scale=scale_agent_ffn, gate=gate_agent_ffn)
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
        self.num_hist_tokens = num_hist_tokens
        self.num_map_tokens = num_map_tokens
        self.num_agent_tokens = num_agent_tokens
        self.head_dim = head_dim
        self.dropout = dropout
        self.num_intents = vae_num_intents

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
        )

        # residual_mean + residual_transport_scale
        # residual_transport_scale is not a prototype residual std. It is a
        # scene-conditioned transport budget for the stochastic residual flow.
        self.residual_transport_scale_min = 0.05
        self.residual_transport_scale_max = 1.00
        self.residual_mean_scale_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim + 1),
        )
        
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
        self._init_residual_mean_scale_head(init_scale=0.5)
        nn.init.zeros_(self.to_vel.weight)
        nn.init.zeros_(self.to_vel.bias)
        for block in self.blocks:
            block._init_adaln_zero()


    def init_prototype_id_embedding(self, num_prototypes: int) -> None:
        """Create learnable prototype-ID embeddings after the prototype bank is loaded.

        The geometric prototype latent is still used as the actual anchor in latent
        space. The ID embedding is only an additional discrete conditioning signal.
        """
        num_prototypes = int(num_prototypes)
        if num_prototypes <= 0:
            raise ValueError(f"num_prototypes 必须为正数，实际为 {num_prototypes}")

        self.prototype_id_embedding = nn.Embedding(num_prototypes, self.hidden_dim)
        self.prototype_id_gate = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.prototype_id_embedding.weight, mean=0.0, std=0.02)

    @staticmethod
    def _inverse_sigmoid(x: float) -> float:
        x = min(max(float(x), 1e-6), 1.0 - 1e-6)
        return math.log(x / (1.0 - x))

    def _init_residual_mean_scale_head(self, init_scale: float = 0.5) -> None:
        """Initialize mean to zero and scale near init_scale."""
        last = self.residual_mean_scale_head[-1]
        scale_min = float(self.residual_transport_scale_min)
        scale_max = float(self.residual_transport_scale_max)
        init_scale = min(max(float(init_scale), scale_min + 1e-6), scale_max - 1e-6)
        normalized = (init_scale - scale_min) / max(scale_max - scale_min, 1e-6)
        scale_bias = self._inverse_sigmoid(normalized)
        with torch.no_grad():
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            last.bias[:self.latent_dim].zero_()
            last.bias[self.latent_dim].fill_(scale_bias)

    def set_residual_transport_scale_range(
            self,
            scale_min: float = 0.05,
            scale_max: float = 3.00,
            init_scale: float = 0.5,
    ) -> None:
        """Set scale range and reinitialize the residual scale bias.

        residual_delta_std_scale from QCNetFM is used only as the initial value
        of the learnable transport scale, not as a fixed sampling multiplier.
        """
        if scale_min <= 0.0:
            raise ValueError(f"scale_min 必须 > 0，实际 {scale_min}")
        if scale_max <= scale_min:
            raise ValueError(
                f"scale_max 必须大于 scale_min，实际 scale_min={scale_min}, scale_max={scale_max}"
            )
        self.residual_transport_scale_min = float(scale_min)
        self.residual_transport_scale_max = float(scale_max)
        self._init_residual_mean_scale_head(init_scale=init_scale)

    def _make_prototype_embedding(
            self,
            prototype_latent: torch.Tensor,
            prototype_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build prototype condition embedding from geometry + optional discrete ID.

        Invalid prototype indices (e.g. -1 from cached invalid agents) are never
        sent directly into nn.Embedding. They are clamped for lookup and then
        masked to zero contribution, preventing CUDA device-side asserts.
        """
        proto_emb = self.prototype_proj(prototype_latent)

        if (
            prototype_index is not None
            and hasattr(self, "prototype_id_embedding")
            and self.prototype_id_embedding is not None
        ):
            num_proto = int(self.prototype_id_embedding.num_embeddings)
            prototype_index = prototype_index.to(device=proto_emb.device, dtype=torch.long)

            invalid = prototype_index.lt(0) | prototype_index.ge(num_proto)
            safe_index = prototype_index.clamp(min=0, max=num_proto - 1)

            if safe_index.ndim == 1:
                id_emb = self.prototype_id_embedding(safe_index)  # [N,H]
                id_emb = id_emb.unsqueeze(1).expand(-1, proto_emb.size(1), -1)
                invalid = invalid.unsqueeze(1).expand(-1, proto_emb.size(1))

            elif safe_index.ndim == 2:
                id_emb = self.prototype_id_embedding(safe_index)  # [N,K,H]
                if id_emb.size(1) == 1 and proto_emb.size(1) > 1:
                    id_emb = id_emb.expand(-1, proto_emb.size(1), -1)
                    invalid = invalid.expand(-1, proto_emb.size(1))
                elif id_emb.size(1) != proto_emb.size(1):
                    raise ValueError(
                        "prototype_index 的 K 维与 prototype_latent 不一致："
                        f"index={tuple(prototype_index.shape)}, prototype={tuple(prototype_latent.shape)}"
                    )
            else:
                raise ValueError(
                    f"prototype_index 应为 [N] 或 [N,K]，实际 {tuple(prototype_index.shape)}"
                )

            id_emb = id_emb.to(dtype=proto_emb.dtype)
            if invalid.any():
                id_emb = id_emb.masked_fill(invalid.unsqueeze(-1), 0.0)

            proto_emb = proto_emb + self.prototype_id_gate * id_emb

        return proto_emb

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
    

    def predict_residual_mean_and_scale(
            self,
            scene_conditions: Mapping[str, torch.Tensor],
            prototype_latent: torch.Tensor,
            prototype_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict deterministic residual mean and stochastic transport scale.

        residual_transport_scale is a scene-conditioned multiplier on
        prototype_local_std. It controls the source Gaussian radius used by the
        residual flow; it is not interpreted as the prototype residual std.
        """
        if prototype_latent.ndim == 2:
            prototype_latent = prototype_latent.unsqueeze(1)

        if prototype_latent.ndim != 3:
            raise ValueError(
                f"prototype_latent 应为 [N,K,D] 或 [N,D]，实际 {tuple(prototype_latent.shape)}"
            )

        n, k, d = prototype_latent.shape
        if k != self.num_intents or d != self.latent_dim:
            raise ValueError(
                "prototype_latent 形状与 decoder 不一致："
                f"prototype={tuple(prototype_latent.shape)}, K={self.num_intents}, D={self.latent_dim}"
            )

        c_hist = scene_conditions["c_hist"]
        c_map = scene_conditions["c_map"]
        c_agent = scene_conditions["c_agent"]

        prototype_latent = prototype_latent.to(device=c_hist.device, dtype=c_hist.dtype)
        proto_emb = self._make_prototype_embedding(
            prototype_latent=prototype_latent,
            prototype_index=prototype_index,
        )

        c_hist = c_hist.unsqueeze(1).expand(-1, k, -1)
        c_map = c_map.unsqueeze(1).expand(-1, k, -1)
        c_agent = c_agent.unsqueeze(1).expand(-1, k, -1)

        head_input = torch.cat([c_hist, c_map, c_agent, proto_emb], dim=-1)
        out = self.residual_mean_scale_head(head_input)

        residual_mean = out[..., :self.latent_dim]
        scale_raw = out[..., self.latent_dim:self.latent_dim + 1]

        scale_min = float(self.residual_transport_scale_min)
        scale_max = float(self.residual_transport_scale_max)
        residual_transport_scale = scale_min + (scale_max - scale_min) * torch.sigmoid(scale_raw)

        return residual_mean, residual_transport_scale

    def predict_residual_mean(
            self,
            scene_conditions: Mapping[str, torch.Tensor],
            prototype_latent: torch.Tensor,
            prototype_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Backward-compatible wrapper returning only residual_mean."""
        residual_mean, _ = self.predict_residual_mean_and_scale(
            scene_conditions=scene_conditions,
            prototype_latent=prototype_latent,
            prototype_index=prototype_index,
        )
        return residual_mean


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
        if prototype_latent is not None:
            if prototype_latent.ndim == 2:
                prototype_latent = prototype_latent.unsqueeze(1)
            if prototype_latent.shape != x_t.shape:
                raise ValueError(
                    "prototype_latent 的形状必须与 x_t 一致："
                    f"prototype_shape={tuple(prototype_latent.shape)}, x_t_shape={tuple(x_t.shape)}"
                )
            prototype_latent = prototype_latent.to(device=x_t.device, dtype=x_t.dtype)
            proto_emb = self._make_prototype_embedding(prototype_latent=prototype_latent, prototype_index=prototype_index)
            x = x + proto_emb
        else:
            proto_emb = torch.zeros(n_a, k, self.hidden_dim, device=x_t.device, dtype=x_t.dtype)

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
                proto_emb=proto_emb
            )
        return self.to_vel(self.final_norm(x))

    def build_scene_conditions(self,
                               data: HeteroData,
                               scene_enc: Mapping[str, torch.Tensor]
                              ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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
        v_theta = self._forward_core(scene_conditions, x_t, t, prototype_latent=prototype_latent, prototype_index=prototype_index)

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
               scene_conditions: Optional[Mapping[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        if residual_mode:
            prototype_latents = prototype_latents.to(device=device, dtype=dtype)
            if prototype_latents.ndim == 3:
                prototype_latents = prototype_latents.unsqueeze(2)
            if prototype_latents.ndim != 4:
                raise ValueError(
                    "prototype_latents 应为 [N_a,M,K,D] 或 [N_a,M,D]："
                    f"实际 {tuple(prototype_latents.shape)}"
                )
            if prototype_latents.size(0) != N_a or prototype_latents.size(2) != self.num_intents or prototype_latents.size(3) != self.latent_dim:
                raise ValueError(
                    "prototype_latents 形状与 decoder 不匹配："
                    f"prototype={tuple(prototype_latents.shape)}, N_a={N_a}, K={self.num_intents}, D={self.latent_dim}"
                )
            if prototype_local_std is None:
                prototype_local_std = torch.ones_like(prototype_latents)
            else:
                prototype_local_std = prototype_local_std.to(device=device, dtype=dtype)
                if prototype_local_std.ndim == 3:
                    prototype_local_std = prototype_local_std.unsqueeze(2)
                prototype_local_std = prototype_local_std.clamp_min(1e-4)
            if prototype_latents.size(1) < num_modes:
                raise ValueError(f"prototype_latents 的 mode 数小于 num_modes: "f"prototype_latents={tuple(prototype_latents.shape)}, num_modes={num_modes}")

            if prototype_latents.size(1) > num_modes:
                prototype_latents = prototype_latents[:, :num_modes]

            if prototype_local_std.shape != prototype_latents.shape:
                raise ValueError("prototype_local_std 形状必须与 prototype_latents 一致："f"std={tuple(prototype_local_std.shape)}, "f"prototype={tuple(prototype_latents.shape)}")

            if prototype_indices is not None:
                prototype_indices = prototype_indices.to(device=device, dtype=torch.long)
                if prototype_indices.ndim == 1:
                    prototype_indices = prototype_indices.unsqueeze(1)
                if prototype_indices.ndim != 2:
                    raise ValueError(f"prototype_indices 应为 [N,M] 或 [N]，实际 {tuple(prototype_indices.shape)}")
                if prototype_indices.size(0) != N_a:
                    raise ValueError(
                        f"prototype_indices batch 维度不一致：indices={tuple(prototype_indices.shape)}, N_a={N_a}"
                    )
                if prototype_indices.size(1) < num_modes:
                    raise ValueError(
                        f"prototype_indices 的 mode 数小于 num_modes：indices={tuple(prototype_indices.shape)}, num_modes={num_modes}"
                    )
                if prototype_indices.size(1) > num_modes:
                    prototype_indices = prototype_indices[:, :num_modes]
        else:
            if latent_std is None:
                raise ValueError("centered-raw FM 采样必须显式传入 latent_std，以保证训练与推理使用相同的先验尺度。")
            latent_std = latent_std.to(device=device, dtype=dtype)
            if latent_std.shape[-1] != self.latent_dim:
                raise ValueError(
                    "latent_std 最后一维必须等于 latent_dim："
                    f"std_shape={tuple(latent_std.shape)}, latent_dim={self.latent_dim}"
                )
        
        residual_mean = None
        residual_transport_scale = None
        if residual_mode:
            # prototype_latents: [N, M, K, D]
            n, m, k, d = prototype_latents.shape
            residual_mean_list = []
            residual_scale_list = []
            for mode_idx in range(num_modes):
                p_i = prototype_latents[:, mode_idx]
                idx_i = prototype_indices[:, mode_idx] if prototype_indices is not None else None
                r_mean_i, r_scale_i = self.predict_residual_mean_and_scale(
                    scene_conditions=scene_conditions,
                    prototype_latent=p_i,
                    prototype_index=idx_i,
                )
                residual_mean_list.append(r_mean_i)
                residual_scale_list.append(r_scale_i)
            residual_mean = torch.stack(residual_mean_list, dim=1)
            residual_transport_scale = torch.stack(residual_scale_list, dim=1)

        for mode_idx in range(num_modes):
            if residual_mode:
                p = prototype_latents[:, mode_idx]
                r_mean = residual_mean[:, mode_idx]
                r_scale = residual_transport_scale[:, mode_idx]
                std = prototype_local_std[:, mode_idx]
                idx_i = prototype_indices[:, mode_idx] if prototype_indices is not None else None
                x_t = torch.randn_like(p) * std.to(dtype=dtype) * r_scale.to(dtype=dtype)
            else:
                p = None
                idx_i = None
                x_t = torch.randn(N_a, self.num_intents, self.latent_dim, device=device, dtype=dtype)
                x_t = x_t * latent_std.to(dtype=dtype)

            for t_val in t_grid:
                t_cur.fill_(t_val)
                t_next.fill_(t_val + dt)
                v1 = self._forward_core(scene_conditions, x_t, t_cur, prototype_latent=p, prototype_index=idx_i)
                # x_euler = x_t + v1 * dt
                # v2 = self._forward_core(scene_conditions, x_euler, t_next, prototype_latent=p)
                # v_heun = 0.5 * (v1 + v2)
                # x_t = x_t + v_heun * dt
                x_t = x_t + v1 * dt

            z_final = p + x_t + r_mean if residual_mode else x_t
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