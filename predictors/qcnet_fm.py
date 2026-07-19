import gc
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional, Dict, Mapping

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from utils import weight_init

from layers import AttentionLayer

from losses import FlowMatchingLoss
from losses import LatentFlowMatchingLoss
from losses import VAELoss
from metrics import Brier
from metrics import MR
from metrics import minADE
from metrics import minAHE
from metrics import minFDE
from metrics import minFHE
from modules import QCNetEncoder
from modules import QCNetFMDecoder
from modules import LatentSpaceEncoder
from modules import LatentSpaceDecoder
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, MultiStepLR

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object

class PrototypeSelectorTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, use_prototype_self_attention: bool = True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_prototype_self_attention = bool(use_prototype_self_attention)
        self.q_norm_hist = nn.LayerNorm(hidden_dim)
        self.kv_norm_hist = nn.LayerNorm(hidden_dim)
        self.hist_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.hist_ffn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.q_norm_map = nn.LayerNorm(hidden_dim)
        self.kv_norm_map = nn.LayerNorm(hidden_dim)
        self.map_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.map_ffn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.q_norm_agent = nn.LayerNorm(hidden_dim)
        self.kv_norm_agent = nn.LayerNorm(hidden_dim)
        self.agent_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.agent_ffn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        if self.use_prototype_self_attention:
            self.prototype_self_norm = nn.LayerNorm(hidden_dim)
            self.prototype_self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.prototype_self_ffn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        else:
            self.prototype_self_norm = None
            self.prototype_self_attn = None
            self.prototype_self_ffn = None
        self.apply(weight_init)

    @staticmethod
    def _attend(x: torch.Tensor, tokens: torch.Tensor, q_norm: nn.Module, kv_norm: nn.Module, attn: nn.MultiheadAttention, ffn: nn.Module) -> torch.Tensor:
        q = q_norm(x)
        kv = kv_norm(tokens)
        out, _ = attn(query=q, key=kv, value=kv, need_weights=False)
        x = x + out
        return x + ffn(x)

    def _prototype_self_attend(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_prototype_self_attention:
            return x
        h = self.prototype_self_norm(x)
        out, _ = self.prototype_self_attn(h, h, h, need_weights=False)
        x = x + out
        return x + self.prototype_self_ffn(x)

    def forward(self, prototype_tokens: torch.Tensor, c_hist_tokens: torch.Tensor, c_map_tokens: torch.Tensor, c_agent_tokens: torch.Tensor) -> torch.Tensor:
        x = self._attend(prototype_tokens, c_hist_tokens, self.q_norm_hist, self.kv_norm_hist, self.hist_attn, self.hist_ffn)
        x = self._attend(x, c_map_tokens, self.q_norm_map, self.kv_norm_map, self.map_attn, self.map_ffn)
        x = self._attend(x, c_agent_tokens, self.q_norm_agent, self.kv_norm_agent, self.agent_attn, self.agent_ffn)
        return self._prototype_self_attend(x)

class PrototypeSelector(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 latent_dim: int,
                 num_heads: int,
                 num_layers: int = 1,
                 dropout: float = 0.1,
                 use_prior_bias: bool = False,
                 num_prototypes: Optional[int] = None,
                 use_prototype_self_attention: bool = True,
                 agent_chunk_size: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_prior_bias = use_prior_bias
        self.num_prototypes = num_prototypes
        self.use_prototype_self_attention = bool(use_prototype_self_attention)
        self.agent_chunk_size = int(agent_chunk_size)
        self.prototype_latent_proj = nn.Sequential( nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        if num_prototypes is not None:
            self.prototype_id_embedding = nn.Embedding(num_prototypes, hidden_dim)
            self.prototype_id_gate = nn.Parameter(torch.tensor(0.1))
        else:
            self.prototype_id_embedding = None
            self.register_parameter("prototype_id_gate", None)
        self.blocks = nn.ModuleList([PrototypeSelectorTransformerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, use_prototype_self_attention=self.use_prototype_self_attention) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.to_logit = nn.Linear(hidden_dim, 1)
        if use_prior_bias:
            self.prior_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter("prior_scale", None)
        self.apply(weight_init)
        if self.prototype_id_embedding is not None:
            nn.init.normal_(self.prototype_id_embedding.weight, mean=0.0, std=0.02)

    def forward(self,
                c_hist_tokens: torch.Tensor,
                c_map_tokens: torch.Tensor,
                c_agent_tokens: torch.Tensor,
                prototype_latents: torch.Tensor,
                prototype_local_std: Optional[torch.Tensor] = None,
                prototype_log_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Score prototypes using continuous latent geometry plus discrete prototype ID.

        prototype_local_std is intentionally ignored here. It is still used by
        residual FM as the source Gaussian scale, but it is not mixed into the
        selector token, avoiding a shortcut where the selector prefers wide or
        narrow prototypes rather than scene-compatible prototypes.
        """
        if prototype_latents.ndim == 3:
            prototype_latents = prototype_latents[:, 0]
        N = c_hist_tokens.size(0)
        M = prototype_latents.size(0)
        proto_lat = prototype_latents.to(device=c_hist_tokens.device, dtype=c_hist_tokens.dtype)
        prototype_tokens = self.prototype_latent_proj(proto_lat)
        if self.prototype_id_embedding is not None:
            if M > self.prototype_id_embedding.num_embeddings:
                raise ValueError( f"prototype 数量 M={M} 超过 prototype_id_embedding 大小 " f"{self.prototype_id_embedding.num_embeddings}" )
            proto_ids = torch.arange(M, device=prototype_tokens.device, dtype=torch.long)
            proto_id_emb = self.prototype_id_embedding(proto_ids).to(dtype=prototype_tokens.dtype)
            prototype_tokens = prototype_tokens + self.prototype_id_gate * proto_id_emb
        base_prototype_tokens = prototype_tokens
        chunk_size = N if self.agent_chunk_size <= 0 else min(self.agent_chunk_size, N)
        logits_chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            prototype_tokens = base_prototype_tokens.unsqueeze(0).expand(end - start, M, self.hidden_dim)
            for block in self.blocks:
                prototype_tokens = block(prototype_tokens=prototype_tokens, c_hist_tokens=c_hist_tokens[start:end], c_map_tokens=c_map_tokens[start:end], c_agent_tokens=c_agent_tokens[start:end])
            logits_chunks.append(self.to_logit(self.out_norm(prototype_tokens)).squeeze(-1))
        logits = torch.cat(logits_chunks, dim=0)
        if self.use_prior_bias and prototype_log_prior is not None:
            prior = prototype_log_prior.to(device=logits.device, dtype=logits.dtype)
            logits = logits + self.prior_scale * prior.unsqueeze(0)
        return logits

class QCNetFM(pl.LightningModule):
    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 output_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_dec_layers: int,
                 num_heads: int,
                 num_hist_tokens: int,
                 num_map_tokens: int,
                 num_agent_tokens: int,
                 head_dim: int,
                 dropout: float,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 fm_num_steps: int,
                 scorer_only: bool,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 submission_dir: str,
                 submission_file_name: str,
                 vae_only: bool = False,
                 freeze_vae: bool = False,
                 vae_beta: float = 0.1,
                 vae_kl_warmup_epochs: int = 10,
                 vae_kl_start_beta: float = 0.0,
                 vae_gamma: float = 3.0,
                 vae_num_intents: int = 3,
                 band_weights: Optional[list] = None,
                 decoder_aux_ade_weight: float = 0.08,
                 decoder_aux_fde_weight: float = 0.04,
                 decoder_aux_warmup_epochs: int = 5,
                 decoder_aux_focal_only: bool = True,
                 trajectory_scale: float = 10.0,
                 residual_fm: bool = False,
                 prototype_bank_path: Optional[str] = None,
                 use_prototype_local_std: bool = True,
                 prototype_selector_loss_weight: float = 0.015,
                 prototype_std_floor: float = 0.03,
                 num_selector_layers: int = 1,
                 prototype_sampling_topk: int = 0,
                 residual_samples_per_prototype: int = 1,
                 selector_hard_ce_weight: float = 1.0,
                 selector_rank_loss_weight: float = 0.25,
                 selector_rank_margin: float = 0.2,
                 selector_num_hard_negatives: int = 12,
                 selector_use_prototype_self_attention: bool = True,
                 selector_agent_chunk_size: int = 64,
                 num_modality_token_refiner_layers: int = 1,
                 num_scene_token_refiner_layers: int = 1,
                 num_center_layers: int = 2,
                 residual_center_loss_weight: float = 1.0,
                 residual_source_scale: float = 0.7,
                 cfg_scene_dropout: float = 0.15,
                 cfg_guidance_scale: float = 1.5,
                 boundary_top2_training: bool = True,
                 **kwargs) -> None:
        super(QCNetFM, self).__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_freq_bands = num_freq_bands
        self.num_map_layers = num_map_layers
        self.num_agent_layers = num_agent_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.pl2pl_radius = pl2pl_radius
        self.time_span = time_span
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_t2m_steps = num_t2m_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.fm_num_steps = fm_num_steps
        self.scorer_only = scorer_only
        self.vae_only = vae_only
        self.freeze_vae = freeze_vae
        self.vae_beta = vae_beta
        self.vae_kl_warmup_epochs = vae_kl_warmup_epochs
        self.vae_kl_start_beta = vae_kl_start_beta
        self.vae_gamma = vae_gamma
        self.vae_num_intents = vae_num_intents
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.submission_dir = submission_dir
        self.submission_file_name = submission_file_name
        self.decoder_aux_ade_weight = decoder_aux_ade_weight
        self.decoder_aux_fde_weight = decoder_aux_fde_weight
        self.decoder_aux_warmup_epochs = decoder_aux_warmup_epochs
        self.decoder_aux_focal_only = decoder_aux_focal_only
        self.trajectory_scale = trajectory_scale
        self.residual_fm = residual_fm
        self.prototype_bank_path = prototype_bank_path
        self.use_prototype_local_std = use_prototype_local_std
        self.prototype_selector_loss_weight = prototype_selector_loss_weight
        self.prototype_std_floor = prototype_std_floor
        self.num_selector_layers = int(num_selector_layers)
        self.prototype_sampling_topk = int(prototype_sampling_topk)
        self.residual_samples_per_prototype = max(1, int(residual_samples_per_prototype))
        self.selector_hard_ce_weight = float(selector_hard_ce_weight)
        self.selector_rank_loss_weight = float(selector_rank_loss_weight)
        self.selector_rank_margin = float(selector_rank_margin)
        self.selector_num_hard_negatives = max(1, int(selector_num_hard_negatives))
        self.selector_use_prototype_self_attention = bool(selector_use_prototype_self_attention)
        self.selector_agent_chunk_size = int(selector_agent_chunk_size)
        self.num_modality_token_refiner_layers = max(0, int(num_modality_token_refiner_layers))
        self.num_scene_token_refiner_layers = max(0, int(num_scene_token_refiner_layers))
        self.num_center_layers = max(1, int(num_center_layers))
        self.residual_center_loss_weight = float(residual_center_loss_weight)
        self.residual_source_scale = float(residual_source_scale)
        self.cfg_scene_dropout = float(cfg_scene_dropout)
        self.cfg_guidance_scale = float(cfg_guidance_scale)
        self.boundary_top2_training = bool(boundary_top2_training)
        if not 0.0 <= self.cfg_scene_dropout < 1.0:
            raise ValueError( "cfg_scene_dropout 必须位于 [0,1)：" f"实际 {self.cfg_scene_dropout}" )
        if self.cfg_guidance_scale < 0.0:
            raise ValueError( "cfg_guidance_scale 必须 >= 0：" f"实际 {self.cfg_guidance_scale}" )
        if self.residual_center_loss_weight < 0.0:
            raise ValueError(f"residual_center_loss_weight 必须 >= 0，实际 {self.residual_center_loss_weight}")
        if self.residual_source_scale <= 0.0:
            raise ValueError(f"residual_source_scale 必须 > 0，实际 {self.residual_source_scale}")
        selector_weights = {
            "selector_hard_ce_weight": self.selector_hard_ce_weight,
            "selector_rank_loss_weight": self.selector_rank_loss_weight,
            "prototype_selector_loss_weight": self.prototype_selector_loss_weight,
        }
        invalid_weights = {name: value for name, value in selector_weights.items() if value < 0.0}
        if invalid_weights:
            raise ValueError(f"selector loss 权重必须 >= 0，非法值：{invalid_weights}")
        if self.residual_fm:
            if self.prototype_sampling_topk <= 0:
                self.prototype_sampling_topk = max( 1, (self.num_modes + self.residual_samples_per_prototype - 1) // self.residual_samples_per_prototype)
            if self.prototype_sampling_topk * self.residual_samples_per_prototype < self.num_modes:
                raise ValueError(
                    "prototype_sampling_topk * residual_samples_per_prototype 必须大于等于 num_modes："
                    f"prototype_sampling_topk={self.prototype_sampling_topk}, "
                    f"residual_samples_per_prototype={self.residual_samples_per_prototype}, "
                    f"num_modes={self.num_modes}"
                )
        self.latent_regression_only = False
        self.encoder = QCNetEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.fm_decoder = QCNetFMDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_t2m_steps=num_t2m_steps,
            pl2m_radius=pl2m_radius,
            a2m_radius=a2m_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_dec_layers,
            num_heads=num_heads,
            num_hist_tokens=num_hist_tokens,
            num_map_tokens=num_map_tokens,
            num_agent_tokens=num_agent_tokens,
            head_dim=head_dim,
            dropout=dropout,
            vae_num_intents=vae_num_intents,
            num_modality_token_refiner_layers=self.num_modality_token_refiner_layers,
            num_scene_token_refiner_layers=self.num_scene_token_refiner_layers,
            num_center_layers=self.num_center_layers,
        )
        if band_weights is not None:
            bw_tensor = torch.tensor(band_weights, dtype=torch.float32)
        else:
            bw_tensor = None
        self.latent_fm_loss = LatentFlowMatchingLoss(band_weights=bw_tensor)
        self.latent_encoder = LatentSpaceEncoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            input_dim=output_dim,
            num_future_steps=num_future_steps,
            num_intents=vae_num_intents,
            num_freq_bands=self.num_freq_bands,
        )
        self.latent_decoder = LatentSpaceDecoder(vae=self.latent_encoder.vae)
        self.vae_loss = VAELoss(beta=vae_beta, gamma=vae_gamma)
        self.minADE = minADE(max_guesses=6)
        self.minAHE = minAHE(max_guesses=6)
        self.minFDE = minFDE(max_guesses=6)
        self.minFHE = minFHE(max_guesses=6)
        self.MR = MR(max_guesses=6)
        self.test_predictions = dict()
        default_means = [0.0002, -0.0189, -0.0105, 0.0394, 0.0083]
        default_stds = [0.5341, 0.9854, 0.6526, 1.1099, 0.6366]
        if self.latent_dim == len(default_means):
            means, stds = default_means, default_stds
        else:
            means = [0.0] * self.latent_dim
            stds = [1.0] * self.latent_dim
        if any(s <= 0.0 for s in stds):
            raise ValueError("z_std 的每一维都必须大于 0。")
        if len(means) != self.latent_dim or len(stds) != self.latent_dim:
            raise ValueError( "z_mean/z_std 的维数必须与 latent_dim 一致：" f"latent_dim={self.latent_dim}, len(mean)={len(means)}, len(std)={len(stds)}" )
        if any(s <= 0.0 for s in stds):
            raise ValueError("z_std 的每一维都必须大于 0。")
        self.register_buffer('z_mean', torch.tensor(means, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer('z_std', torch.tensor(stds, dtype=torch.float32).view(1, 1, -1))
        # FM 工作在 centered-raw latent：z_c = z_raw - mean。
        # 解码时只加回 mean，不乘 std。
        class CenteredRawDecoderWrapper(nn.Module):
            def __init__(self, decoder, mean):
                super().__init__()
                self.decoder = decoder
                self.mean = mean

            def forward(self, z, *args, **kwargs):
                mean = self.mean.to(device=z.device, dtype=z.dtype)
                return self.decoder(z + mean, *args, **kwargs)
        self.latent_decoder = CenteredRawDecoderWrapper(self.latent_decoder, self.z_mean)
        if self.residual_fm:
            if self.prototype_bank_path is None:
                raise ValueError("启用 residual_fm=True 时必须提供 prototype_bank_path。")
            self._load_prototype_bank(self.prototype_bank_path)
            num_prototypes = int(self.prototype_latents_centered_raw.size(0))
            if hasattr(self.fm_decoder, "init_prototype_id_embedding"):
                self.fm_decoder.init_prototype_id_embedding(num_prototypes)
            self.prototype_selector = PrototypeSelector(
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                num_heads=self.num_heads,
                num_layers=self.num_selector_layers,
                dropout=self.dropout,
                use_prior_bias=False,
                num_prototypes=num_prototypes,
                use_prototype_self_attention=self.selector_use_prototype_self_attention,
                agent_chunk_size=self.selector_agent_chunk_size,
            )
        else:
            self.prototype_selector = None
        if self.freeze_vae:
            for param in self.latent_encoder.parameters():
                param.requires_grad_(False)
            for param in self.latent_decoder.parameters():
                param.requires_grad_(False)
        for module_name, module in self.encoder.named_modules():
            if isinstance(module, AttentionLayer):
                module.debug_name = module_name

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        """Allow loading a pre-center checkpoint by zero-initializing missing center-head keys."""
        state_dict = checkpoint.get("state_dict", None)
        if not isinstance(state_dict, dict):
            return
        current_state = self.state_dict()
        added = 0
        for key, value in current_state.items():
            if key.startswith("fm_decoder.residual_center_head.") and key not in state_dict:
                state_dict[key] = value.detach().cpu()
                added += 1
        if added > 0:
            print(f"[INFO] loaded legacy checkpoint: initialized {added} residual-center parameters from current model defaults.")

    def _load_prototype_bank(self, bank_path: str) -> None:
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        if "prototype_latents_centered_raw" not in bank:
            raise KeyError("prototype_bank.pt 中缺少 prototype_latents_centered_raw。")
        proto = bank["prototype_latents_centered_raw"].float()
        if proto.ndim == 2:
            proto = proto.unsqueeze(1)
        if proto.ndim != 3:
            raise ValueError(f"prototype_latents_centered_raw 应为 [M,K,D] 或 [M,D]，实际 {tuple(proto.shape)}")
        if proto.size(1) != self.vae_num_intents or proto.size(2) != self.latent_dim:
            raise ValueError( "prototype latent 形状与模型不一致：" f"prototype={tuple(proto.shape)}, vae_num_intents={self.vae_num_intents}, latent_dim={self.latent_dim}" )
        if "prototype_residual_std_population" in bank:
            local_std = bank["prototype_residual_std_population"].float()
        elif "residual_std" in bank:
            local_std = bank["residual_std"].float()
        else:
            local_std = self.z_std.detach().cpu().float()
        if local_std.ndim == 1:
            local_std = local_std.view(1, 1, -1).expand_as(proto).clone()
        elif local_std.ndim == 2:
            local_std = local_std.unsqueeze(1)
        if local_std.shape != proto.shape:
            if local_std.size(0) == 1 and local_std.size(1) == 1 and local_std.size(2) == proto.size(2):
                local_std = local_std.expand_as(proto).clone()
            else:
                raise ValueError( "prototype residual std 形状与 prototype latent 不一致：" f"std={tuple(local_std.shape)}, proto={tuple(proto.shape)}" )
        local_std = torch.nan_to_num(local_std, nan=float(self.prototype_std_floor), posinf=1.0, neginf=float(self.prototype_std_floor))
        local_std = local_std.clamp_min(float(self.prototype_std_floor))
        if "prototype_residual_count" in bank:
            count = bank["prototype_residual_count"].float()
        else:
            count = torch.ones(proto.size(0), dtype=torch.float32)
        count = count.clamp_min(1.0)
        log_prior = (count / count.sum()).clamp_min(1e-12).log()
        self.register_buffer("prototype_latents_centered_raw", proto)
        self.register_buffer("prototype_residual_std_population", local_std)
        self.register_buffer("prototype_residual_count", count)
        self.register_buffer("prototype_log_prior", log_prior)

    def _assign_nearest_prototype(self, z_centered_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign each GT latent to its nearest prototype in raw centered latent space."""
        proto = self.prototype_latents_centered_raw.to(device=z_centered_raw.device, dtype=z_centered_raw.dtype)
        # dist: [N_a, M], summing over VAE latent tokens and latent dimensions.
        dist = (z_centered_raw.unsqueeze(1) - proto.unsqueeze(0)).pow(2).sum(dim=(-1, -2))
        prototype_index = dist.argmin(dim=1)
        prototype_latent = proto[prototype_index]
        local_std = self.prototype_residual_std_population.to(device=z_centered_raw.device, dtype=z_centered_raw.dtype)[prototype_index]
        return prototype_index, prototype_latent, local_std

    def _sample_residual_noise(self, residual_target: torch.Tensor, local_std: torch.Tensor, agent_batch: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x_0, t = FlowMatchingLoss.sample_noise_and_time_latent( self.vae_num_intents, residual_target.size(0), self.latent_dim, self.device, agent_batch )
        if self.use_prototype_local_std:
            source_std = local_std.to(device=x_0.device, dtype=x_0.dtype)
        else:
            source_std = self.z_std.to(device=x_0.device, dtype=x_0.dtype)
        x_0 = x_0 * source_std * self.residual_source_scale
        return x_0, t

    def _compute_selector_outputs(self, scene_conditions: Mapping[str, torch.Tensor], agent_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if self.prototype_selector is None:
            raise RuntimeError("residual_fm selector 尚未初始化。")
        cond = scene_conditions if agent_mask is None else {key: value[agent_mask] for key, value in scene_conditions.items()}
        selector_logits = self.prototype_selector(
            c_hist_tokens=cond["c_hist_tokens"],
            c_map_tokens=cond["c_map_tokens"],
            c_agent_tokens=cond["c_agent_tokens"],
            prototype_latents=self.prototype_latents_centered_raw,
            prototype_local_std=self.prototype_residual_std_population,
            prototype_log_prior=None,
        )
        return {
            "logits": selector_logits,
            "logits_std": selector_logits.std(dim=-1, unbiased=False).mean(),
        }

    def _select_prototypes_for_sampling(
            self,
            data: HeteroData,
            scene_enc: Dict[str, torch.Tensor],
            num_modes: int,
            scene_conditions: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top prototypes and optionally repeat residual samples per prototype."""
        if self.prototype_selector is None:
            raise RuntimeError("residual_fm 采样需要 prototype_selector。")
        if scene_conditions is None:
            _, scene_conditions = self.fm_decoder.build_scene_conditions(data, scene_enc)
        selector_outputs = self._compute_selector_outputs(scene_conditions=scene_conditions)
        logits = selector_outputs["logits"]
        repeats = max(1, int(self.residual_samples_per_prototype))
        proto_topk = int(self.prototype_sampling_topk)
        if proto_topk <= 0:
            proto_topk = max(1, (num_modes + repeats - 1) // repeats)
        proto_topk = min(proto_topk, logits.size(-1))
        if proto_topk * repeats < num_modes:
            repeats = (num_modes + proto_topk - 1) // proto_topk
        top_score, top_index = torch.topk(logits, k=proto_topk, dim=-1)
        proto_bank = self.prototype_latents_centered_raw.to(device=logits.device, dtype=logits.dtype)
        std_bank = self.prototype_residual_std_population.to(device=logits.device, dtype=logits.dtype)
        proto = proto_bank[top_index]
        local_std = std_bank[top_index]
        pi_proto = F.softmax(top_score, dim=-1)
        if repeats > 1:
            top_index = top_index.repeat_interleave(repeats, dim=1)
            proto = proto.repeat_interleave(repeats, dim=1)
            local_std = local_std.repeat_interleave(repeats, dim=1)
            pi = pi_proto.repeat_interleave(repeats, dim=1) / float(repeats)
        else:
            pi = pi_proto
        if proto.size(1) > num_modes:
            top_index = top_index[:, :num_modes]
            proto = proto[:, :num_modes]
            local_std = local_std[:, :num_modes]
            pi = pi[:, :num_modes]
        if proto.size(1) != num_modes:
            raise RuntimeError( f"采样 mode 数量不等于 num_modes：proto_modes={proto.size(1)}, " f"num_modes={num_modes}, prototype_sampling_topk={proto_topk}, repeats={repeats}" )
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return top_index, proto, local_std, pi

    def _compute_selector_loss(self, selector_outputs: Mapping[str, torch.Tensor], hard_target: Optional[torch.Tensor] = None, support_ids: Optional[torch.Tensor] = None, support_weights: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits=selector_outputs["logits"]; zero=logits.sum()*0.0; ce_loss=zero; rank_loss=zero; num_classes=int(logits.size(1))
        if support_ids is not None or support_weights is not None:
            if support_ids is None or support_weights is None: raise ValueError("support_ids和support_weights必须同时提供。")
            support_ids=support_ids.to(device=logits.device,dtype=torch.long); support_weights=support_weights.to(device=logits.device,dtype=logits.dtype)
            if support_ids.ndim!=2 or support_weights.shape!=support_ids.shape or support_ids.size(0)!=logits.size(0):
                raise ValueError(f"selector support形状错误：ids={tuple(support_ids.shape)}, weights={tuple(support_weights.shape)}, logits={tuple(logits.shape)}")
            positive_weight=torch.isfinite(support_weights)&support_weights.gt(0)
            bad_positive=positive_weight&(support_ids.lt(0)|support_ids.ge(num_classes))
            if bad_positive.any():
                bad=support_ids[bad_positive]
                raise RuntimeError(f"selector support id越界：num_classes={num_classes}, count={int(bad.numel())}, min={int(bad.min())}, max={int(bad.max())}")
            valid_support=positive_weight&support_ids.ge(0)&support_ids.lt(num_classes)
            clean_weights=torch.where(valid_support,support_weights.clamp_min(0),torch.zeros_like(support_weights))
            row_sum=clean_weights.sum(dim=1,keepdim=True)
            if row_sum.le(0).any(): raise RuntimeError(f"selector存在无有效support的样本：count={int(row_sum.le(0).sum())}")
            clean_weights=clean_weights/row_sum
            safe_ids=support_ids.clamp(0,max(num_classes-1,0))
            logits=logits.float()
            safe_ids=safe_ids.to(device=logits.device,dtype=torch.long)
            clean_weights=clean_weights.to(device=logits.device,dtype=logits.dtype)
            target_prob=torch.zeros_like(logits)
            target_prob.scatter_add_(1,safe_ids,clean_weights)
            ce_loss=-(target_prob*F.log_softmax(logits,dim=-1)).sum(dim=-1).mean()
            if self.selector_rank_loss_weight>0.0 and num_classes>1:
                positive_mask=target_prob.gt(0); positive_logit=(target_prob*logits).sum(dim=-1,keepdim=True)
                negative_logits=logits.masked_fill(positive_mask,float("-inf")); hard_k=min(self.selector_num_hard_negatives,num_classes-1)
                hard_negative=negative_logits.topk(hard_k,dim=1).values; finite=torch.isfinite(hard_negative)
                if finite.any():
                    rank_terms=F.softplus(hard_negative-positive_logit+self.selector_rank_margin); rank_loss=rank_terms[finite].mean()
        elif hard_target is not None:
            hard_target=hard_target.to(device=logits.device,dtype=torch.long)
            if hard_target.numel()>0:
                invalid=hard_target.lt(0)|hard_target.ge(num_classes)
                if invalid.any():
                    bad=hard_target[invalid]
                    raise RuntimeError(f"selector hard_target越界：num_classes={num_classes}, count={int(bad.numel())}, min={int(bad.min())}, max={int(bad.max())}")
                ce_loss=F.cross_entropy(logits,hard_target,label_smoothing=0.0)
                if self.selector_rank_loss_weight>0.0 and num_classes>1:
                    positive_logit=logits.gather(1,hard_target[:,None]); positive_mask=F.one_hot(hard_target,num_classes=num_classes).bool()
                    negative_logits=logits.masked_fill(positive_mask,float("-inf")); hard_k=min(self.selector_num_hard_negatives,num_classes-1)
                    rank_loss=F.softplus(negative_logits.topk(hard_k,dim=1).values-positive_logit+self.selector_rank_margin).mean()
        weighted_ce=self.selector_hard_ce_weight*ce_loss; weighted_rank=self.selector_rank_loss_weight*rank_loss
        return weighted_ce+weighted_rank,{"ce":ce_loss.detach(),"topk_rank":rank_loss.detach(),"weighted_ce":weighted_ce.detach(),"weighted_rank":weighted_rank.detach()}


    @staticmethod
    def _get_agent_cached_field(data, key: str):
        """Read cached per-agent fields from PyG HeteroData/Batch or plain dict."""
        try:
            agent_store = data["agent"]
            try:
                value = agent_store[key]
                if value is not None:
                    return value
            except Exception:
                pass
            mapping = getattr(agent_store, "_mapping", None)
            if mapping is not None and key in mapping:
                return mapping[key]
            if hasattr(agent_store, key):
                value = getattr(agent_store, key)
                if value is not None:
                    return value
            if hasattr(agent_store, "get"):
                value = agent_store.get(key, None)
                if value is not None:
                    return value
        except Exception:
            pass
        if isinstance(data, dict):
            if key in data:
                return data[key]
            agent_store = data.get("agent", None)
            if isinstance(agent_store, dict):
                return agent_store.get(key, None)
        return None

    def _get_cached_residual_targets(self, data, predict_mask: torch.Tensor, sample_boundary_support: bool = False):
        """读取离线assignment。训练时boundary样本按Top-2权重采样anchor；验证时固定使用primary。"""
        primary_index=self._get_agent_cached_field(data,"prototype_index")
        z_target=self._get_agent_cached_field(data,"z_gt_centered_raw")
        cached_valid=self._get_agent_cached_field(data,"valid_agent_mask")
        support_ids=self._get_agent_cached_field(data,"support_prototype_ids")
        support_weights=self._get_agent_cached_field(data,"support_weights")
        missing=[name for name,value in (("prototype_index",primary_index),("z_gt_centered_raw",z_target)) if value is None]
        if self.boundary_top2_training:
            missing.extend(name for name,value in (("support_prototype_ids",support_ids),("support_weights",support_weights)) if value is None)
        if missing: raise RuntimeError("residual_fm缺少离线assignment字段："+", ".join(missing))
        device=self.device; primary_index=primary_index.to(device=device,dtype=torch.long)
        if primary_index.ndim>1: primary_index=primary_index.view(primary_index.size(0),-1)[:,0]
        z_target=z_target.to(device=device,dtype=torch.float32)
        if z_target.ndim==2: z_target=z_target.unsqueeze(1)
        if z_target.ndim!=3 or z_target.size(1)!=self.vae_num_intents or z_target.size(2)!=self.latent_dim:
            raise ValueError(f"z_gt_centered_raw形状错误：{tuple(z_target.shape)}，期望[N,{self.vae_num_intents},{self.latent_dim}]")
        num_agents=z_target.size(0); num_proto=int(self.prototype_latents_centered_raw.size(0))
        if primary_index.numel()!=num_agents: raise ValueError(f"prototype_index数量与agent不一致：{primary_index.numel()} vs {num_agents}")
        bad_primary=primary_index.ge(num_proto)
        if bad_primary.any():
            bad=primary_index[bad_primary]
            raise RuntimeError(f"prototype_index超过bank范围：num_proto={num_proto}, count={int(bad.numel())}, min={int(bad.min())}, max={int(bad.max())}")
        if cached_valid is None: valid_mask=predict_mask.any(dim=-1).to(device=device).bool()
        else: valid_mask=cached_valid.to(device=device).bool()
        valid_mask=valid_mask&predict_mask.any(dim=-1).to(device=device).bool()&primary_index.ge(0)&primary_index.lt(num_proto)
        if self.boundary_top2_training:
            support_ids=support_ids.to(device=device,dtype=torch.long); support_weights=support_weights.to(device=device,dtype=torch.float32)
            if support_ids.ndim==1: support_ids=support_ids[:,None]
            if support_weights.ndim==1: support_weights=support_weights[:,None]
            if support_ids.ndim!=2 or support_weights.shape!=support_ids.shape or support_ids.size(0)!=num_agents:
                raise ValueError(f"Top-2 support形状错误：ids={tuple(support_ids.shape)}, weights={tuple(support_weights.shape)}, N={num_agents}")
            positive=torch.isfinite(support_weights)&support_weights.gt(0)
            bad_positive=positive&(support_ids.lt(0)|support_ids.ge(num_proto))
            if bad_positive.any():
                bad=support_ids[bad_positive]
                raise RuntimeError(f"正权重support id越界：num_proto={num_proto}, count={int(bad.numel())}, min={int(bad.min())}, max={int(bad.max())}")
            support_valid=positive&support_ids.ge(0)&support_ids.lt(num_proto)
            support_weights=torch.where(support_valid,support_weights.clamp_min(0),torch.zeros_like(support_weights))
            row_sum=support_weights.sum(dim=1,keepdim=True); fallback=valid_mask&row_sum.squeeze(1).le(0)
            if fallback.any():
                support_ids=support_ids.clone(); support_weights=support_weights.clone()
                support_ids[fallback,0]=primary_index[fallback]; support_weights[fallback,0]=1.0; row_sum=support_weights.sum(dim=1,keepdim=True)
            support_weights=torch.where(row_sum.gt(0),support_weights/row_sum.clamp_min(1e-12),support_weights)
        else:
            support_ids=torch.stack([primary_index,torch.full_like(primary_index,-1)],dim=1)
            support_weights=torch.zeros((num_agents,2),device=device,dtype=torch.float32); support_weights[:,0]=valid_mask.float()
        boundary_mask=(support_weights.gt(0).sum(dim=1)>1)&valid_mask
        safe_primary=primary_index.clamp(0,max(num_proto-1,0))
        if sample_boundary_support and self.boundary_top2_training:
            safe_support=support_ids.clamp(0,max(num_proto-1,0)); sample_weights=support_weights.clone(); zero_row=sample_weights.sum(dim=1).le(0)
            if zero_row.any(): sample_weights[zero_row,0]=1.0
            selected_slot=torch.multinomial(sample_weights,num_samples=1).squeeze(1)
            selected_index=safe_support.gather(1,selected_slot[:,None]).squeeze(1); selected_index=torch.where(valid_mask,selected_index,safe_primary)
        else:
            selected_index=safe_primary
        proto_bank=self.prototype_latents_centered_raw.to(device=device,dtype=z_target.dtype)
        std_bank=self.prototype_residual_std_population.to(device=device,dtype=z_target.dtype)
        prototype_latent=proto_bank[selected_index]; prototype_local_std=std_bank[selected_index].clamp_min(float(self.prototype_std_floor))
        z_residual=z_target-prototype_latent
        return z_target,z_residual,selected_index,prototype_latent,prototype_local_std,valid_mask,support_ids,support_weights,boundary_mask


    @torch.no_grad()
    def _get_standardized_latent_target(self, target: torch.Tensor, predict_mask: torch.Tensor) -> torch.Tensor:
        self.latent_encoder.eval()
        z_target_raw = self.latent_encoder.encode(target, predict_mask=predict_mask)
        z_target_centered_raw = z_target_raw - self.z_mean
        return z_target_centered_raw

    def forward(self, data: HeteroData, scene_enc: dict, x_t: torch.Tensor, t: torch.Tensor,
                prototype_latent: Optional[torch.Tensor] = None,
                prototype_index: Optional[torch.Tensor] = None,
                scene_conditions: Optional[Mapping[str, torch.Tensor]] = None) -> torch.Tensor:
        v_theta = self.fm_decoder( data, scene_enc, x_t, t, prototype_latent=prototype_latent, prototype_index=prototype_index, scene_conditions=scene_conditions)
        return v_theta

    def _compute_residual_center_loss(
            self,
            center_pred: torch.Tensor,
            center_target: torch.Tensor,
            valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not valid_mask.any():
            zero = center_pred.sum() * 0.0
            return zero, zero.detach(), zero.detach()
        pred = center_pred[valid_mask].float()
        target = center_target[valid_mask].float()
        # MSE directly minimizes the remaining residual energy assigned to FM.
        loss = F.mse_loss(pred, target)
        error_energy = (pred.detach() - target.detach()).pow(2).mean()
        target_energy = target.detach().pow(2).mean().clamp_min(1e-12)
        rmse = error_energy.clamp_min(1e-12).sqrt()
        explained_energy = 1.0 - error_energy / target_energy
        return loss, rmse, explained_energy

    def _get_decoder_aux_scale(self) -> float:
        if self.decoder_aux_warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(self.current_epoch + 1) / float(self.decoder_aux_warmup_epochs))

    def _get_current_vae_beta(self) -> float:
        if self.vae_kl_warmup_epochs <= 0:
            return float(self.vae_beta)
        if self.trainer is None:
            return float(self.vae_beta)
        num_batches = self.trainer.num_training_batches
        if not isinstance(num_batches, int) or num_batches <= 0:
            progress = min(float(self.current_epoch)/ max(float(self.vae_kl_warmup_epochs), 1.0), 1.0)
        else:
            warmup_steps = max(1,self.vae_kl_warmup_epochs * num_batches)
            progress = min(float(self.global_step) / float(warmup_steps),1.0)
        beta_now = self.vae_kl_start_beta + progress * (self.vae_beta - self.vae_kl_start_beta)
        return float(beta_now)

    def _decoder_aware_trajectory_loss(
            self,
            v_theta: torch.Tensor,
            x_t: torch.Tensor,
            t: torch.Tensor,
            z_target: torch.Tensor,
            predict_mask: torch.Tensor,
            category: torch.Tensor,
            prototype_latent: Optional[torch.Tensor] = None,
            residual_center: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Time-aware decoder center-consistency loss.

        Original FM:
            z_hat_1|t = x_t + (1 - t) v_theta.

        Residual FM:
            r_hat_1|t = x_t + (1 - t) v_theta,
            z_hat_1|t = prototype_latent + r_hat_1|t.

        The target branch is always D(z_target), where z_target is the full
        centered-raw latent. This keeps the auxiliary objective aligned with
        the latent FM optimum.
        """
        future_valid_mask = predict_mask.any(dim=-1)
        if self.decoder_aux_focal_only:
            aux_agent_mask = (category == 3) & future_valid_mask
        else:
            aux_agent_mask = future_valid_mask
        num_aux_agents = int(aux_agent_mask.sum().item())
        if num_aux_agents == 0:
            zero = v_theta.sum() * 0.0
            return zero, zero, zero.detach(), 0
        remaining_time = (1.0 - t[aux_agent_mask]).view(-1, 1, 1)
        residual_terminal = x_t[aux_agent_mask] + remaining_time * v_theta[aux_agent_mask]
        if prototype_latent is not None:
            anchor = prototype_latent[aux_agent_mask]
            if residual_center is not None:
                anchor = anchor + residual_center[aux_agent_mask]
            z_pred_terminal = anchor + residual_terminal
        else:
            z_pred_terminal = residual_terminal
        z_target_aux = z_target[aux_agent_mask]
        self.latent_decoder.eval()
        traj_pred = self.latent_decoder(z_pred_terminal)
        with torch.no_grad():
            target_aux = self.latent_decoder(z_target_aux)
        mask_aux = predict_mask[aux_agent_mask].bool()
        error_m = ( traj_pred[..., :self.output_dim] - target_aux[..., :self.output_dim] ) * self.trajectory_scale
        distance_m = torch.sqrt(error_m.pow(2).sum(dim=-1) + 1e-8)
        mask_float = mask_aux.to(distance_m.dtype)
        valid_steps = mask_float.sum(dim=-1).clamp_min(1.0)
        ade_per_agent = (distance_m * mask_float).sum(dim=-1) / valid_steps
        ade_loss_m = ade_per_agent.mean()
        time_index = torch.arange( mask_aux.size(1), device=mask_aux.device ).view(1, -1)
        last_valid_index = time_index.masked_fill(~mask_aux, -1).max(dim=-1).values
        agent_index = torch.arange(num_aux_agents, device=mask_aux.device)
        fde_per_agent = distance_m[agent_index, last_valid_index]
        fde_loss_m = fde_per_agent.mean()
        latent_terminal_rmse = ( z_pred_terminal.detach() - z_target_aux.detach() ).pow(2).mean().clamp_min(1e-12).sqrt()
        return ade_loss_m, fde_loss_m, latent_terminal_rmse, num_aux_agents

    def training_step(self, data, batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        if self.vae_only:
            return self._training_step_vae(data)
        if self.scorer_only:
            return self._training_step_scorer(data)
        target = data['agent']['target'][..., :self.output_dim] / float(self.trajectory_scale)
        predict_mask = data['agent']['predict_mask'][:, self.num_historical_steps:].bool()
        agent_batch = data['agent'].get('batch', None)
        prototype_index = None
        prototype_latent = None
        prototype_local_std = None
        selector_support_ids = None
        selector_support_weights = None
        residual_center = None
        original_residual = None
        boundary_mask = torch.zeros(target.size(0), device=self.device, dtype=torch.bool)
        if self.residual_fm:
            (
                z_target,
                original_residual,
                prototype_index,
                prototype_latent,
                prototype_local_std,
                valid_mask,
                selector_support_ids,
                selector_support_weights,
                boundary_mask,
            ) = self._get_cached_residual_targets(data, predict_mask, sample_boundary_support=True)
        else:
            self.latent_encoder.eval()
            with torch.no_grad():
                z_target = self.latent_encoder.encode(target, predict_mask=predict_mask) - self.z_mean
            valid_mask = predict_mask.any(dim=-1)
        scene_enc = self.encoder(data)
        _, scene_conditions = self.fm_decoder.build_scene_conditions(data, scene_enc)
        center_loss = scene_conditions['c_hist'].sum() * 0.0
        center_rmse = center_loss.detach()
        center_explained_energy = center_loss.detach()
        center_for_fm = None
        if self.residual_fm:
            residual_center = self.fm_decoder.predict_residual_center(
                scene_conditions=scene_conditions,
                prototype_latent=prototype_latent,
                prototype_index=prototype_index,
            )
            center_loss, center_rmse, center_explained_energy = self._compute_residual_center_loss(
                center_pred=residual_center,
                center_target=original_residual,
                valid_mask=valid_mask,
            )
            # FM-related losses see only the numeric center value. They cannot update
            # the center head through this branch; the center head is trained only by center_loss.
            center_for_fm = residual_center.detach()
            z_fm_target = original_residual - center_for_fm
            x_0, t = self._sample_residual_noise(z_fm_target, prototype_local_std, agent_batch)
        else:
            z_fm_target = z_target
            x_0, t = FlowMatchingLoss.sample_noise_and_time_latent(
                self.vae_num_intents, target.size(0), self.latent_dim, self.device, agent_batch
            )
            x_0 = x_0 * self.z_std.to(device=x_0.device, dtype=x_0.dtype)
        if self.current_epoch == 0 and batch_idx == 0 and agent_batch is not None:
            first_scene_mask = agent_batch == agent_batch[0]
            print('first scene t:')
            print(t[first_scene_mask][:20].detach().cpu())
            print('unique t in first scene:')
            print(torch.unique(t[first_scene_mask]).numel())
        t_exp = t[:, None, None]
        x_t = (1.0 - t_exp) * x_0 + t_exp * z_fm_target
        cfg_drop_mask = torch.zeros(
            scene_conditions['c_hist'].size(0),
            device=scene_conditions['c_hist'].device,
            dtype=torch.bool,
        )
        fm_scene_conditions = scene_conditions
        if self.residual_fm and self.cfg_scene_dropout > 0.0:
            cfg_drop_mask = torch.rand(cfg_drop_mask.size(0), device=cfg_drop_mask.device) < self.cfg_scene_dropout
            cfg_drop_mask = cfg_drop_mask & valid_mask.to(cfg_drop_mask.device)
            fm_scene_conditions = self.fm_decoder.apply_scene_condition_dropout(
                scene_conditions=scene_conditions,
                drop_mask=cfg_drop_mask,
            )
        v_theta = self(
            data=data,
            scene_enc=scene_enc,
            x_t=x_t,
            t=t,
            prototype_latent=prototype_latent,
            prototype_index=prototype_index,
            scene_conditions=fm_scene_conditions,
        )
        if not valid_mask.any():
            return v_theta.sum() * 0.0
        v_theta_valid = v_theta[valid_mask]
        z_fm_target_valid = z_fm_target[valid_mask]
        x_0_valid = x_0[valid_mask]
        fm_loss, loss_dict = self.latent_fm_loss(v_theta_valid, z_fm_target_valid, x_0_valid)
        selector_loss = v_theta.sum() * 0.0
        zero = selector_loss.detach()
        selector_acc = zero
        selector_topk = zero
        selector_logits_std = zero
        selector_loss_dict = {key: zero for key in ('ce', 'topk_rank', 'weighted_ce', 'weighted_rank')}
        if self.residual_fm and self.prototype_selector is not None:
            selector_label_mask = valid_mask & selector_support_weights.sum(dim=1).gt(0)
            if selector_label_mask.any():
                selector_outputs = self._compute_selector_outputs(scene_conditions=scene_conditions, agent_mask=selector_label_mask)
                selector_logits = selector_outputs['logits']
                support_ids_now = selector_support_ids[selector_label_mask]
                support_weights_now = selector_support_weights[selector_label_mask]
                selector_loss, selector_loss_dict = self._compute_selector_loss(
                    selector_outputs=selector_outputs,
                    support_ids=support_ids_now,
                    support_weights=support_weights_now,
                )
                positive = support_weights_now.gt(0)
                predicted = selector_logits.argmax(dim=-1)
                selector_acc = ((predicted[:, None] == support_ids_now) & positive).any(dim=1).float().mean()
                selector_logits_std = selector_outputs['logits_std'].detach().float()
                topk = min(self.num_modes, selector_logits.size(-1))
                top_indices = selector_logits.topk(topk, dim=-1).indices
                selector_topk = ((top_indices[:, :, None] == support_ids_now[:, None, :]) & positive[:, None, :]).any(dim=(1, 2)).float().mean()
        aux_ade_m, aux_fde_m, aux_latent_rmse, num_aux_agents = self._decoder_aware_trajectory_loss(
            v_theta=v_theta,
            x_t=x_t,
            t=t,
            z_target=z_target,
            predict_mask=predict_mask,
            category=data['agent']['category'],
            prototype_latent=prototype_latent,
            residual_center=center_for_fm,
        )
        aux_scale = self._get_decoder_aux_scale()
        weighted_aux_ade = aux_scale * self.decoder_aux_ade_weight * aux_ade_m
        weighted_aux_fde = aux_scale * self.decoder_aux_fde_weight * aux_fde_m
        weighted_selector = self.prototype_selector_loss_weight * selector_loss
        weighted_center = self.residual_center_loss_weight * center_loss
        loss = fm_loss + weighted_aux_ade + weighted_aux_fde + weighted_selector + weighted_center
        batch_size_valid = int(valid_mask.sum().item())
        self.log('train_fm_loss', fm_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size_valid)
        if self.residual_fm:
            original_valid = original_residual[valid_mask]
            self.log('train_center_loss', center_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_center_weighted', weighted_center, prog_bar=False, on_step=True, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_center_rmse', center_rmse, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_center_explained_energy', center_explained_energy, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_original_residual_rmse', original_valid.pow(2).mean().clamp_min(1e-12).sqrt(), prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_residual_target_rmse', z_fm_target_valid.pow(2).mean().clamp_min(1e-12).sqrt(), prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_residual_noise_rmse', x_0_valid.pow(2).mean().clamp_min(1e-12).sqrt(), prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_residual_source_scale', torch.as_tensor(self.residual_source_scale, device=self.device), prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_selector_loss', selector_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_prototype_acc', selector_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_prototype_top6', selector_topk, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_selector_ce', selector_loss_dict['ce'], prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_selector_topk_rank', selector_loss_dict['topk_rank'], prog_bar=False, on_step=True, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_selector_weighted_ce', selector_loss_dict['weighted_ce'], prog_bar=False, on_step=True, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_selector_weighted_rank', selector_loss_dict['weighted_rank'], prog_bar=False, on_step=True, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_selector_logits_std', selector_logits_std, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_cfg_scene_drop_ratio', cfg_drop_mask[valid_mask].float().mean(), prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size_valid)
            boundary_valid = boundary_mask & valid_mask
            boundary_rate = boundary_valid.float().sum() / valid_mask.float().sum().clamp_min(1.0)
            secondary_rate = ((prototype_index == selector_support_ids[:, 1]) & boundary_valid).float().sum() / boundary_valid.float().sum().clamp_min(1.0) if selector_support_ids.size(1) > 1 else boundary_rate * 0.0
            self.log('train_boundary_agent_rate', boundary_rate, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_valid)
            self.log('train_boundary_secondary_sample_rate', secondary_rate, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size_valid)
        self.log('train_decoder_center_ADE_m', aux_ade_m, prog_bar=True, on_step=True, on_epoch=True, batch_size=max(num_aux_agents, 1))
        self.log('train_decoder_center_FDE_m', aux_fde_m, prog_bar=True, on_step=True, on_epoch=True, batch_size=max(num_aux_agents, 1))
        self.log('train_selector_weighted', weighted_selector, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_valid)
        self.log('train_total_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size_valid)
        return loss

    def _training_step_vae(self, data):
        """Stage 0: train latent-space VAE on ground-truth trajectories.

        Accepts either:
        - dict with 'target' [N_a, T_f, D] and 'predict_mask' [N_a, T_f] (lightweight VAE loader)
        - HeteroData with 'agent' key (legacy full DataLoader path, kept for validation)
        """
        if isinstance(data, dict):
            target = data['target']
            predict_mask = data['predict_mask']
        else:
            target = data['agent']['target'][..., :self.output_dim]
            target = target / 10.0
            predict_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
            current_valid_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1]
            predict_mask = predict_mask.clone()
            predict_mask[~current_valid_mask] = False
        #print(f"Train_Target_Mean: {target.mean()}")
        recon_x, mu, logvar = self.latent_encoder(target, predict_mask=predict_mask)
        beta_now = self._get_current_vae_beta()
        loss, loss_dict = self.vae_loss(recon_x, mu, logvar, target, mask=predict_mask, beta=beta_now)
        self.log('train_vae_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_vae_recon', loss_dict['loss_recon'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_vae_kl', loss_dict['loss_kl'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_vae_beta', loss_dict['vae_beta'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_vae_ortho', loss_dict['ortho_aux'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def _training_step_scorer(self, data):
        """Stage 2: train only the trajectory scorer."""
        target = data['agent']['target'][..., :self.output_dim] / float(self.trajectory_scale)
        predict_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        self.encoder.eval()
        self.fm_decoder.eval()
        with torch.no_grad():
            scene_enc = self.encoder(data)
            _, scene_conditions = self.fm_decoder.build_scene_conditions(data, scene_enc)
            if self.residual_fm:
                proto_indices, proto_modes, proto_std_modes, proto_pi = self._select_prototypes_for_sampling(
                    data=data,
                    scene_enc=scene_enc,
                    num_modes=self.num_modes,
                    scene_conditions=scene_conditions,
                )
                trajectories, _ = self.fm_decoder.sample(
                    data,
                    scene_enc,
                    num_modes=self.num_modes,
                    num_steps=self.fm_num_steps,
                    latent_decoder=self.latent_decoder,
                    prototype_latents=proto_modes,
                    prototype_local_std=proto_std_modes,
                    prototype_pi=proto_pi,
                    prototype_indices=proto_indices,
                    scene_conditions=scene_conditions,
                    cfg_guidance_scale=self.cfg_guidance_scale,
                    residual_source_scale=self.residual_source_scale,
                )
            else:
                trajectories, _ = self.fm_decoder.sample(
                    data,
                    scene_enc,
                    num_modes=self.num_modes,
                    num_steps=self.fm_num_steps,
                    latent_decoder=self.latent_decoder,
                    latent_std=self.z_std,
                    scene_conditions=scene_conditions,
                )
        self.train()
        agent_context = scene_enc['x_a'][:, -1, :]
        scorer_loss = self.fm_decoder.compute_scorer_loss(trajectories, target, predict_mask, agent_context)
        self.log('train_scorer_loss', scorer_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        return scorer_loss

    def validation_step(self, data, batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        if self.vae_only:
            target = data['agent']['target'][..., :self.output_dim]
            target = target / float(self.trajectory_scale)
            predict_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
            current_valid_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1]
            predict_mask = predict_mask.clone()
            predict_mask[~current_valid_mask] = False
            with torch.no_grad():
                recon_x, mu, logvar = self.latent_encoder(target, predict_mask=predict_mask)
                beta_now = self._get_current_vae_beta()
                loss, loss_dict = self.vae_loss(recon_x, mu, logvar, target, mask=predict_mask, beta=beta_now)
            self.log('val_vae_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_vae_recon', loss_dict['loss_recon'], prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_vae_kl', loss_dict['loss_kl'], prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_vae_beta', loss_dict['vae_beta'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
            self.log('val_vae_ortho', loss_dict['ortho_aux'], prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            return
        target = data['agent']['target'][..., :self.output_dim]
        target = target / float(self.trajectory_scale)
        predict_mask = data['agent']['predict_mask'][:, self.num_historical_steps:].bool()
        agent_batch = data['agent'].get('batch', None)
        with torch.no_grad():
            scene_enc = self.encoder(data)
            _, scene_conditions = self.fm_decoder.build_scene_conditions(data, scene_enc)
            prototype_index = None
            prototype_latent = None
            prototype_local_std = None
            selector_support_ids = None
            selector_support_weights = None
            residual_center = None
            original_residual = None
            boundary_mask = torch.zeros(target.size(0), device=self.device, dtype=torch.bool)
            if self.residual_fm:
                (
                    z_target,
                    original_residual,
                    prototype_index,
                    prototype_latent,
                    prototype_local_std,
                    valid_mask,
                    selector_support_ids,
                    selector_support_weights,
                    boundary_mask,
                ) = self._get_cached_residual_targets(data, predict_mask, sample_boundary_support=False)
                residual_center = self.fm_decoder.predict_residual_center(
                    scene_conditions=scene_conditions,
                    prototype_latent=prototype_latent,
                    prototype_index=prototype_index,
                )
                z_fm_target = original_residual - residual_center
                x_0, t = self._sample_residual_noise(z_fm_target, prototype_local_std, agent_batch)
            else:
                z_target = self.latent_encoder.encode(target, predict_mask=predict_mask) - self.z_mean
                z_fm_target = z_target
                valid_mask = predict_mask.any(dim=-1)
                x_0, t = FlowMatchingLoss.sample_noise_and_time_latent(
                    self.vae_num_intents, target.size(0), self.latent_dim, self.device, agent_batch
                )
                x_0 = x_0 * self.z_std.to(device=x_0.device, dtype=x_0.dtype)
            t_exp = t[:, None, None]
            x_t = (1.0 - t_exp) * x_0 + t_exp * z_fm_target
            v_theta = self(
                data=data,
                scene_enc=scene_enc,
                x_t=x_t,
                t=t,
                prototype_latent=prototype_latent,
                prototype_index=prototype_index,
                scene_conditions=scene_conditions,
            )
        if not valid_mask.any():
            return v_theta.sum() * 0.0
        v_theta_valid = v_theta[valid_mask]
        z_fm_target_valid = z_fm_target[valid_mask]
        x_0_valid = x_0[valid_mask]
        fm_loss, loss_dict = self.latent_fm_loss(v_theta_valid, z_fm_target_valid, x_0_valid)
        if self.residual_fm:
            center_loss, center_rmse, center_explained_energy = self._compute_residual_center_loss(
                center_pred=residual_center,
                center_target=original_residual,
                valid_mask=valid_mask,
            )
        else:
            center_loss = v_theta.sum() * 0.0
            center_rmse = center_loss.detach()
            center_explained_energy = center_loss.detach()
        selector_loss = v_theta.sum() * 0.0
        zero = selector_loss.detach()
        selector_acc = zero
        selector_topk = zero
        selector_logits_std = zero
        selector_loss_dict = {key: zero for key in ("ce", "topk_rank", "weighted_ce", "weighted_rank")}
        if self.residual_fm and self.prototype_selector is not None:
            selector_label_mask=valid_mask&selector_support_weights.sum(dim=1).gt(0)
            if selector_label_mask.any():
                selector_outputs=self._compute_selector_outputs(scene_conditions=scene_conditions,agent_mask=selector_label_mask)
                selector_logits=selector_outputs["logits"]; support_ids_now=selector_support_ids[selector_label_mask]; support_weights_now=selector_support_weights[selector_label_mask]
                selector_loss,selector_loss_dict=self._compute_selector_loss(selector_outputs=selector_outputs,support_ids=support_ids_now,support_weights=support_weights_now)
                positive=support_weights_now.gt(0); predicted=selector_logits.argmax(dim=-1)
                selector_acc=((predicted[:,None]==support_ids_now)&positive).any(dim=1).float().mean()
                selector_logits_std=selector_outputs["logits_std"].detach().float(); topk=min(self.num_modes,selector_logits.size(-1))
                top_indices=selector_logits.topk(topk,dim=-1).indices
                selector_topk=((top_indices[:,:,None]==support_ids_now[:,None,:])&positive[:,None,:]).any(dim=(1,2)).float().mean()
        aux_ade_m, aux_fde_m, aux_latent_rmse, num_aux_agents = (
            self._decoder_aware_trajectory_loss(
                v_theta=v_theta,
                x_t=x_t,
                t=t,
                z_target=z_target,
                predict_mask=predict_mask,
                category=data["agent"]["category"],
                prototype_latent=prototype_latent,
                residual_center=residual_center,
            )
        )
        batch_size_valid = int(valid_mask.sum().item())
        self.log('val_fm_loss', fm_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_valid, sync_dist=True)
        if self.residual_fm:
            self.log("val_center_loss", center_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_valid, sync_dist=True)
            self.log("val_center_rmse", center_rmse, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_valid, sync_dist=True)
            self.log("val_center_explained_energy", center_explained_energy, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_valid, sync_dist=True)
            self.log("val_original_residual_rmse", original_residual[valid_mask].pow(2).mean().clamp_min(1e-12).sqrt(), prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size_valid, sync_dist=True)
            self.log("val_selector_loss", selector_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_valid, sync_dist=True)
            self.log("val_prototype_acc", selector_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_valid, sync_dist=True)
            self.log("val_prototype_top6", selector_topk, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_valid, sync_dist=True)
            self.log("val_selector_ce", selector_loss_dict["ce"], prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size_valid, sync_dist=True)
            self.log("val_selector_topk_rank", selector_loss_dict["topk_rank"], prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size_valid, sync_dist=True)
            self.log("val_selector_weighted_ce", selector_loss_dict["weighted_ce"], prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size_valid, sync_dist=True)
            self.log("val_selector_weighted_rank", selector_loss_dict["weighted_rank"], prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size_valid, sync_dist=True)
            self.log("val_selector_logits_std", selector_logits_std, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size_valid, sync_dist=True)
            self.log("val_residual_target_rmse", z_fm_target_valid.pow(2).mean().clamp_min(1e-12).sqrt(), prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_valid, sync_dist=True)
            self.log("val_residual_noise_rmse",x_0_valid.pow(2).mean().clamp_min(1e-12).sqrt(),prog_bar=False,on_step=False,on_epoch=True,batch_size=batch_size_valid,sync_dist=True)
            boundary_rate=(boundary_mask&valid_mask).float().sum()/valid_mask.float().sum().clamp_min(1.0)
            self.log("val_boundary_agent_rate",boundary_rate,prog_bar=True,on_step=False,on_epoch=True,batch_size=batch_size_valid,sync_dist=True)
        self.log("val_decoder_center_ADE_m", aux_ade_m, prog_bar=True, on_step=False, on_epoch=True, batch_size=max(num_aux_agents, 1), sync_dist=True)
        self.log("val_decoder_center_FDE_m", aux_fde_m, prog_bar=True, on_step=False, on_epoch=True, batch_size=max(num_aux_agents, 1), sync_dist=True)
        if not self.scorer_only:
            if (self.current_epoch + 1) % 3 != 0:
                return
        if self.dataset == 'argoverse_v2':
            eval_mask = (data['agent']['category'] == 3) & predict_mask.any(dim=-1)
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        if eval_mask.any():
            with torch.no_grad():
                if self.residual_fm:
                    proto_indices, proto_modes, proto_std_modes, proto_pi = self._select_prototypes_for_sampling(
                        data=data,
                        scene_enc=scene_enc,
                        num_modes=self.num_modes,
                        scene_conditions=scene_conditions,
                    )
                    traj_samples, pi = self.fm_decoder.sample(
                        data,
                        scene_enc,
                        num_modes=self.num_modes,
                        num_steps=self.fm_num_steps,
                        latent_decoder=self.latent_decoder,
                        prototype_latents=proto_modes,
                        prototype_local_std=proto_std_modes,
                        prototype_pi=proto_pi,
                        prototype_indices=proto_indices,
                        scene_conditions=scene_conditions,
                        cfg_guidance_scale=self.cfg_guidance_scale,
                        residual_source_scale=self.residual_source_scale,
                    )
                else:
                    traj_samples, pi = self.fm_decoder.sample(
                        data,
                        scene_enc,
                        num_modes=self.num_modes,
                        num_steps=self.fm_num_steps,
                        latent_decoder=self.latent_decoder,
                        latent_std=self.z_std,
                        scene_conditions=scene_conditions,
                    )
                agent_context = scene_enc['x_a'][:, -1, :]
                scorer_loss = self.fm_decoder.compute_scorer_loss( traj_samples, target, predict_mask, agent_context )
                traj_samples = traj_samples * float(self.trajectory_scale)
                target_m = target * float(self.trajectory_scale)
            self.log('val_scorer_loss', scorer_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            traj_eval = traj_samples[eval_mask]
            pi_eval = pi[eval_mask]
            valid_mask_eval = predict_mask[eval_mask]
            gt_eval = target_m[eval_mask]
            self.minADE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
            self.minFDE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
            self.MR.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
            self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

    def on_validation_epoch_start(self):
        self.minADE.reset()
        self.minFDE.reset()
        self.MR.reset()

    def on_train_epoch_end(self):
        gc.collect()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        gc.collect()
        torch.cuda.empty_cache()

    def test_step(self, data, batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        scene_enc = self.encoder(data)
        _, scene_conditions = self.fm_decoder.build_scene_conditions(data, scene_enc)
        if self.residual_fm:
            proto_indices, proto_modes, proto_std_modes, proto_pi = self._select_prototypes_for_sampling(
                data=data,
                scene_enc=scene_enc,
                num_modes=self.num_modes,
                scene_conditions=scene_conditions,
            )
            traj_samples, pi = self.fm_decoder.sample(
                data,
                scene_enc,
                num_modes=self.num_modes,
                num_steps=self.fm_num_steps,
                latent_decoder=self.latent_decoder,
                prototype_latents=proto_modes,
                prototype_local_std=proto_std_modes,
                prototype_pi=proto_pi,
                prototype_indices=proto_indices,
                scene_conditions=scene_conditions,
                cfg_guidance_scale=self.cfg_guidance_scale,
                residual_source_scale=self.residual_source_scale,
            )
        else:
            traj_samples, pi = self.fm_decoder.sample(
                data,
                scene_enc,
                num_modes=self.num_modes,
                num_steps=self.fm_num_steps,
                latent_decoder=self.latent_decoder,
                latent_std=self.z_std,
                scene_conditions=scene_conditions,
            )
        # traj_samples: [N_a, num_modes, T_f, D], pi: [N_a, num_modes]
        traj_samples = traj_samples * float(self.trajectory_scale)
        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = traj_samples[eval_mask][..., :2]
        # Rotate and translate to global frame
        traj_eval = torch.matmul(traj_eval, rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        pi_eval = pi[eval_mask]
        traj_eval = traj_eval.cpu().numpy()
        pi_eval = pi_eval.cpu().numpy()
        if self.dataset == 'argoverse_v2':
            eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
            if isinstance(data, Batch):
                for i in range(data.num_graphs):
                    self.test_predictions[data['scenario_id'][i]] = (pi_eval[i], {eval_id[i]: traj_eval[i]})
            else:
                self.test_predictions[data['scenario_id']] = (pi_eval[0], {eval_id[0]: traj_eval[0]})
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def on_test_end(self):
        if self.dataset == 'argoverse_v2':
            ChallengeSubmission(self.test_predictions).to_parquet( Path(self.submission_dir) / f'{self.submission_file_name}.parquet')
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        # 🌟 核心修复：把那些 PyTorch 模型树中因为参数共享产生的“幻影别名”过滤掉
        decay = {p for p in decay if p in param_dict}
        no_decay = {p for p in no_decay if p in param_dict}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0
        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]
        if self.latent_regression_only:
            reg_decay = {p for p in decay if "latent_regressor" in p}
            reg_no_decay = {p for p in no_decay if "latent_regressor" in p}
            optim_groups = [
                {"params": [param_dict[p] for p in sorted(reg_decay)], "weight_decay": self.weight_decay},
                {"params": [param_dict[p] for p in sorted(reg_no_decay)], "weight_decay": 0.0},
            ]
        elif self.vae_only:
            # Stage 0: only optimize latent_encoder parameters
            vae_decay = {p for p in decay if 'latent_encoder' in p}
            vae_no_decay = {p for p in no_decay if 'latent_encoder' in p}
            optim_groups = [
                {"params": [param_dict[p] for p in sorted(vae_decay)],
                 "weight_decay": self.weight_decay},
                {"params": [param_dict[p] for p in sorted(vae_no_decay)],
                 "weight_decay": 0.0},
            ]
        elif self.freeze_vae:
            # Stage 1: train encoder + fm_decoder, freeze latent_encoder
            freeze_decay = {p for p in decay if 'latent_encoder' not in p}
            freeze_no_decay = {p for p in no_decay if 'latent_encoder' not in p}
            optim_groups = [
                {"params": [param_dict[p] for p in sorted(freeze_decay)],
                 "weight_decay": self.weight_decay},
                {"params": [param_dict[p] for p in sorted(freeze_no_decay)],
                 "weight_decay": 0.0},
            ]
        elif self.scorer_only:
            # Stage 2: only optimize scorer parameters, respecting whitelist/blacklist
            scorer_decay = {p for p in decay if 'scorer' in p}
            scorer_no_decay = {p for p in no_decay if 'scorer' in p}
            optim_groups = [
                {"params": [param_dict[p] for p in sorted(scorer_decay)],
                 "weight_decay": self.weight_decay},
                {"params": [param_dict[p] for p in sorted(scorer_no_decay)],
                 "weight_decay": 0.0},
            ]
        # else: default — optim_groups already built above from all params
        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        if self.latent_regression_only:
            warmup_epochs = 1
        elif getattr(self, 'vae_only', False):
            warmup_epochs = 1  # Stage 0 (VAE): 梯度平稳，给 1 个 Epoch 象征性热身即可
        elif getattr(self, 'scorer_only', False):
            warmup_epochs = 1  # Stage 2 (Scorer): 只训练打分器，1 个 Epoch 足够
        else:
            warmup_epochs = 1  # Stage 1 (FM): 核心潜空间流匹配，必须给足 3 个 Epoch 防爆
        # 2. 安全构建调度器 (加入 VAE 与 FM 的调度分流)
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR( optimizer, start_factor=0.4, total_iters=warmup_epochs )
            cosine_scheduler = CosineAnnealingLR( optimizer, T_max=self.T_max - warmup_epochs, eta_min=1e-6 )
            scheduler = SequentialLR( optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs] )
        else:
            # 如果某天你决定把某阶段的 warmup_epochs 设为 0，直接退化为纯余弦
            scheduler = CosineAnnealingLR( optimizer, T_max=self.T_max, eta_min=0.0 )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNetFM')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--latent_dim', type=int, default=16)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--num_historical_steps', type=int, required=True)
        parser.add_argument('--num_future_steps', type=int, required=True)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_dec_layers', type=int, default=1)
        parser.add_argument('--num_selector_layers', type=int, default=1)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--fm_num_steps', type=int, default=3)
        parser.add_argument('--scorer_only', action='store_true', default=False)
        parser.add_argument('--vae_only', action='store_true', default=False)
        parser.add_argument('--freeze_vae', action='store_true', default=False)
        parser.add_argument('--vae_kl_warmup_epochs', type=int, default=10)
        parser.add_argument('--vae_kl_start_beta', type=float,default=0.0)
        parser.add_argument('--vae_beta', type=float, default=0.015)
        parser.add_argument('--vae_gamma', type=float, default=0.1)
        parser.add_argument('--vae_num_intents', type=int, default=1)
        parser.add_argument('--band_weights', nargs='+', type=float, default=None, help='e.g. --band_weights 1.5 0.5 0.5 1.5')
        parser.add_argument('--pl2pl_radius', type=float, required=True)
        parser.add_argument('--time_span', type=int, default=None)
        parser.add_argument('--pl2a_radius', type=float, required=True)
        parser.add_argument('--a2a_radius', type=float, required=True)
        parser.add_argument('--num_t2m_steps', type=int, default=None)
        parser.add_argument('--pl2m_radius', type=float, required=True)
        parser.add_argument('--a2m_radius', type=float, required=True)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--submission_dir', type=str, default='./')
        parser.add_argument('--submission_file_name', type=str, default='submission')
        parser.add_argument("--decoder_aux_ade_weight", type=float, default=0.016) 
        parser.add_argument("--decoder_aux_fde_weight", type=float, default=0.008)
        parser.add_argument("--decoder_aux_warmup_epochs", type=int, default=5)
        parser.add_argument("--decoder_aux_focal_only", action="store_true", dest="decoder_aux_focal_only", help="辅助轨迹损失仅作用于 category==3 且未来有效的 focal agent。")
        parser.add_argument("--decoder_aux_all_valid", action="store_false", dest="decoder_aux_focal_only", help="辅助轨迹损失作用于所有未来有效 agent。")
        parser.set_defaults(decoder_aux_focal_only=True)
        parser.add_argument("--trajectory_scale", type=float, default=10.0)
        parser.add_argument("--num_hist_tokens", type=int, default=3)
        parser.add_argument("--num_map_tokens", type=int, default=4)
        parser.add_argument("--num_agent_tokens", type=int, default=4)
        parser.add_argument("--residual_fm", action="store_true", default=False)
        parser.add_argument("--prototype_bank_path", type=str, default=None)
        parser.add_argument("--use_prototype_local_std", action="store_true", default=True)
        parser.add_argument("--no_use_prototype_local_std", action="store_false", dest="use_prototype_local_std")
        parser.add_argument("--prototype_selector_loss_weight", type=float, default=0.015)
        parser.add_argument("--prototype_std_floor", type=float, default=0.01)
        parser.add_argument("--prototype_sampling_topk", type=int, default=6)
        parser.add_argument("--residual_samples_per_prototype", type=int, default=1)
        parser.add_argument("--selector_hard_ce_weight", type=float, default=1.0, help="prototype selector 交叉熵权重。")
        parser.add_argument("--selector_rank_loss_weight", type=float, default=0.4)
        parser.add_argument("--selector_rank_margin", type=float, default=0.2)
        parser.add_argument("--selector_num_hard_negatives", type=int, default=6)
        parser.add_argument("--selector_use_prototype_self_attention", action="store_true", dest="selector_use_prototype_self_attention", help="在场景交互后启用 prototype 间 self-attention。")
        parser.add_argument("--no_selector_use_prototype_self_attention", action="store_false", dest="selector_use_prototype_self_attention")
        parser.set_defaults(selector_use_prototype_self_attention=True)
        parser.add_argument("--selector_agent_chunk_size", type=int, default=0, help="prototype self-attention 按 agent 分块；0 表示不分块。")
        parser.add_argument("--num_modality_token_refiner_layers", type=int, default=1, help="history/map/agent 三组条件 token 各自独立的组内 self-attention 层数；0 表示关闭。")
        parser.add_argument("--num_scene_token_refiner_layers", type=int, default=0, help="三组条件 token 拼接后的联合 scene self-attention 层数；0 表示关闭。")
        parser.add_argument("--num_center_layers", type=int, default=1, help="确定性 residual center 的 prototype-query cross-attention 层数。")
        parser.add_argument("--residual_center_loss_weight", type=float, default=4.0, help="确定性 residual center 的 MSE 损失权重。")
        parser.add_argument("--residual_source_scale", type=float, default=0.7, help="在原 prototype residual std 基础上的 source 半径系数。")
        parser.add_argument( "--cfg_scene_dropout", type=float, default=0.15, help="训练速度场时随机丢弃全部 scene 条件但保留 prototype 条件的概率。")
        parser.add_argument( "--cfg_guidance_scale", type=float, default=1.0, help="推理时 prototype-conditioned CFG 强度：0=prototype-only，1=普通条件流，>1 放大场景修正。")
        return parent_parser