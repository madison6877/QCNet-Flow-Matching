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
import gc
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from utils import weight_init

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

class DeterministicLatentRegressor(nn.Module):

    def __init__(self, hidden_dim: int, latent_dim: int, num_intents: int,) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.num_intents = num_intents

        self.net = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, num_intents * latent_dim,)
        )

        self.apply(weight_init)

        # 初始输出为标准化 latent 0，也就是 latent 均值。
        last_layer = self.net[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def forward(self, x_m: torch.Tensor) -> torch.Tensor:
        n_agents = x_m.size(0)
        z_pred = self.net(x_m)
        return z_pred.view(n_agents,self.num_intents,self.latent_dim)

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
                 latent_regression_only: bool = False,
                 decoder_aux_ade_weight: float = 0.08,
                 decoder_aux_fde_weight: float = 0.04,
                 decoder_aux_warmup_epochs: int = 5,
                 decoder_aux_focal_only: bool = True,
                 trajectory_scale: float = 10.0,
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
        self.latent_regression_only = latent_regression_only
        self.decoder_aux_ade_weight = decoder_aux_ade_weight
        self.decoder_aux_fde_weight = decoder_aux_fde_weight
        self.decoder_aux_warmup_epochs = decoder_aux_warmup_epochs
        self.decoder_aux_focal_only = decoder_aux_focal_only
        self.trajectory_scale = trajectory_scale

        num_active_special_modes = sum([bool(self.vae_only),bool(self.scorer_only),bool(self.latent_regression_only)])
        if num_active_special_modes > 1:
            raise ValueError("vae_only、scorer_only 和 latent_regression_only ""不能同时启用。")

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
            head_dim=head_dim,
            dropout=dropout,
            vae_num_intents=vae_num_intents,
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

        means =  [-0.011081701144576073, -0.010677292011678219, 0.01709645800292492, 0.0015562275657430291, -0.020165948197245598]
        stds  =  [0.9458926320075989, 0.6536939144134521, 1.17396879196167, 0.5958303809165955, 0.6562486886978149]
        self.register_buffer('z_mean', torch.tensor(means, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer('z_std', torch.tensor(stds, dtype=torch.float32).view(1, 1, -1))

        # 包裹 VAE Decoder 自动反归一化
        class UnnormDecoderWrapper(nn.Module):
            def __init__(self, decoder, mean, std):
                super().__init__()
                self.decoder = decoder
                self.mean = mean
                self.std = std
            def forward(self, z, *args, **kwargs):
                z_unnorm = z + self.mean.to(z.device)
                return self.decoder(z_unnorm, *args, **kwargs)
        self.latent_decoder = UnnormDecoderWrapper(self.latent_decoder, self.z_mean, self.z_std)

        if self.latent_regression_only:
            self.latent_regressor = DeterministicLatentRegressor(hidden_dim=self.hidden_dim,latent_dim=self.latent_dim,num_intents=self.vae_num_intents)
        else:
            self.latent_regressor = None
        if self.freeze_vae:
            for param in self.latent_encoder.parameters():
                param.requires_grad_(False)
            for param in self.latent_decoder.parameters():
                param.requires_grad_(False)
            
    def _training_step_latent_regression_cached(self, batch, batch_idx):
        
        # Lightning 会自动把字典中的 tensor 搬到 GPU
        x_m = batch["x_m"]
        z_target_std = batch["z_target_std"]

        z_pred_std = self.latent_regressor(x_m)
        error = z_pred_std - z_target_std

        # 与当前 FM loss 类似：K 和 latent dim 求和，agent 求平均
        per_agent_loss = error.pow(2).sum(dim=(-1, -2))
        loss = per_agent_loss.mean()

        latent_mse = error.pow(2).mean()
        latent_rmse = latent_mse.clamp_min(1e-12).sqrt()

        self.log("train_latent_reg_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x_m.size(0))
        self.log("train_latent_reg_rmse", latent_rmse, prog_bar=True, on_step=False, on_epoch=True, batch_size=x_m.size(0))

        return loss

    @torch.no_grad()
    def _get_standardized_latent_target(self, target: torch.Tensor, predict_mask: torch.Tensor) -> torch.Tensor:

        self.latent_encoder.eval()

        z_target_raw = self.latent_encoder.encode(target, predict_mask=predict_mask)
        z_target_std = z_target_raw - self.z_mean

        return z_target_std


    def forward(self, data: HeteroData, scene_enc: dict, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v_theta = self.fm_decoder(data, scene_enc, x_t, t)
        return v_theta
    

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
    
    
    def _decoder_aware_trajectory_loss(self, v_theta: torch.Tensor, x_0: torch.Tensor, target: torch.Tensor, 
                                       predict_mask: torch.Tensor, category: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        future_valid_mask = predict_mask.any(dim=-1)

        if self.decoder_aux_focal_only:
            aux_agent_mask = (category == 3) & future_valid_mask
        else:
            aux_agent_mask = future_valid_mask

        num_aux_agents = int(aux_agent_mask.sum().item())

        if num_aux_agents == 0:
            zero = v_theta.sum() * 0.0
            return zero, zero, 0

        # 对线性 FM：target velocity = z_target - x_0，所以 x_0 + v_theta 是预测的 endpoint latent。
        z_pred_std = x_0[aux_agent_mask] + v_theta[aux_agent_mask]

        # decoder 必须为 eval，但不能使用 no_grad，需要让梯度从轨迹误差穿过 decoder 返回 v_theta。
        self.latent_decoder.eval()
        traj_pred = self.latent_decoder(z_pred_std)

        target_aux = target[aux_agent_mask]
        mask_aux = predict_mask[aux_agent_mask].bool()

        # decoder 输出和 target 都处于 /10 的归一化空间。转换成米后计算与正式指标同口径的距离。
        error_m = (traj_pred[..., :self.output_dim] - target_aux[..., :self.output_dim]) * self.trajectory_scale
        distance_m = torch.sqrt(error_m.pow(2).sum(dim=-1) + 1e-8)  # [N_aux, T_future]

        mask_float = mask_aux.to(distance_m.dtype)
        valid_steps = mask_float.sum(dim=-1).clamp_min(1.0)

        # 每个 agent 先对有效时间点平均，再对 agent 平均。
        ade_per_agent = (distance_m * mask_float).sum(dim=-1) / valid_steps
        ade_loss_m = ade_per_agent.mean()

        # 每个 agent 的最后一个有效未来点。
        time_index = torch.arange(mask_aux.size(1), device=mask_aux.device).view(1, -1)
        last_valid_index = time_index.masked_fill(~mask_aux, -1).max(dim=-1).values
        agent_index = torch.arange(num_aux_agents, device=mask_aux.device)
        fde_per_agent = distance_m[agent_index, last_valid_index]
        fde_loss_m = fde_per_agent.mean()

        return ade_loss_m, fde_loss_m, num_aux_agents


    def training_step(self, data, batch_idx):

        if (self.latent_regression_only and isinstance(data, dict) and "x_m" in data):
            return self._training_step_latent_regression_cached(data, batch_idx)

        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        if self.vae_only:
            return self._training_step_vae(data)

        if self.scorer_only:
            return self._training_step_scorer(data)
        
        if self.latent_regression_only:
            return self._training_step_latent_regression(data=data, batch_idx=batch_idx)

        # ---- Stage 1: Latent Flow Matching ----
        target = data['agent']['target'][..., :self.output_dim]
        target = target / 10.0
        predict_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]

        # Encode ground-truth trajectory into 3 latent intent vectors [N_a, 3, H]
        self.latent_encoder.eval()
        with torch.no_grad():
            z_target = self.latent_encoder.encode(target, predict_mask=predict_mask)  # [N_a, 3, H]

        #-----------------------------------------------------------------------
        z_target = z_target - self.z_mean
        #-----------------------------------------------------------------------

        agent_batch = data['agent'].get('batch', None)
        x_0, t = FlowMatchingLoss.sample_noise_and_time_latent(
            self.vae_num_intents, target.size(0), self.latent_dim, self.device, agent_batch)
        x_0 = x_0 * self.z_std
        if self.current_epoch == 0 and batch_idx == 0:
            first_scene_mask = agent_batch == agent_batch[0]
            print("first scene t:")
            print(t[first_scene_mask][:20].detach().cpu())
            print("unique t in first scene:")
            print(torch.unique(t[first_scene_mask]).numel())
        # x_0: [N_a, 3, H], t: [N_a]

        # Linear interpolation in latent space: x_t = t * z_target + (1 - t) * x_0
        t_exp = t[:, None, None]
        x_t = (1 - t_exp) * x_0 + t_exp * z_target

        scene_enc = self.encoder(data)
        v_theta, pinn_loss = self(data, scene_enc, x_t, t)  # [N_a, 3, H]

        # 🌟 修复 1：过滤掉 99% 不需要预测的背景车辆的垃圾梯度
        valid_mask = predict_mask.any(dim=-1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        v_theta_valid = v_theta[valid_mask]
        z_target_valid = z_target[valid_mask]
        x_0_valid = x_0[valid_mask]

        # 只对有效的车辆计算 Flow Matching Loss
        fm_loss, loss_dict = self.latent_fm_loss(v_theta_valid, z_target_valid, x_0_valid)
        fm_loss = fm_loss + 1000.0 * pinn_loss

        aux_ade_m, aux_fde_m, num_aux_agents = self._decoder_aware_trajectory_loss(v_theta=v_theta, x_0=x_0, target=target, 
                                                                                   predict_mask=predict_mask, category=data["agent"]["category"])
        aux_scale = self._get_decoder_aux_scale()
        weighted_aux_ade = aux_scale * self.decoder_aux_ade_weight * aux_ade_m
        weighted_aux_fde = aux_scale * self.decoder_aux_fde_weight * aux_fde_m
        loss = fm_loss + weighted_aux_ade + weighted_aux_fde
        
        self.log('train_fm_loss', fm_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0))
        self.log("train_decoder_aux_ADE_m", aux_ade_m, prog_bar=True, on_step=False, on_epoch=True, batch_size=max(num_aux_agents, 1))
        self.log("train_decoder_aux_FDE_m", aux_fde_m, prog_bar=True, on_step=False, on_epoch=True, batch_size=max(num_aux_agents, 1))
        self.log("train_decoder_aux_weighted", weighted_aux_ade + weighted_aux_fde, prog_bar=False, on_step=False, on_epoch=True, batch_size=max(num_aux_agents, 1))
        self.log("train_decoder_aux_scale", torch.tensor(aux_scale, device=self.device), prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
        self.log("train_total_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=int(valid_mask.sum()))
        #self.log('train_pinn_loss', pinn_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0))
        # for k_name, v_loss in loss_dict.items():
        #     self.log(f'train_{k_name}', v_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0))
        
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
        target = data['agent']['target'][..., :self.output_dim]
        target = target / 10.0
        predict_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]

        self.encoder.eval()
        self.fm_decoder.eval()

        # Compute scene encoding and sample trajectories with frozen encoder/decoder
        with torch.no_grad():
            scene_enc = self.encoder(data)
            trajectories, _ = self.fm_decoder.sample(
                data, scene_enc, num_modes=self.num_modes, num_steps=self.fm_num_steps,
                latent_decoder=self.latent_decoder,
            )
        
        self.train()
        agent_context = scene_enc['x_a'][:, -1, :]  # [N_a, hidden_dim]
        scorer_loss = self.fm_decoder.compute_scorer_loss(
            trajectories, target, predict_mask, agent_context
        )
        self.log('train_scorer_loss', scorer_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        return scorer_loss
    
    def _training_step_latent_regression(self, data: HeteroData, batch_idx: int) -> torch.Tensor:

        target = data["agent"]["target"][..., :self.output_dim] / 10.0
        predict_mask = data["agent"]["predict_mask"][:, self.num_historical_steps:].bool()
        valid_agent_mask = predict_mask.any(dim=-1)

        if not valid_agent_mask.any():
            return torch.zeros((), device=self.device, requires_grad=True)

        self.encoder.eval()
        self.latent_encoder.eval()
        self.latent_decoder.eval()

        with torch.no_grad():
            scene_enc = self.encoder(data)
            z_target_std = self._get_standardized_latent_target(target=target,predict_mask=predict_mask)

        x_m = scene_enc["x_a"][:, -1, :]

        z_pred_std = self.latent_regressor(x_m)
        z_pred_valid = z_pred_std[valid_agent_mask]
        z_target_valid = z_target_std[valid_agent_mask]

        per_agent_se = (z_pred_valid - z_target_valid).pow(2).sum(dim=(-1, -2))
        latent_reg_loss = per_agent_se.mean()
        latent_mse = (z_pred_valid - z_target_valid).pow(2).mean()
        latent_rmse = torch.sqrt(latent_mse.clamp_min(1e-12))

        self.log("train_latent_reg_loss", latent_reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0))
        self.log("train_latent_reg_rmse", latent_rmse, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0))

        return latent_reg_loss
    
    @torch.no_grad()
    def _validation_step_latent_regression(self, data: HeteroData, batch_idx: int) -> None:
        """Validation step for latent regression."""
        target = data["agent"]["target"][..., :self.output_dim] / 10.0

        predict_mask = data["agent"]["predict_mask"][:, self.num_historical_steps:].bool()
        valid_agent_mask = predict_mask.any(dim=-1)

        if not valid_agent_mask.any():
            return

        # Set all modules to evaluation mode
        self.encoder.eval()
        self.latent_encoder.eval()
        self.latent_decoder.eval()
        self.latent_regressor.eval()

        # Encode scene and get standardized latent target
        scene_enc = self.encoder(data)
        z_target_std = self._get_standardized_latent_target(target=target, predict_mask=predict_mask)

        # Predict latent representation
        x_m = scene_enc["x_a"][:, -1, :]
        z_pred_std = self.latent_regressor(x_m)

        # Filter valid agents
        z_pred_valid = z_pred_std[valid_agent_mask]
        z_target_valid = z_target_std[valid_agent_mask]

        # Compute latent regression loss and metrics
        per_agent_se = (z_pred_valid - z_target_valid).pow(2).sum(dim=(-1, -2))
        latent_reg_loss = per_agent_se.mean()

        latent_mse = (z_pred_valid - z_target_valid).pow(2).mean()
        latent_rmse = torch.sqrt(latent_mse.clamp_min(1e-12))

        # Log latent regression metrics
        self.log("val_latent_reg_loss", latent_reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0), sync_dist=True)
        self.log("val_latent_reg_rmse", latent_rmse, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0), sync_dist=True)

        # Decode trajectories from predicted latent (wrapper handles inverse standardization)
        traj_pred = self.latent_decoder(z_pred_std)  # Shape: [N_a, T_f, 2], scale: /10

        # Determine evaluation mask based on dataset
        if self.dataset == "argoverse_v2":
            eval_mask = (data["agent"]["category"] == 3) & predict_mask.any(dim=-1)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        if not eval_mask.any():
            return

        # Convert back to meters for evaluation
        traj_eval = (traj_pred[eval_mask] * 10.0).unsqueeze(1)  # [N_eval, 1, T_f, 2]
        gt_eval = target[eval_mask] * 10.0
        valid_mask_eval = predict_mask[eval_mask]

        # Single trajectory with probability 1
        pi_eval = torch.ones(traj_eval.size(0), 1, device=traj_eval.device, dtype=traj_eval.dtype)

        # Update metrics
        self.minADE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.minFDE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.MR.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)

        # Log evaluation metrics
        self.log("val_reg_ADE1", self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0), sync_dist=True)
        self.log("val_reg_FDE1", self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0), sync_dist=True)
        self.log("val_reg_MR1", self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0), sync_dist=True)

    @torch.no_grad()
    def _validation_step_latent_regression_cached(self, batch, batch_idx):
        """Validation step for latent regression using cached data."""
        x_m = batch["x_m"]
        z_target_std = batch["z_target_std"]
        target = batch["target"]
        predict_mask = batch["predict_mask"].bool()
        eval_mask = batch["eval_mask"].bool()

        z_pred_std = self.latent_regressor(x_m)
        error = z_pred_std - z_target_std

        latent_loss = error.pow(2).sum(dim=(-1, -2)).mean()
        latent_rmse = error.pow(2).mean().clamp_min(1e-12).sqrt()

        self.log("val_latent_reg_loss", latent_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x_m.size(0), sync_dist=True)
        self.log("val_latent_reg_rmse", latent_rmse, prog_bar=True, on_step=False, on_epoch=True, batch_size=x_m.size(0), sync_dist=True)

        # latent_decoder 接收标准化 latent，wrapper 内部会自动反标准化
        trajectory = self.latent_decoder(z_pred_std)

        if not eval_mask.any():
            return

        pred_eval = (trajectory[eval_mask] * 10.0).unsqueeze(1)  # [N_eval, 1, 60, 2]
        target_eval = target[eval_mask] * 10.0
        mask_eval = predict_mask[eval_mask]

        probability = torch.ones(pred_eval.size(0), 1, device=pred_eval.device, dtype=pred_eval.dtype)

        self.minADE.update(pred=pred_eval, target=target_eval, prob=probability, valid_mask=mask_eval)
        self.minFDE.update(pred=pred_eval, target=target_eval, prob=probability, valid_mask=mask_eval)
        self.MR.update(pred=pred_eval, target=target_eval, prob=probability, valid_mask=mask_eval)

        self.log("val_reg_ADE1", self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=target_eval.size(0), sync_dist=True)
        self.log("val_reg_FDE1", self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=target_eval.size(0), sync_dist=True)
        self.log("val_reg_MR1", self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=target_eval.size(0), sync_dist=True)
    
    def validation_step(self, data, batch_idx):

        if (self.latent_regression_only and isinstance(data, dict) and "x_m" in data):
            return self._validation_step_latent_regression_cached(data,batch_idx)

        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        if self.vae_only:
            target = data['agent']['target'][..., :self.output_dim]
            target = target / 10.0
            predict_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
            current_valid_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1]
            predict_mask = predict_mask.clone()
            predict_mask[~current_valid_mask] = False
            #print(f"Val_Target_Mean: {target.mean()}")

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
        
        if self.latent_regression_only:
            return self._validation_step_latent_regression(data=data, batch_idx=batch_idx)

        target = data['agent']['target'][..., :self.output_dim]
        target = target / 10.0
        predict_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]

        agent_batch = data['agent'].get('batch', None)
        with torch.no_grad():
            scene_enc = self.encoder(data)

            # Encode target to latent for validation loss computation
            z_target = self.latent_encoder.encode(target, predict_mask=predict_mask)  # [N_a, 3, H]
            #-----------------------------------------------------------------------
            z_target = z_target - self.z_mean
            #-----------------------------------------------------------------------

            x_0, t = FlowMatchingLoss.sample_noise_and_time_latent(
                self.vae_num_intents, target.size(0), self.latent_dim, self.device, agent_batch)
            x_0 = x_0 * self.z_std
            t_exp = t[:, None, None]
            x_t = (1 - t_exp) * x_0 + t_exp * z_target
            v_theta, pinn_loss = self(data, scene_enc, x_t, t)

        # 🌟 修复 1：过滤掉 99% 不需要预测的背景车辆的垃圾梯度
        valid_mask = predict_mask.any(dim=-1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        v_theta_valid = v_theta[valid_mask]
        z_target_valid = z_target[valid_mask]
        x_0_valid = x_0[valid_mask]

        # 只对有效的车辆计算 Flow Matching Loss
        fm_loss, loss_dict = self.latent_fm_loss(v_theta_valid, z_target_valid, x_0_valid)
        fm_loss = fm_loss + 1000.0 * pinn_loss

        aux_ade_m, aux_fde_m, num_aux_agents = self._decoder_aware_trajectory_loss(v_theta=v_theta,x_0=x_0,target=target,
                                                                                   predict_mask=predict_mask,category=data["agent"]["category"])
        
        self.log('val_fm_loss', fm_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0), sync_dist=True)
        self.log("val_decoder_aux_ADE_m", aux_ade_m, prog_bar=True, on_step=False, on_epoch=True, batch_size=max(num_aux_agents, 1), sync_dist=True,)
        self.log("val_decoder_aux_FDE_m", aux_fde_m, prog_bar=True, on_step=False, on_epoch=True, batch_size=max(num_aux_agents, 1), sync_dist=True)
        #self.log('val_pinn_loss', pinn_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0), sync_dist=True)
        # for k_name, v_loss in loss_dict.items():
        #     self.log(f'val_{k_name}', v_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0), sync_dist=True)

        # Stage 1: stop here, only evaluate FM velocity field loss
        if not self.scorer_only :
            if (self.current_epoch + 1) % 5 != 0:
                return

        # Stage 2: additionally evaluate scorer loss and trajectory prediction metrics
        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        if eval_mask.any():
            with torch.no_grad():
                traj_samples, pi = self.fm_decoder.sample(
                    data, scene_enc, num_modes=self.num_modes, num_steps=self.fm_num_steps,
                    latent_decoder=self.latent_decoder,
                )
                # traj_samples: [N_a, num_modes, T_f, D], pi: [N_a, num_modes]

                agent_context = scene_enc['x_a'][:, -1, :]
                scorer_loss = self.fm_decoder.compute_scorer_loss(
                    traj_samples, target, predict_mask, agent_context
                )

                traj_samples = traj_samples * 10.0
                target = target * 10.0

            self.log('val_scorer_loss', scorer_loss, prog_bar=True, on_step=False, on_epoch=True,
                     batch_size=1, sync_dist=True)

            traj_eval = traj_samples[eval_mask]
            pi_eval = pi[eval_mask]
            valid_mask_eval = predict_mask[eval_mask]
            gt_eval = target[eval_mask]

            self.minADE.update(
                pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval
            )
            self.minFDE.update(
                pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval
            )
            self.MR.update(
                pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval
            )

            self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True,
                     batch_size=gt_eval.size(0))
            self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True,
                     batch_size=gt_eval.size(0))
            self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True,
                     batch_size=gt_eval.size(0))
            
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
        traj_samples, pi = self.fm_decoder.sample(
            data, scene_enc, num_modes=self.num_modes, num_steps=self.fm_num_steps,
            latent_decoder=self.latent_decoder,
        )
        # traj_samples: [N_a, num_modes, T_f, D], pi: [N_a, num_modes]
        traj_samples = traj_samples * 10.0

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
            ChallengeSubmission(self.test_predictions).to_parquet(
                Path(self.submission_dir) / f'{self.submission_file_name}.parquet')
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
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
            warmup_scheduler = LinearLR(
                optimizer, 
                start_factor=0.3, 
                total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=self.T_max - warmup_epochs, 
                eta_min=1e-6
            )
            scheduler = SequentialLR(
                optimizer, 
                schedulers=[warmup_scheduler, cosine_scheduler], 
                milestones=[warmup_epochs]
            )
        else:
            # 如果某天你决定把某阶段的 warmup_epochs 设为 0，直接退化为纯余弦
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=self.T_max, 
                eta_min=0.0
            )
            
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
        parser.add_argument('--num_dec_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--fm_num_steps', type=int, default=10)
        parser.add_argument('--scorer_only', action='store_true', default=False)
        parser.add_argument('--vae_only', action='store_true', default=False)
        parser.add_argument('--freeze_vae', action='store_true', default=False)
        parser.add_argument("--latent_regression_only", action="store_true", default=False)
        parser.add_argument('--vae_kl_warmup_epochs', type=int, default=10)
        parser.add_argument('--vae_kl_start_beta', type=float,default=0.0)
        parser.add_argument('--vae_beta', type=float, default=0.01)
        parser.add_argument('--vae_gamma', type=float, default=0.1)
        parser.add_argument('--vae_num_intents', type=int, default=3)
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
        parser.add_argument("--decoder_aux_ade_weight", type=float, default=0.08)
        parser.add_argument("--decoder_aux_fde_weight", type=float, default=0.04)
        parser.add_argument("--decoder_aux_warmup_epochs", type=int, default=5)
        parser.add_argument("--decoder_aux_focal_only", action="store_true", default=False)
        parser.add_argument("--trajectory_scale", type=float, default=10.0)
        return parent_parser



