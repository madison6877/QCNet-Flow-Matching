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
                 vae_gamma: float = 3.0,
                 vae_num_intents: int = 3,
                 band_weights: Optional[list] = None,
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
        self.vae_gamma = vae_gamma
        self.vae_num_intents = vae_num_intents
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.submission_dir = submission_dir
        self.submission_file_name = submission_file_name

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

    def forward(self, data: HeteroData, scene_enc: dict, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v_theta = self.fm_decoder(data, scene_enc, x_t, t)
        return v_theta

    # def train_dataloader(self):
    #     """Override default DataLoader for VAE-only training.

    #     When vae_only=True, returns the lightweight VAE DataLoader (tiny .pt files)
    #     instead of the full HeteroData DataLoader.  This bypasses the heavy .pkl
    #     deserialization + TargetBuilder CPU bottleneck.
    #     """
    #     if self.vae_only:
    #         return self.trainer.datamodule.vae_train_dataloader()
    #     return super().train_dataloader()

    def training_step(self, data, batch_idx):

        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        if self.vae_only:
            return self._training_step_vae(data)

        if self.scorer_only:
            return self._training_step_scorer(data)

        # ---- Stage 1: Latent Flow Matching ----
        target = data['agent']['target'][..., :self.output_dim]
        target = target / 10.0
        predict_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]

        # Encode ground-truth trajectory into 3 latent intent vectors [N_a, 3, H]
        self.latent_encoder.eval()
        with torch.no_grad():
            z_target = self.latent_encoder.encode(target, predict_mask=predict_mask)  # [N_a, 3, H]

        agent_batch = data['agent'].get('batch', None)
        x_0, t = FlowMatchingLoss.sample_noise_and_time_latent(
            self.vae_num_intents, target.size(0), self.latent_dim, self.device, agent_batch)
        # x_0: [N_a, 3, H], t: [N_a]

        # Linear interpolation in latent space: x_t = t * z_target + (1 - t) * x_0
        t_exp = t[:, None, None]
        x_t = (1 - t_exp) * x_0 + t_exp * z_target

        scene_enc = self.encoder(data)
        v_theta = self(data, scene_enc, x_t, t)  # [N_a, 3, H]

        loss, loss_dict = self.latent_fm_loss(v_theta, z_target, x_0)
        
        self.log('train_fm_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0))
        for k_name, v_loss in loss_dict.items():
            self.log(f'train_{k_name}', v_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0))
        
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
        loss, loss_dict = self.vae_loss(recon_x, mu, logvar, target, mask=predict_mask)
         

        self.log('train_vae_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_vae_recon', loss_dict['loss_recon'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_vae_kl', loss_dict['loss_kl'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
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

    def validation_step(self, data, batch_idx):
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
                loss, loss_dict = self.vae_loss(recon_x, mu, logvar, target, mask=predict_mask)

            self.log('val_vae_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_vae_recon', loss_dict['loss_recon'], prog_bar=True, on_step=False, on_epoch=True,
                     batch_size=1, sync_dist=True)
            self.log('val_vae_kl', loss_dict['loss_kl'], prog_bar=True, on_step=False, on_epoch=True,
                     batch_size=1, sync_dist=True)
            self.log('val_vae_ortho', loss_dict['ortho_aux'], prog_bar=True, on_step=False, on_epoch=True,
                     batch_size=1, sync_dist=True)
            return

        target = data['agent']['target'][..., :self.output_dim]
        target = target / 10.0
        predict_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]

        agent_batch = data['agent'].get('batch', None)
        with torch.no_grad():
            scene_enc = self.encoder(data)

            # Encode target to latent for validation loss computation
            z_target = self.latent_encoder.encode(target, predict_mask=predict_mask)  # [N_a, 3, H]

            x_0, t = FlowMatchingLoss.sample_noise_and_time_latent(
                self.vae_num_intents, target.size(0), self.hidden_dim, self.device, agent_batch)
            t_exp = t[:, None, None]
            x_t = (1 - t_exp) * x_0 + t_exp * z_target
            v_theta = self(data, scene_enc, x_t, t)

        loss, loss_dict = self.latent_fm_loss(v_theta, z_target, x_0)
        
        self.log('val_fm_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0), sync_dist=True)
        for k_name, v_loss in loss_dict.items():
            self.log(f'val_{k_name}', v_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0), sync_dist=True)

        # Stage 1: stop here, only evaluate FM velocity field loss
        if not self.scorer_only :
            if (self.current_epoch + 1) % 10 != 0:
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

        if self.vae_only:
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
        if getattr(self, 'vae_only', False):
            warmup_epochs = 1  # Stage 0 (VAE): 梯度平稳，给 1 个 Epoch 象征性热身即可
        elif getattr(self, 'scorer_only', False):
            warmup_epochs = 1  # Stage 2 (Scorer): 只训练打分器，1 个 Epoch 足够
        else:
            warmup_epochs = 3  # Stage 1 (FM): 核心潜空间流匹配，必须给足 4 个 Epoch 防爆

        # 2. 安全构建调度器 (加入 VAE 与 FM 的调度分流)
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer, 
                start_factor=0.01, 
                total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=self.T_max - warmup_epochs, 
                eta_min=0.0
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
        return parent_parser



