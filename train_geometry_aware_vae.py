# Copyright (c) 2026.
# Geometry-aware VAE automatic training pipeline:
# A. ordinary VAE warm-up
# B. joint encoder/decoder geometry-aware training with online EMA latent statistics
# C. exact full-train-set latent statistics
# D. frozen-encoder decoder calibration in the final standardized latent coordinates

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.data import Batch, HeteroData

from datamodules import ArgoverseV2DataModule
from predictors import QCNetFM

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("high")


class FiniteDifferenceGeometryLoss(nn.Module):

    def __init__(self, num_directions: int = 8, perturbation: float = 0.05, scale_weight: float = 0.1,
                 output_dim: int = 2, eps: float = 1e-8) -> None:
        super().__init__()
        if num_directions < 2:
            raise ValueError("num_directions must be at least 2.")
        if perturbation <= 0:
            raise ValueError("perturbation must be positive.")
        if scale_weight < 0:
            raise ValueError("scale_weight must be non-negative.")
        self.num_directions = int(num_directions)
        self.perturbation = float(perturbation)
        self.scale_weight = float(scale_weight)
        self.output_dim = int(output_dim)
        self.eps = float(eps)

    def forward(self, decoder_std: nn.Module, z_std: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z_std.ndim != 3:
            raise ValueError(f"z_std must be [B, num_intents, latent_dim], got {tuple(z_std.shape)}.")

        batch_size, num_intents, latent_dim = z_std.shape
        if batch_size == 0:
            zero = z_std.sum() * 0.0
            return {"loss": zero, "direction_loss": zero, "scale_loss": zero, "mean_sensitivity": zero,
                    "min_sensitivity": zero, "max_sensitivity": zero}

        flat_dim = num_intents * latent_dim
        z_flat = z_std.reshape(batch_size, flat_dim)
        directions = torch.randn(batch_size, self.num_directions, flat_dim, device=z_std.device, dtype=z_std.dtype)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        delta = self.perturbation * directions

        z_plus = (z_flat[:, None, :] + delta).reshape(batch_size * self.num_directions, num_intents, latent_dim)
        z_minus = (z_flat[:, None, :] - delta).reshape(batch_size * self.num_directions, num_intents, latent_dim)
        trajectory_plus = decoder_std(z_plus)[..., :self.output_dim]
        trajectory_minus = decoder_std(z_minus)[..., :self.output_dim]

        directional_derivative = (trajectory_plus - trajectory_minus) / (2.0 * self.perturbation)
        directional_derivative = directional_derivative.reshape(batch_size, self.num_directions,
                                                                 directional_derivative.size(-2), self.output_dim)
        directional_sensitivity = directional_derivative.pow(2).mean(dim=(-1, -2))
        log_sensitivity = torch.log(directional_sensitivity.clamp_min(self.eps))

        direction_loss = log_sensitivity.var(dim=1, unbiased=False).mean()
        per_sample_log_scale = log_sensitivity.mean(dim=1)
        scale_loss = per_sample_log_scale.var(unbiased=False)
        total_loss = direction_loss + self.scale_weight * scale_loss

        return {"loss": total_loss, "direction_loss": direction_loss, "scale_loss": scale_loss,
                "mean_sensitivity": directional_sensitivity.mean(), "min_sensitivity": directional_sensitivity.min(),
                "max_sensitivity": directional_sensitivity.max()}


class GeometryAwareQCNetFM(QCNetFM):

    def __init__(self, geometry_stage: str = "joint", pretrain_epochs: int = 12, joint_epochs: int = 28,
                 calibration_epochs: int = 5, geometry_weight: float = 0.01, geometry_warmup_epochs: int = 5,
                 geometry_num_agents: int = 8, geometry_num_directions: int = 8,
                 geometry_perturbation: float = 0.05, geometry_scale_weight: float = 0.1,
                 latent_stats_momentum: float = 0.99, calibration_lr: float = 2e-5, **kwargs) -> None:
        kwargs["vae_only"] = True
        kwargs["freeze_vae"] = False
        kwargs["scorer_only"] = False
        kwargs["latent_regression_only"] = False
        super().__init__(**kwargs)

        if geometry_stage not in {"joint", "calibration"}:
            raise ValueError("geometry_stage must be 'joint' or 'calibration'.")
        if not 0.0 <= latent_stats_momentum < 1.0:
            raise ValueError("latent_stats_momentum must be in [0, 1).")

        self.geometry_stage = geometry_stage
        self.pretrain_epochs = int(pretrain_epochs)
        self.joint_epochs = int(joint_epochs)
        self.calibration_epochs = int(calibration_epochs)
        self.geometry_weight = float(geometry_weight)
        self.geometry_warmup_epochs = int(geometry_warmup_epochs)
        self.geometry_num_agents = int(geometry_num_agents)
        self.latent_stats_momentum = float(latent_stats_momentum)
        self.calibration_lr = float(calibration_lr)
        self.geometry_loss_fn = FiniteDifferenceGeometryLoss(geometry_num_directions, geometry_perturbation,
                                                              geometry_scale_weight, self.output_dim)

        stats_shape = (1, self.vae_num_intents, self.latent_dim)
        if tuple(self.z_mean.shape) != stats_shape:
            self._buffers["z_mean"] = torch.zeros(stats_shape)
            self._buffers["z_std"] = torch.ones(stats_shape)
            self.latent_decoder.mean = self.z_mean
            self.latent_decoder.std = self.z_std
        else:
            self.z_mean.zero_()
            self.z_std.fill_(1.0)

        self.register_buffer("z_second_moment", torch.ones(stats_shape), persistent=False)
        self.z_stats_initialized = False
        self._set_trainable_parameters()

        self.save_hyperparameters({"geometry_stage": geometry_stage, "pretrain_epochs": pretrain_epochs,
                                   "joint_epochs": joint_epochs, "calibration_epochs": calibration_epochs,
                                   "geometry_weight": geometry_weight, "geometry_warmup_epochs": geometry_warmup_epochs,
                                   "geometry_num_agents": geometry_num_agents,
                                   "geometry_num_directions": geometry_num_directions,
                                   "geometry_perturbation": geometry_perturbation,
                                   "geometry_scale_weight": geometry_scale_weight,
                                   "latent_stats_momentum": latent_stats_momentum,
                                   "calibration_lr": calibration_lr})

    def _set_trainable_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        if self.geometry_stage == "joint":
            for parameter in self.latent_encoder.parameters():
                parameter.requires_grad_(True)
            return

        vae = self.latent_encoder.vae
        vae.time_queries_base.requires_grad_(True)
        for module in [vae.z_proj, vae.decoder_blocks, vae.decoder_mlp]:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        checkpoint["geometry_ema_second_moment"] = self.z_second_moment.detach().cpu()
        checkpoint["geometry_stats_initialized"] = bool(self.z_stats_initialized)

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        second = checkpoint.get("geometry_ema_second_moment")
        if second is not None and tuple(second.shape) == tuple(self.z_second_moment.shape):
            self.z_second_moment.copy_(second.to(self.z_second_moment))
        self.z_stats_initialized = bool(checkpoint.get("geometry_stats_initialized", False))

    def _extract_target_mask(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(data, dict):
            return data["target"], data["predict_mask"].bool()

        target = data["agent"]["target"][..., :self.output_dim] / self.trajectory_scale
        predict_mask = data["agent"]["predict_mask"][:, self.num_historical_steps:].bool()
        current_valid_mask = data["agent"]["valid_mask"][:, self.num_historical_steps - 1]
        predict_mask = predict_mask.clone()
        predict_mask[~current_valid_mask] = False
        return target, predict_mask

    def _masked_reconstruction_loss(self, recon_x: torch.Tensor, target: torch.Tensor,
                                    predict_mask: torch.Tensor) -> torch.Tensor:
        raw = F.mse_loss(recon_x, target, reduction="none")
        valid_agent_mask = predict_mask.any(dim=-1)
        if not valid_agent_mask.any():
            return recon_x.sum() * 0.0

        mask_f = predict_mask.unsqueeze(-1).to(raw.dtype)
        per_agent_sum = (raw * mask_f).sum(dim=(1, 2))
        num_valid_coords = (predict_mask.sum(dim=-1) * recon_x.size(-1)).clamp_min(1)
        per_agent_mse = per_agent_sum / num_valid_coords.to(per_agent_sum.dtype)
        return per_agent_mse[valid_agent_mask].mean() * float(self.num_future_steps * self.output_dim)

    @torch.no_grad()
    def _update_online_latent_stats(self, mu_batch: torch.Tensor, valid_agent_mask: torch.Tensor) -> None:
        valid_mu = mu_batch[valid_agent_mask].detach().float()
        latent_sum = torch.zeros(self.vae_num_intents, self.latent_dim, device=mu_batch.device)
        latent_sq_sum = torch.zeros_like(latent_sum)
        count = torch.tensor(float(valid_mu.size(0)), device=mu_batch.device)
        if valid_mu.numel() > 0:
            latent_sum.copy_(valid_mu.sum(dim=0))
            latent_sq_sum.copy_(valid_mu.pow(2).sum(dim=0))

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(latent_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(latent_sq_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        if count.item() <= 0:
            return

        batch_mean = (latent_sum / count).unsqueeze(0)
        batch_second = (latent_sq_sum / count).unsqueeze(0)
        if not self.z_stats_initialized:
            self.z_mean.copy_(batch_mean)
            self.z_second_moment.copy_(batch_second)
            self.z_stats_initialized = True
        else:
            momentum = self.latent_stats_momentum
            self.z_mean.mul_(momentum).add_(batch_mean, alpha=1.0 - momentum)
            self.z_second_moment.mul_(momentum).add_(batch_second, alpha=1.0 - momentum)

        variance = (self.z_second_moment - self.z_mean.pow(2)).clamp_min(1e-4)
        self.z_std.copy_(variance.sqrt())

    def _geometry_scale(self) -> float:
        if self.geometry_stage == "calibration":
            return 1.0
        if self.current_epoch < self.pretrain_epochs:
            return 0.0
        if self.geometry_warmup_epochs <= 0:
            return 1.0
        geometry_epoch = self.current_epoch - self.pretrain_epochs
        return min(1.0, float(geometry_epoch + 1) / float(self.geometry_warmup_epochs))

    def _select_geometry_agents(self, predict_mask: torch.Tensor, random_selection: bool) -> torch.Tensor:
        candidate_index = torch.where(predict_mask.all(dim=-1))[0]
        if candidate_index.numel() == 0:
            return candidate_index
        num_selected = min(self.geometry_num_agents, candidate_index.numel())
        if random_selection:
            order = torch.randperm(candidate_index.numel(), device=candidate_index.device)
            return candidate_index[order[:num_selected]]
        return candidate_index[:num_selected]

    def _geometry_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=False)
        return contextlib.nullcontext()

    def _decode_standardized(self, z_std: torch.Tensor) -> torch.Tensor:
        # Do not call UnnormDecoderWrapper here. Its mean/std attributes point to the
        # tensors that existed during __init__, while z_mean/z_std are updated online.
        z_raw = z_std * self.z_std.to(device=z_std.device, dtype=z_std.dtype) + self.z_mean.to(device=z_std.device, dtype=z_std.dtype)
        return self.latent_decoder.decoder(z_raw)

    def _empty_geometry_result(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = reference.sum() * 0.0
        return {"loss": zero, "direction_loss": zero, "scale_loss": zero,
                "mean_sensitivity": zero, "min_sensitivity": zero, "max_sensitivity": zero}

    def _compute_geometry(self, mu_std: torch.Tensor, predict_mask: torch.Tensor,
                          random_selection: bool) -> Tuple[Dict[str, torch.Tensor], int]:
        selected = self._select_geometry_agents(predict_mask, random_selection)
        if selected.numel() == 0:
            return self._empty_geometry_result(mu_std), 0

        was_training = self.latent_encoder.training
        self.latent_decoder.eval()
        with self._geometry_context():
            result = self.geometry_loss_fn(self._decode_standardized, mu_std[selected].float())
        if was_training and self.geometry_stage == "joint":
            self.latent_encoder.train()
        return result, int(selected.numel())

    def training_step(self, data, batch_idx):
        target, predict_mask = self._extract_target_mask(data)
        valid_agent_mask = predict_mask.any(dim=-1)
        if not valid_agent_mask.any():
            return target.sum() * 0.0

        if self.geometry_stage == "joint":
            self.latent_encoder.train()
            recon_x, mu, logvar = self.latent_encoder(target, predict_mask=predict_mask)
            beta_now = self._get_current_vae_beta()
            base_loss, loss_dict = self.vae_loss(recon_x, mu, logvar, target,
                                                 mask=predict_mask, beta=beta_now)
            mu_batch = mu.transpose(0, 1).contiguous()
            self._update_online_latent_stats(mu_batch, valid_agent_mask)
            mu_std = (mu_batch - self.z_mean.detach()) / (self.z_std.detach() + 1e-6)
        else:
            self.latent_encoder.eval()
            self.latent_decoder.eval()
            with torch.no_grad():
                mu, logvar, _ = self.latent_encoder.vae.encode(target, predict_mask=predict_mask)
            mu_batch = mu.transpose(0, 1).contiguous()
            mu_std = (mu_batch - self.z_mean.detach()) / (self.z_std.detach() + 1e-6)
            recon_x = self._decode_standardized(mu_std)
            base_loss = self._masked_reconstruction_loss(recon_x, target, predict_mask)
            loss_dict = {"loss_recon": base_loss.detach(), "loss_kl": base_loss.detach() * 0.0,
                         "ortho_aux": base_loss.detach() * 0.0, "vae_beta": base_loss.new_tensor(0.0)}

        geometry_scale = self._geometry_scale()
        if geometry_scale > 0.0:
            geometry_result, num_geometry_agents = self._compute_geometry(mu_std, predict_mask, True)
        else:
            geometry_result, num_geometry_agents = self._empty_geometry_result(base_loss), 0

        weighted_geometry = geometry_scale * self.geometry_weight * geometry_result["loss"]
        total_loss = base_loss + weighted_geometry
        num_valid_agents = max(int(valid_agent_mask.sum().item()), 1)
        geo_batch_size = max(num_geometry_agents, 1)

        self.log("train_vae_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=num_valid_agents)
        self.log("train_vae_base_loss", base_loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=num_valid_agents)
        self.log("train_vae_recon", loss_dict["loss_recon"], prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=num_valid_agents)
        self.log("train_vae_kl", loss_dict["loss_kl"], on_step=False, on_epoch=True,
                 batch_size=num_valid_agents)
        self.log("train_vae_beta", loss_dict["vae_beta"], on_step=False, on_epoch=True,
                 batch_size=num_valid_agents)
        self.log("train_geo_scale_factor", total_loss.new_tensor(geometry_scale), on_step=False, on_epoch=True,
                 batch_size=1)
        self.log("train_geo_loss", geometry_result["loss"], prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=geo_batch_size)
        self.log("train_geo_direction", geometry_result["direction_loss"], on_step=False, on_epoch=True,
                 batch_size=geo_batch_size)
        self.log("train_geo_location_scale", geometry_result["scale_loss"], on_step=False, on_epoch=True,
                 batch_size=geo_batch_size)
        self.log("train_geo_mean_sensitivity", geometry_result["mean_sensitivity"], on_step=False, on_epoch=True,
                 batch_size=geo_batch_size)
        self.log("train_geo_min_sensitivity", geometry_result["min_sensitivity"], on_step=False, on_epoch=True,
                 batch_size=geo_batch_size)
        self.log("train_geo_max_sensitivity", geometry_result["max_sensitivity"], on_step=False, on_epoch=True,
                 batch_size=geo_batch_size)
        self.log("train_geo_weighted", weighted_geometry, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=geo_batch_size)
        self.log("train_geo_to_base_ratio", weighted_geometry.detach() / base_loss.detach().clamp_min(1e-8),
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        return total_loss

    @torch.no_grad()
    def validation_step(self, data, batch_idx):
        target, predict_mask = self._extract_target_mask(data)
        valid_agent_mask = predict_mask.any(dim=-1)
        if not valid_agent_mask.any():
            return

        self.latent_encoder.eval()
        self.latent_decoder.eval()
        mu, logvar, _ = self.latent_encoder.vae.encode(target, predict_mask=predict_mask)
        mu_batch = mu.transpose(0, 1).contiguous()
        mu_std = (mu_batch - self.z_mean) / (self.z_std + 1e-6)
        recon_x = self._decode_standardized(mu_std)

        if batch_idx == 0 and self.current_epoch == 0:
            z_raw_check = mu_std * self.z_std + self.z_mean
            inverse_error = (z_raw_check - mu_batch).abs().max()
            print(f"[Validation Check] target_abs_mean={target.abs().mean().item():.6f}, target_abs_max={target.abs().max().item():.6f}, inverse_error={inverse_error.item():.8f}")

        if self.geometry_stage == "joint":
            beta_now = self._get_current_vae_beta()
            base_loss, loss_dict = self.vae_loss(recon_x, mu, logvar, target,
                                                 mask=predict_mask, beta=beta_now)
        else:
            base_loss = self._masked_reconstruction_loss(recon_x, target, predict_mask)
            loss_dict = {"loss_recon": base_loss, "loss_kl": base_loss * 0.0}

        geometry_scale = self._geometry_scale()
        if geometry_scale > 0.0:
            geometry_result, num_geometry_agents = self._compute_geometry(mu_std, predict_mask, False)
        else:
            geometry_result, num_geometry_agents = self._empty_geometry_result(base_loss), 0

        weighted_geometry = geometry_scale * self.geometry_weight * geometry_result["loss"]
        total_loss = base_loss + weighted_geometry
        num_valid_agents = max(int(valid_agent_mask.sum().item()), 1)

        self.log("val_vae_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=num_valid_agents, sync_dist=True)
        self.log("val_vae_base_loss", base_loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=num_valid_agents, sync_dist=True)
        self.log("val_vae_recon", loss_dict["loss_recon"], prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=num_valid_agents, sync_dist=True)
        self.log("val_geo_loss", geometry_result["loss"], prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=max(num_geometry_agents, 1), sync_dist=True)
        self.log("val_geo_weighted", weighted_geometry, on_step=False, on_epoch=True,
                 batch_size=max(num_geometry_agents, 1), sync_dist=True)

    def configure_optimizers(self):
        trainable = [(name, parameter) for name, parameter in self.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters were found.")

        decay = [parameter for _, parameter in trainable if parameter.ndim >= 2]
        no_decay = [parameter for _, parameter in trainable if parameter.ndim < 2]
        groups = []
        if decay:
            groups.append({"params": decay, "weight_decay": self.weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})

        lr = self.calibration_lr if self.geometry_stage == "calibration" else self.lr
        stage_epochs = self.calibration_epochs if self.geometry_stage == "calibration" else self.pretrain_epochs + self.joint_epochs
        optimizer = torch.optim.AdamW(groups, lr=lr)
        if stage_epochs <= 1:
            return optimizer

        warmup_epochs = 1
        warmup = LinearLR(optimizer, start_factor=0.3, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, stage_epochs - warmup_epochs), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        return [optimizer], [scheduler]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatic geometry-aware VAE training pipeline.")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--val_batch_size", type=int, required=True)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_raw_dir", type=str, default=None)
    parser.add_argument("--val_raw_dir", type=str, default=None)
    parser.add_argument("--test_raw_dir", type=str, default=None)
    parser.add_argument("--train_processed_dir", type=str, default=None)
    parser.add_argument("--val_processed_dir", type=str, default=None)
    parser.add_argument("--test_processed_dir", type=str, default=None)
    parser.add_argument("--vae_processed_dir", type=str, required=True)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_dir", type=str, default="./geometry_vae_runs")
    parser.add_argument("--init_ckpt", type=str, default=None)
    parser.add_argument("--skip_prepare_vae_data", action="store_true", default=False)

    parser.add_argument("--pretrain_epochs", type=int, default=12)
    parser.add_argument("--joint_epochs", type=int, default=28)
    parser.add_argument("--calibration_epochs", type=int, default=5)
    parser.add_argument("--geometry_weight", type=float, default=0.01)
    parser.add_argument("--geometry_warmup_epochs", type=int, default=5)
    parser.add_argument("--geometry_num_agents", type=int, default=8)
    parser.add_argument("--geometry_num_directions", type=int, default=8)
    parser.add_argument("--geometry_perturbation", type=float, default=0.05)
    parser.add_argument("--geometry_scale_weight", type=float, default=0.1)
    parser.add_argument("--latent_stats_momentum", type=float, default=0.99)
    parser.add_argument("--calibration_lr", type=float, default=2e-5)
    parser.add_argument(
        "--calibration_save_top_k",
        type=int,
        default=3,
        help=(
            "Number of calibration checkpoints retained independently for "
            "val_vae_loss, val_geo_loss and val_vae_recon."
        ),
    )
    parser.add_argument("--stats_device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stats_log_interval", type=int, default=200)

    QCNetFM.add_model_specific_args(parser)
    args = parser.parse_args()
    args.vae_only = True
    args.freeze_vae = False
    args.scorer_only = False
    args.latent_regression_only = False
    return args


def build_model_kwargs(args: argparse.Namespace, geometry_stage: str) -> Dict:
    cfg = vars(args).copy()
    cfg["geometry_stage"] = geometry_stage
    for key in ["init_ckpt", "skip_prepare_vae_data", "output_dir", "precision", "stats_device",
                "stats_log_interval", "seed", "accelerator", "devices",
                "calibration_save_top_k"]:
        cfg.pop(key, None)
    return cfg


def load_initial_weights(model: nn.Module, checkpoint_path: Optional[str]) -> None:
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Init] loaded {checkpoint_path}")
    print(f"[Init] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def build_trainer(args: argparse.Namespace, stage_name: str, max_epochs: int,
                  checkpoint_callbacks: Sequence[ModelCheckpoint]) -> pl.Trainer:
    logger = TensorBoardLogger(save_dir=args.output_dir, name=stage_name)
    strategy = DDPStrategy(find_unused_parameters=False) if args.devices > 1 else "auto"
    callbacks = list(checkpoint_callbacks)
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    return pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy=strategy,
                      precision=args.precision, max_epochs=max_epochs, accumulate_grad_batches=1,
                      callbacks=callbacks, logger=logger, log_every_n_steps=20)


def move_batch_to_device(data, device: torch.device):
    if isinstance(data, dict):
        return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in data.items()}
    if isinstance(data, (Batch, HeteroData)):
        return data.to(device)
    raise TypeError(f"Unsupported batch type: {type(data)}")


@torch.no_grad()
def compute_exact_latent_stats(model: GeometryAwareQCNetFM, train_loader, device: torch.device,
                               log_interval: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    model.to(device)
    model.eval()
    latent_sum = torch.zeros(model.vae_num_intents, model.latent_dim, dtype=torch.float64)
    latent_sq_sum = torch.zeros_like(latent_sum)
    count = 0

    for batch_idx, data in enumerate(train_loader):
        data = move_batch_to_device(data, device)
        target, predict_mask = model._extract_target_mask(data)
        valid_agent_mask = predict_mask.any(dim=-1)
        if valid_agent_mask.any():
            mu, _, _ = model.latent_encoder.vae.encode(target, predict_mask=predict_mask)
            mu_batch = mu.transpose(0, 1).contiguous()[valid_agent_mask].double().cpu()
            latent_sum += mu_batch.sum(dim=0)
            latent_sq_sum += mu_batch.pow(2).sum(dim=0)
            count += int(mu_batch.size(0))

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            print(f"[Exact Stats] batch={batch_idx + 1}, valid agents={count:,}")

    if count == 0:
        raise RuntimeError("No valid agents were found while computing exact latent statistics.")

    mean = latent_sum / float(count)
    variance = (latent_sq_sum / float(count) - mean.pow(2)).clamp_min(1e-8)
    std = variance.sqrt()
    model.z_mean.copy_(mean.float().unsqueeze(0).to(model.z_mean.device))
    model.z_std.copy_(std.float().unsqueeze(0).to(model.z_std.device))
    model.z_second_moment.copy_((variance + mean.pow(2)).float().unsqueeze(0).to(model.z_second_moment.device))
    model.z_stats_initialized = True
    return mean.float(), std.float(), count


def save_stats_json(path: Path, mean: torch.Tensor, std: torch.Tensor, count: int) -> None:
    payload = {"num_valid_agents": count, "mean": mean.tolist(), "std": std.tolist()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = ArgoverseV2DataModule(**vars(args))
    datamodule.setup(stage="fit")
    if not args.skip_prepare_vae_data:
        print(f"[Data] preparing lightweight VAE data at {args.vae_processed_dir}")
        datamodule.prepare_vae_data()

    train_loader = datamodule.vae_train_dataloader()
    val_loader = datamodule.val_dataloader()

    print("\n========== Stage A+B: warm-up + joint geometry-aware VAE ==========")
    joint_model = GeometryAwareQCNetFM(**build_model_kwargs(args, "joint"))
    load_initial_weights(joint_model, args.init_ckpt)

    joint_dir = output_dir / "joint"
    joint_checkpoint = ModelCheckpoint(dirpath=joint_dir, filename="joint-{epoch:02d}-{val_vae_loss:.4f}",
                                       save_top_k=0, save_last=True, save_weights_only=False)
    joint_trainer = build_trainer(
        args,
        "joint",
        args.pretrain_epochs + args.joint_epochs,
        [joint_checkpoint],
    )
    joint_trainer.fit(joint_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("\n========== Stage C1: exact full-train-set latent statistics ==========")
    stats_device = torch.device(args.stats_device)
    exact_mean, exact_std, count = compute_exact_latent_stats(joint_model, train_loader, stats_device,
                                                               args.stats_log_interval)
    save_stats_json(output_dir / "exact_latent_stats.json", exact_mean, exact_std, count)
    print(f"[Exact Stats] valid agents: {count:,}")
    print(f"[Exact Stats] mean: {exact_mean.tolist()}")
    print(f"[Exact Stats] std:  {exact_std.tolist()}")

    joint_state = {key: value.detach().cpu() for key, value in joint_model.state_dict().items()}
    del joint_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n========== Stage C2: frozen-encoder decoder calibration ==========")
    calibration_model = GeometryAwareQCNetFM(**build_model_kwargs(args, "calibration"))
    calibration_model.load_state_dict(joint_state, strict=True)

    calibration_dir = output_dir / "calibration"
    save_top_k = max(1, int(args.calibration_save_top_k))

    calibration_total_checkpoint = ModelCheckpoint(
        dirpath=calibration_dir / "best_total",
        filename="total-{epoch:02d}-{val_vae_loss:.6f}",
        monitor="val_vae_loss",
        mode="min",
        save_top_k=save_top_k,
        save_last=True,
        save_weights_only=False,
    )
    calibration_geometry_checkpoint = ModelCheckpoint(
        dirpath=calibration_dir / "best_geometry",
        filename="geometry-{epoch:02d}-{val_geo_loss:.6f}",
        monitor="val_geo_loss",
        mode="min",
        save_top_k=save_top_k,
        save_last=False,
        save_weights_only=False,
    )
    calibration_reconstruction_checkpoint = ModelCheckpoint(
        dirpath=calibration_dir / "best_reconstruction",
        filename="reconstruction-{epoch:02d}-{val_vae_recon:.6f}",
        monitor="val_vae_recon",
        mode="min",
        save_top_k=save_top_k,
        save_last=False,
        save_weights_only=False,
    )

    calibration_trainer = build_trainer(
        args,
        "calibration",
        args.calibration_epochs,
        [
            calibration_total_checkpoint,
            calibration_geometry_checkpoint,
            calibration_reconstruction_checkpoint,
        ],
    )
    calibration_trainer.fit(
        calibration_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    best_total_path = calibration_total_checkpoint.best_model_path
    best_geometry_path = calibration_geometry_checkpoint.best_model_path
    best_reconstruction_path = calibration_reconstruction_checkpoint.best_model_path
    last_path = calibration_total_checkpoint.last_model_path

    print("\n========== Calibration checkpoint summary ==========")
    print(f"[Calibration] best total          : {best_total_path or 'not available'}")
    print(f"[Calibration] best geometry       : {best_geometry_path or 'not available'}")
    print(f"[Calibration] best reconstruction : {best_reconstruction_path or 'not available'}")
    print(f"[Calibration] last                : {last_path or 'not available'}")

    aliases = {
        "best_total": (
            best_total_path,
            output_dir / "geometry_aware_vae_best_total.ckpt",
        ),
        "best_geometry": (
            best_geometry_path,
            output_dir / "geometry_aware_vae_best_geometry.ckpt",
        ),
        "best_reconstruction": (
            best_reconstruction_path,
            output_dir / "geometry_aware_vae_best_reconstruction.ckpt",
        ),
        "last": (
            last_path,
            output_dir / "geometry_aware_vae_last.ckpt",
        ),
    }

    for label, (source_path, destination_path) in aliases.items():
        if source_path:
            shutil.copy2(source_path, destination_path)
            print(f"[Calibration] copied {label}: {destination_path}")

    # The formal final model and exported VAE-only weights use the checkpoint
    # with the lowest validation total loss by default. The geometry/reconstruction
    # alternatives remain available for independent reconstruction/alignment tests.
    final_path = output_dir / "geometry_aware_vae_final.ckpt"
    if best_total_path:
        best_checkpoint = torch.load(best_total_path, map_location="cpu")
        calibration_model.load_state_dict(best_checkpoint["state_dict"], strict=True)
        shutil.copy2(best_total_path, final_path)
        print(f"[Final] selected best-total calibration checkpoint: {best_total_path}")
    else:
        calibration_trainer.save_checkpoint(final_path)
        print("[Final] no monitored best-total checkpoint was available; saved current model.")

    weights_path = output_dir / "geometry_aware_vae_weights.pt"
    torch.save(
        {
            "latent_encoder": calibration_model.latent_encoder.state_dict(),
            "z_mean": calibration_model.z_mean.detach().cpu(),
            "z_std": calibration_model.z_std.detach().cpu(),
            "source_checkpoint": str(final_path.resolve()),
        },
        weights_path,
    )

    def score_to_float(callback: ModelCheckpoint) -> Optional[float]:
        score = callback.best_model_score
        return None if score is None else float(score.detach().cpu().item())

    checkpoint_manifest = {
        "selection_policy": (
            "geometry_aware_vae_final.ckpt and geometry_aware_vae_weights.pt "
            "use the calibration checkpoint with minimum val_vae_loss."
        ),
        "save_top_k_per_metric": save_top_k,
        "best_total": {
            "monitor": "val_vae_loss",
            "score": score_to_float(calibration_total_checkpoint),
            "path": best_total_path,
            "alias": str((output_dir / "geometry_aware_vae_best_total.ckpt").resolve())
            if best_total_path else None,
        },
        "best_geometry": {
            "monitor": "val_geo_loss",
            "score": score_to_float(calibration_geometry_checkpoint),
            "path": best_geometry_path,
            "alias": str((output_dir / "geometry_aware_vae_best_geometry.ckpt").resolve())
            if best_geometry_path else None,
        },
        "best_reconstruction": {
            "monitor": "val_vae_recon",
            "score": score_to_float(calibration_reconstruction_checkpoint),
            "path": best_reconstruction_path,
            "alias": str((output_dir / "geometry_aware_vae_best_reconstruction.ckpt").resolve())
            if best_reconstruction_path else None,
        },
        "last": {
            "path": last_path,
            "alias": str((output_dir / "geometry_aware_vae_last.ckpt").resolve())
            if last_path else None,
        },
        "formal_final_checkpoint": str(final_path.resolve()),
        "vae_only_weights": str(weights_path.resolve()),
        "latent_statistics": str((output_dir / "exact_latent_stats.json").resolve()),
    }
    manifest_path = output_dir / "calibration_checkpoint_manifest.json"
    manifest_path.write_text(
        json.dumps(checkpoint_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[Final] checkpoint: {final_path}")
    print(f"[Final] VAE-only weights: {weights_path}")
    print(f"[Final] latent statistics: {output_dir / 'exact_latent_stats.json'}")
    print(f"[Final] checkpoint manifest: {manifest_path}")
    print("[Next] evaluate best-total, best-geometry, best-reconstruction and last with the same")
    print("       focal-reconstruction and VAE-only alignment settings before choosing the FM VAE.")


if __name__ == "__main__":
    main()