#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standardized-latent geometry-aware VAE training with a 3-layer decoder and decoder calibration.

Pipeline
--------
A. Ordinary VAE warm-up.
B. Joint encoder/decoder training. Geometry perturbations are applied in the
   standardized posterior-mean latent space using online EMA statistics. The
   global sensitivity anchor constrains E[log sensitivity].
C. Select a joint checkpoint and recompute exact full-train-set latent stats.
D. Freeze the encoder and calibrate decoder-side parameters with exact stats.
E. Export independent best-total, best-reconstruction, best-geometry and last
   calibration checkpoints plus VAE-only weights.

The optional endpoint loss is retained. Set --endpoint_loss_weight 0 to disable it.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

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


class FiniteDifferenceStandardizedGeometryLoss(nn.Module):
    def __init__(self, num_directions: int = 8, perturbation: float = 0.05, scale_weight: float = 0.1,
                 anchor_weight: float = 0.1, output_dim: int = 2, eps: float = 1e-8) -> None:
        super().__init__()
        if num_directions < 2:
            raise ValueError("num_directions must be at least 2.")
        if perturbation <= 0:
            raise ValueError("perturbation must be positive.")
        if scale_weight < 0 or anchor_weight < 0:
            raise ValueError("scale_weight and anchor_weight must be non-negative.")
        self.num_directions = int(num_directions)
        self.perturbation = float(perturbation)
        self.scale_weight = float(scale_weight)
        self.anchor_weight = float(anchor_weight)
        self.output_dim = int(output_dim)
        self.eps = float(eps)

    def forward(self, decoder_std: nn.Module, z_std: torch.Tensor,
                target_log_sensitivity: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if z_std.ndim != 3:
            raise ValueError(f"z_std must be [B, num_intents, latent_dim], got {tuple(z_std.shape)}.")
        batch_size, num_intents, latent_dim = z_std.shape
        if batch_size == 0:
            zero = z_std.sum() * 0.0
            return {"loss": zero, "direction_loss": zero, "scale_loss": zero, "anchor_loss": zero,
                    "mean_sensitivity": zero, "log_mean_sensitivity": zero,
                    "geometric_mean_sensitivity": zero, "mean_log_sensitivity": zero,
                    "target_log_sensitivity": zero, "min_sensitivity": zero,
                    "max_sensitivity": zero, "log_sensitivity_sum": zero, "num_sensitivity": zero}
        flat_dim = num_intents * latent_dim
        z_flat = z_std.reshape(batch_size, flat_dim)
        directions = torch.randn(batch_size, self.num_directions, flat_dim, device=z_std.device, dtype=z_std.dtype)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        delta = self.perturbation * directions
        z_plus = (z_flat[:, None, :] + delta).reshape(batch_size * self.num_directions, num_intents, latent_dim)
        z_minus = (z_flat[:, None, :] - delta).reshape(batch_size * self.num_directions, num_intents, latent_dim)
        trajectory_plus = decoder_std(z_plus)[..., :self.output_dim]
        trajectory_minus = decoder_std(z_minus)[..., :self.output_dim]
        derivative = (trajectory_plus - trajectory_minus) / (2.0 * self.perturbation)
        derivative = derivative.reshape(batch_size, self.num_directions, derivative.size(-2), self.output_dim)
        sensitivity = derivative.pow(2).mean(dim=(-1, -2))
        log_sensitivity = torch.log(sensitivity.clamp_min(self.eps))
        direction_loss = log_sensitivity.var(dim=1, unbiased=False).mean()
        scale_loss = log_sensitivity.mean(dim=1).var(unbiased=False)
        mean_sensitivity = sensitivity.mean()
        log_mean_sensitivity = torch.log(mean_sensitivity.clamp_min(self.eps))
        mean_log_sensitivity = log_sensitivity.mean()
        if target_log_sensitivity is None:
            anchor_loss = mean_log_sensitivity * 0.0
            target_value = mean_log_sensitivity.detach()
        else:
            target_value = target_log_sensitivity.to(device=mean_log_sensitivity.device,
                                                     dtype=mean_log_sensitivity.dtype).detach()
            anchor_loss = (mean_log_sensitivity - target_value).pow(2)
        total_loss = direction_loss + self.scale_weight * scale_loss + self.anchor_weight * anchor_loss
        return {"loss": total_loss, "direction_loss": direction_loss, "scale_loss": scale_loss,
                "anchor_loss": anchor_loss, "mean_sensitivity": mean_sensitivity,
                "log_mean_sensitivity": log_mean_sensitivity,
                "geometric_mean_sensitivity": mean_log_sensitivity.exp(),
                "mean_log_sensitivity": mean_log_sensitivity, "target_log_sensitivity": target_value,
                "min_sensitivity": sensitivity.min(), "max_sensitivity": sensitivity.max(),
                "log_sensitivity_sum": log_sensitivity.sum(),
                "num_sensitivity": sensitivity.new_tensor(float(sensitivity.numel()))}


class StandardizedGeometryAwareQCNetFM(QCNetFM):
    def __init__(self, geometry_stage: str = "joint", pretrain_epochs: int = 12, joint_epochs: int = 28,
                 calibration_epochs: int = 5, geometry_weight: float = 0.01, geometry_warmup_epochs: int = 5,
                 geometry_num_agents: int = 8, geometry_num_directions: int = 8,
                 geometry_perturbation: float = 0.05, geometry_scale_weight: float = 0.1,
                 geometry_anchor_weight: float = 0.1, geometry_anchor_init_batches: int = 1,
                 geometry_detach_latent_center: bool = True,
                 latent_stats_momentum: float = 0.99, calibration_lr: float = 2e-5,
                 endpoint_loss_weight: float = 5.0, validation_geometry_seed: int = 1729, **kwargs) -> None:
        kwargs["vae_only"] = True
        kwargs["freeze_vae"] = False
        kwargs["scorer_only"] = False
        kwargs["latent_regression_only"] = False
        super().__init__(**kwargs)
        if geometry_stage not in {"joint", "calibration"}:
            raise ValueError("geometry_stage must be 'joint' or 'calibration'.")
        if not 0.0 <= latent_stats_momentum < 1.0:
            raise ValueError("latent_stats_momentum must be in [0, 1).")
        if geometry_weight < 0 or geometry_anchor_weight < 0 or endpoint_loss_weight < 0:
            raise ValueError("geometry_weight, geometry_anchor_weight and endpoint_loss_weight must be non-negative.")
        if geometry_anchor_init_batches < 1:
            raise ValueError("geometry_anchor_init_batches must be at least 1.")
        self.geometry_stage = geometry_stage
        self.pretrain_epochs = int(pretrain_epochs)
        self.joint_epochs = int(joint_epochs)
        self.calibration_epochs = int(calibration_epochs)
        self.geometry_weight = float(geometry_weight)
        self.geometry_warmup_epochs = int(geometry_warmup_epochs)
        self.geometry_num_agents = int(geometry_num_agents)
        self.geometry_anchor_weight = float(geometry_anchor_weight)
        self.geometry_anchor_init_batches = int(geometry_anchor_init_batches)
        self.geometry_detach_latent_center = bool(geometry_detach_latent_center)
        self.latent_stats_momentum = float(latent_stats_momentum)
        self.calibration_lr = float(calibration_lr)
        self.endpoint_loss_weight = float(endpoint_loss_weight)
        self.validation_geometry_seed = int(validation_geometry_seed)
        self.geometry_loss_fn = FiniteDifferenceStandardizedGeometryLoss(
            geometry_num_directions, geometry_perturbation, geometry_scale_weight,
            geometry_anchor_weight, self.output_dim)

        stats_shape = (1, self.vae_num_intents, self.latent_dim)
        if tuple(self.z_mean.shape) != stats_shape:
            self._buffers["z_mean"] = torch.zeros(stats_shape)
            self._buffers["z_std"] = torch.ones(stats_shape)
            if hasattr(self.latent_decoder, "mean"):
                self.latent_decoder.mean = self.z_mean
            if hasattr(self.latent_decoder, "std"):
                self.latent_decoder.std = self.z_std
        else:
            self.z_mean.zero_()
            self.z_std.fill_(1.0)
        self.register_buffer("z_second_moment", torch.ones(stats_shape), persistent=False)
        self.register_buffer("geometry_log_sensitivity_target", torch.tensor(0.0))
        self.register_buffer("geometry_anchor_log_sensitivity_sum", torch.tensor(0.0))
        self.register_buffer("geometry_anchor_sample_count", torch.tensor(0.0))
        self.register_buffer("geometry_anchor_batches_seen", torch.tensor(0, dtype=torch.long))
        self.z_stats_initialized = False
        self._set_trainable_parameters()
        self.save_hyperparameters({"geometry_stage": geometry_stage, "pretrain_epochs": pretrain_epochs,
                                   "joint_epochs": joint_epochs, "calibration_epochs": calibration_epochs,
                                   "geometry_weight": geometry_weight, "geometry_warmup_epochs": geometry_warmup_epochs,
                                   "geometry_num_agents": geometry_num_agents,
                                   "geometry_num_directions": geometry_num_directions,
                                   "geometry_perturbation": geometry_perturbation,
                                   "geometry_scale_weight": geometry_scale_weight,
                                   "geometry_anchor_weight": geometry_anchor_weight,
                                   "geometry_anchor_init_batches": geometry_anchor_init_batches,
                                   "geometry_detach_latent_center": geometry_detach_latent_center,
                                   "latent_stats_momentum": latent_stats_momentum,
                                   "calibration_lr": calibration_lr,
                                   "endpoint_loss_weight": endpoint_loss_weight,
                                   "validation_geometry_seed": validation_geometry_seed,
                                   "geometry_space": "standardized",
                                   "geometry_anchor_type": "expected_log_sensitivity",
                                   "uses_decoder_calibration": True})

    def _set_trainable_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if self.geometry_stage == "joint":
            for parameter in self.latent_encoder.parameters():
                parameter.requires_grad_(True)
            return
        vae = self.latent_encoder.vae
        decoder_prefixes = ("time_queries_base", "endpoint_query", "z_proj", "decoder_blocks", "decoder_mlp",
                            "endpoint_head", "displacement_head")
        matched = []
        for name, parameter in vae.named_parameters():
            if name.startswith(decoder_prefixes):
                parameter.requires_grad_(True)
                matched.append(name)
        if not matched:
            raise RuntimeError("No decoder-side VAE parameters were selected for calibration.")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["geometry_ema_second_moment"] = self.z_second_moment.detach().cpu()
        checkpoint["geometry_stats_initialized"] = bool(self.z_stats_initialized)
        checkpoint["geometry_anchor_initialized"] = bool(self._geometry_anchor_initialized())
        checkpoint["geometry_anchor_target_log_sensitivity"] = float(
            self.geometry_log_sensitivity_target.detach().cpu().item())
        checkpoint["geometry_anchor_type"] = "expected_log_sensitivity"

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        second = checkpoint.get("geometry_ema_second_moment")
        if second is not None and tuple(second.shape) == tuple(self.z_second_moment.shape):
            self.z_second_moment.copy_(second.to(self.z_second_moment))
        self.z_stats_initialized = bool(checkpoint.get("geometry_stats_initialized", False))

    def _extract_target_mask(self, data: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(data, dict):
            return data["target"], data["predict_mask"].bool()
        target = data["agent"]["target"][..., :self.output_dim] / self.trajectory_scale
        predict_mask = data["agent"]["predict_mask"][:, self.num_historical_steps:].bool()
        current_valid = data["agent"]["valid_mask"][:, self.num_historical_steps - 1]
        predict_mask = predict_mask.clone()
        predict_mask[~current_valid] = False
        return target, predict_mask

    def _masked_reconstruction_loss(self, recon_x: torch.Tensor, target: torch.Tensor,
                                    predict_mask: torch.Tensor) -> torch.Tensor:
        raw = F.mse_loss(recon_x, target, reduction="none")
        valid_agents = predict_mask.any(dim=-1)
        if not valid_agents.any():
            return recon_x.sum() * 0.0
        mask_f = predict_mask.unsqueeze(-1).to(raw.dtype)
        per_agent_sum = (raw * mask_f).sum(dim=(1, 2))
        valid_coords = (predict_mask.sum(dim=-1) * recon_x.size(-1)).clamp_min(1)
        per_agent_mse = per_agent_sum / valid_coords.to(per_agent_sum.dtype)
        return per_agent_mse[valid_agents].mean() * float(self.num_future_steps * self.output_dim)

    def _endpoint_terms(self, recon_x: torch.Tensor, target: torch.Tensor,
                        predict_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        final_valid = predict_mask[:, -1].bool()
        count = int(final_valid.sum().item())
        if count == 0:
            zero = recon_x.sum() * 0.0
            return zero, zero.detach(), 0
        error = recon_x[final_valid, -1] - target[final_valid, -1]
        loss = error.pow(2).sum(dim=-1).mean()
        fde_m = error.float().norm(dim=-1).mean() * float(self.trajectory_scale)
        return loss, fde_m, count

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

    def _geometry_anchor_initialized(self) -> bool:
        return int(self.geometry_anchor_batches_seen.item()) >= self.geometry_anchor_init_batches

    @torch.no_grad()
    def _update_geometry_anchor(self, log_sensitivity_sum: torch.Tensor,
                                num_sensitivity: torch.Tensor) -> None:
        batch_sum = log_sensitivity_sum.detach().float().reshape(())
        batch_count = num_sensitivity.detach().float().reshape(())
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
        if batch_count.item() <= 0:
            return
        self.geometry_anchor_log_sensitivity_sum.add_(
            batch_sum.to(self.geometry_anchor_log_sensitivity_sum))
        self.geometry_anchor_sample_count.add_(batch_count.to(self.geometry_anchor_sample_count))
        self.geometry_anchor_batches_seen.add_(1)
        if self._geometry_anchor_initialized():
            target_mean_log = (self.geometry_anchor_log_sensitivity_sum /
                               self.geometry_anchor_sample_count.clamp_min(1.0))
            self.geometry_log_sensitivity_target.copy_(
                target_mean_log.to(self.geometry_log_sensitivity_target))

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
        candidates = torch.where(predict_mask.all(dim=-1))[0]
        if candidates.numel() == 0:
            return candidates
        if self.geometry_num_agents <= 0:
            return candidates
        count = min(self.geometry_num_agents, candidates.numel())
        if random_selection:
            return candidates[torch.randperm(candidates.numel(), device=candidates.device)[:count]]
        return candidates[:count]

    def _geometry_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=False)
        return contextlib.nullcontext()

    def _decode_standardized(self, z_std: torch.Tensor) -> torch.Tensor:
        z_raw = z_std * self.z_std.to(device=z_std.device, dtype=z_std.dtype) + self.z_mean.to(device=z_std.device, dtype=z_std.dtype)
        return self.latent_decoder.decoder(z_raw)

    def _empty_geometry_result(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = reference.sum() * 0.0
        return {"loss": zero, "direction_loss": zero, "scale_loss": zero, "anchor_loss": zero,
                "mean_sensitivity": zero, "log_mean_sensitivity": zero,
                "geometric_mean_sensitivity": zero, "mean_log_sensitivity": zero,
                "target_log_sensitivity": self.geometry_log_sensitivity_target.detach().to(reference),
                "anchor_initialized": zero, "min_sensitivity": zero, "max_sensitivity": zero,
                "log_sensitivity_sum": zero, "num_sensitivity": zero}

    def _compute_geometry(self, mu_std: torch.Tensor, predict_mask: torch.Tensor, random_selection: bool,
                          deterministic_seed: Optional[int] = None,
                          update_anchor: bool = False) -> Tuple[Dict[str, torch.Tensor], int]:
        selected = self._select_geometry_agents(predict_mask, random_selection)
        if selected.numel() == 0:
            return self._empty_geometry_result(mu_std), 0
        was_training = self.latent_encoder.training
        self.latent_decoder.eval()
        devices = []
        if mu_std.is_cuda and mu_std.device.index is not None:
            devices = [mu_std.device.index]
        rng_context = torch.random.fork_rng(devices=devices, enabled=deterministic_seed is not None)
        with rng_context:
            if deterministic_seed is not None:
                torch.manual_seed(deterministic_seed)
                if mu_std.is_cuda:
                    torch.cuda.manual_seed_all(deterministic_seed)
            with self._geometry_context():
                anchor_target = (self.geometry_log_sensitivity_target.detach()
                                 if self._geometry_anchor_initialized() else None)
                geometry_center = mu_std[selected]
                if self.geometry_detach_latent_center:
                    geometry_center = geometry_center.detach()
                result = self.geometry_loss_fn(
                    self._decode_standardized, geometry_center.float(), anchor_target)
        if update_anchor and self.geometry_anchor_weight > 0.0 and not self._geometry_anchor_initialized():
            self._update_geometry_anchor(result["log_sensitivity_sum"], result["num_sensitivity"])
        result["target_log_sensitivity"] = self.geometry_log_sensitivity_target.detach().to(
            device=result["mean_log_sensitivity"].device, dtype=result["mean_log_sensitivity"].dtype)
        result["anchor_initialized"] = result["mean_log_sensitivity"].new_tensor(
            float(self._geometry_anchor_initialized()))
        if was_training and self.geometry_stage == "joint":
            self.latent_encoder.train()
        return result, int(selected.numel())

    def training_step(self, data: Any, batch_idx: int):
        target, predict_mask = self._extract_target_mask(data)
        valid_agents = predict_mask.any(dim=-1)
        if not valid_agents.any():
            return target.sum() * 0.0
        if self.geometry_stage == "joint":
            self.latent_encoder.train()
            recon_x, mu, logvar = self.latent_encoder(target, predict_mask=predict_mask)
            beta_now = self._get_current_vae_beta()
            native_loss, loss_dict = self.vae_loss(recon_x, mu, logvar, target, mask=predict_mask, beta=beta_now)
            mu_batch = mu.transpose(0, 1).contiguous()
            self._update_online_latent_stats(mu_batch, valid_agents)
            mu_std = (mu_batch - self.z_mean.detach()) / (self.z_std.detach() + 1e-6)
        else:
            self.latent_encoder.eval()
            self.latent_decoder.eval()
            with torch.no_grad():
                mu, logvar, _ = self.latent_encoder.vae.encode(target, predict_mask=predict_mask)
            mu_batch = mu.transpose(0, 1).contiguous()
            mu_std = (mu_batch - self.z_mean.detach()) / (self.z_std.detach() + 1e-6)
            recon_x = self._decode_standardized(mu_std)
            native_loss = self._masked_reconstruction_loss(recon_x, target, predict_mask)
            loss_dict = {"loss_recon": native_loss.detach(), "loss_kl": native_loss.detach() * 0.0,
                         "ortho_aux": native_loss.detach() * 0.0, "vae_beta": native_loss.new_tensor(0.0)}
        endpoint_loss, endpoint_fde_m, num_endpoint = self._endpoint_terms(recon_x, target, predict_mask)
        weighted_endpoint = self.endpoint_loss_weight * endpoint_loss
        base_loss = native_loss + weighted_endpoint
        geometry_scale = self._geometry_scale()
        if geometry_scale > 0.0:
            geometry_result, num_geometry = self._compute_geometry(
                mu_std, predict_mask, True, update_anchor=True)
        else:
            geometry_result, num_geometry = self._empty_geometry_result(base_loss), 0
        weighted_geometry = geometry_scale * self.geometry_weight * geometry_result["loss"]
        total_loss = base_loss + weighted_geometry
        valid_count = max(int(valid_agents.sum().item()), 1)
        geo_count = max(num_geometry, 1)
        endpoint_count = max(num_endpoint, 1)
        self.log("train_vae_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=valid_count)
        self.log("train_vae_base_loss", base_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=valid_count)
        self.log("train_vae_native_loss", native_loss, on_step=False, on_epoch=True, batch_size=valid_count)
        self.log("train_vae_recon", loss_dict["loss_recon"], prog_bar=True, on_step=False, on_epoch=True, batch_size=valid_count)
        self.log("train_vae_kl", loss_dict["loss_kl"], on_step=False, on_epoch=True, batch_size=valid_count)
        self.log("train_vae_beta", loss_dict["vae_beta"], on_step=False, on_epoch=True, batch_size=valid_count)
        if "ortho_aux" in loss_dict:
            self.log("train_vae_ortho_aux", loss_dict["ortho_aux"], on_step=False, on_epoch=True, batch_size=valid_count)
        self.log("train_vae_endpoint", endpoint_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=endpoint_count)
        self.log("train_vae_endpoint_weighted", weighted_endpoint, on_step=False, on_epoch=True, batch_size=endpoint_count)
        self.log("train_endpoint_FDE_m", endpoint_fde_m, prog_bar=True, on_step=False, on_epoch=True, batch_size=endpoint_count)
        self.log("train_geo_scale_factor", total_loss.new_tensor(geometry_scale), on_step=False, on_epoch=True, batch_size=1)
        self.log("train_geo_loss", geometry_result["loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=geo_count)
        self.log("train_geo_direction", geometry_result["direction_loss"], on_step=False, on_epoch=True, batch_size=geo_count)
        self.log("train_geo_location_scale", geometry_result["scale_loss"], on_step=False, on_epoch=True, batch_size=geo_count)
        self.log("train_geo_anchor", geometry_result["anchor_loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=geo_count)
        self.log("train_geo_mean_sensitivity", geometry_result["mean_sensitivity"], on_step=False, on_epoch=True, batch_size=geo_count)
        self.log("train_geo_log_mean_sensitivity", geometry_result["log_mean_sensitivity"],
                 on_step=False, on_epoch=True, batch_size=geo_count)
        self.log("train_geo_geometric_mean_sensitivity", geometry_result["geometric_mean_sensitivity"],
                 on_step=False, on_epoch=True, batch_size=geo_count)
        self.log("train_geo_mean_log_sensitivity", geometry_result["mean_log_sensitivity"],
                 on_step=False, on_epoch=True, batch_size=geo_count)
        self.log("train_geo_target_log_sensitivity", geometry_result["target_log_sensitivity"],
                 on_step=False, on_epoch=True, batch_size=geo_count)
        self.log("train_geo_anchor_initialized", geometry_result["anchor_initialized"],
                 on_step=False, on_epoch=True, batch_size=1)
        self.log("train_geo_latent_center_detached",
                 total_loss.new_tensor(float(self.geometry_detach_latent_center)),
                 on_step=False, on_epoch=True, batch_size=1)
        self.log("train_geo_min_sensitivity", geometry_result["min_sensitivity"], on_step=False, on_epoch=True, batch_size=geo_count)
        self.log("train_geo_max_sensitivity", geometry_result["max_sensitivity"], on_step=False, on_epoch=True, batch_size=geo_count)
        self.log("train_geo_weighted", weighted_geometry, prog_bar=True, on_step=False, on_epoch=True, batch_size=geo_count)
        self.log("train_geo_to_base_ratio", weighted_geometry.detach() / base_loss.detach().clamp_min(1e-8),
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        return total_loss

    @torch.no_grad()
    def validation_step(self, data: Any, batch_idx: int) -> None:
        target, predict_mask = self._extract_target_mask(data)
        valid_agents = predict_mask.any(dim=-1)
        if not valid_agents.any():
            return
        self.latent_encoder.eval()
        self.latent_decoder.eval()
        mu, logvar, _ = self.latent_encoder.vae.encode(target, predict_mask=predict_mask)
        mu_batch = mu.transpose(0, 1).contiguous()
        mu_std = (mu_batch - self.z_mean) / (self.z_std + 1e-6)
        recon_x = self._decode_standardized(mu_std)
        if self.geometry_stage == "joint":
            beta_now = self._get_current_vae_beta()
            native_loss, loss_dict = self.vae_loss(recon_x, mu, logvar, target, mask=predict_mask, beta=beta_now)
        else:
            native_loss = self._masked_reconstruction_loss(recon_x, target, predict_mask)
            loss_dict = {"loss_recon": native_loss, "loss_kl": native_loss * 0.0,
                         "ortho_aux": native_loss * 0.0, "vae_beta": native_loss.new_tensor(0.0)}
        endpoint_loss, endpoint_fde_m, num_endpoint = self._endpoint_terms(recon_x, target, predict_mask)
        weighted_endpoint = self.endpoint_loss_weight * endpoint_loss
        base_loss = native_loss + weighted_endpoint
        geometry_result, num_geometry = self._compute_geometry(
            mu_std, predict_mask, False, self.validation_geometry_seed + batch_idx)
        geometry_scale = self._geometry_scale()
        weighted_geometry = geometry_scale * self.geometry_weight * geometry_result["loss"]
        total_loss = base_loss + weighted_geometry
        selection_score = base_loss + self.geometry_weight * geometry_result["loss"]
        valid_count = max(int(valid_agents.sum().item()), 1)
        geo_count = max(num_geometry, 1)
        endpoint_count = max(num_endpoint, 1)
        self.log("val_vae_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=valid_count, sync_dist=True)
        self.log("val_selection_score", selection_score, prog_bar=True, on_step=False, on_epoch=True, batch_size=valid_count, sync_dist=True)
        self.log("val_vae_base_loss", base_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=valid_count, sync_dist=True)
        self.log("val_vae_native_loss", native_loss, on_step=False, on_epoch=True, batch_size=valid_count, sync_dist=True)
        self.log("val_vae_recon", loss_dict["loss_recon"], prog_bar=True, on_step=False, on_epoch=True, batch_size=valid_count, sync_dist=True)
        self.log("val_vae_kl", loss_dict["loss_kl"], on_step=False, on_epoch=True, batch_size=valid_count, sync_dist=True)
        self.log("val_vae_endpoint", endpoint_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=endpoint_count, sync_dist=True)
        self.log("val_vae_endpoint_weighted", weighted_endpoint, on_step=False, on_epoch=True, batch_size=endpoint_count, sync_dist=True)
        self.log("val_endpoint_FDE_m", endpoint_fde_m, prog_bar=True, on_step=False, on_epoch=True, batch_size=endpoint_count, sync_dist=True)
        self.log("val_geo_loss", geometry_result["loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=geo_count, sync_dist=True)
        self.log("val_geo_direction", geometry_result["direction_loss"], on_step=False, on_epoch=True, batch_size=geo_count, sync_dist=True)
        self.log("val_geo_location_scale", geometry_result["scale_loss"], on_step=False, on_epoch=True, batch_size=geo_count, sync_dist=True)
        self.log("val_geo_anchor", geometry_result["anchor_loss"], on_step=False, on_epoch=True, batch_size=geo_count, sync_dist=True)
        self.log("val_geo_mean_sensitivity", geometry_result["mean_sensitivity"], on_step=False, on_epoch=True, batch_size=geo_count, sync_dist=True)
        self.log("val_geo_log_mean_sensitivity", geometry_result["log_mean_sensitivity"],
                 on_step=False, on_epoch=True, batch_size=geo_count, sync_dist=True)
        self.log("val_geo_geometric_mean_sensitivity", geometry_result["geometric_mean_sensitivity"],
                 on_step=False, on_epoch=True, batch_size=geo_count, sync_dist=True)
        self.log("val_geo_mean_log_sensitivity", geometry_result["mean_log_sensitivity"],
                 on_step=False, on_epoch=True, batch_size=geo_count, sync_dist=True)
        self.log("val_geo_target_log_sensitivity", geometry_result["target_log_sensitivity"],
                 on_step=False, on_epoch=True, batch_size=geo_count, sync_dist=True)
        self.log("val_geo_anchor_initialized", geometry_result["anchor_initialized"],
                 on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log("val_geo_weighted", weighted_geometry, on_step=False, on_epoch=True, batch_size=geo_count, sync_dist=True)

    def configure_optimizers(self):
        trainable = [(name, p) for name, p in self.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters were found.")
        decay = [p for _, p in trainable if p.ndim >= 2]
        no_decay = [p for _, p in trainable if p.ndim < 2]
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
    parser = argparse.ArgumentParser(description="Train a standardized-geometry VAE and run frozen-encoder decoder calibration.")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--val_batch_size", type=int, required=True)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_workers", type=int, default=14)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--output_dir", type=str, default="./geometry_standardized_vae_runs")
    parser.add_argument("--init_ckpt", type=str, default=None, help="Warm-start weights only.")
    parser.add_argument("--resume_joint_ckpt", type=str, default=None,
                        help="Resume joint epoch/step, VAE parameters, optimizer and scheduler state while replacing incompatible frozen non-VAE modules with current initialization.")
    parser.add_argument("--skip_prepare_vae_data", action="store_true", default=False)
    parser.add_argument("--force_prepare_vae_data", action="store_true", default=False)
    parser.add_argument("--pretrain_epochs", type=int, default=12)
    parser.add_argument("--joint_epochs", type=int, default=28)
    parser.add_argument("--calibration_epochs", type=int, default=5)
    parser.add_argument("--geometry_weight", type=float, default=0.01)
    parser.add_argument("--geometry_warmup_epochs", type=int, default=5)
    parser.add_argument("--geometry_num_agents", type=int, default=84)
    parser.add_argument("--geometry_num_directions", type=int, default=20)
    parser.add_argument("--geometry_perturbation", type=float, default=0.05)
    parser.add_argument("--geometry_scale_weight", type=float, default=0.1)
    parser.add_argument("--geometry_anchor_weight", type=float, default=0.1,
                        help="Weight inside geometry loss for fixing E[log(sensitivity)], the global log-Jacobian-energy scale.")
    parser.add_argument("--geometry_anchor_init_batches", type=int, default=4,
                        help="Number of first geometry-training batches used to set the fixed sensitivity target.")
    parser.add_argument("--geometry_detach_latent_center", action=argparse.BooleanOptionalAction, default=True,
                        help="Detach posterior-mean latent centers before geometry decoding so geometry gradients update the decoder, not the encoder.")
    parser.add_argument("--latent_stats_momentum", type=float, default=0.99)
    parser.add_argument("--calibration_lr", type=float, default=2e-5)
    parser.add_argument("--endpoint_loss_weight", type=float, default=5.0,
                        help="Soft final-frame endpoint MSE weight. Use 0 to reproduce a no-L_end run.")
    parser.add_argument("--validation_geometry_seed", type=int, default=1729)
    parser.add_argument("--joint_save_top_k", type=int, default=3)
    parser.add_argument("--calibration_save_top_k", type=int, default=3)
    parser.add_argument("--joint_selection", choices=["total", "reconstruction", "geometry", "last"], default="total")
    parser.add_argument("--final_selection", choices=["total", "reconstruction", "geometry", "last"], default="total")
    parser.add_argument("--stats_device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stats_log_interval", type=int, default=200)
    QCNetFM.add_model_specific_args(parser)
    parser.set_defaults(num_dec_layers=2)
    args = parser.parse_args()
    args.num_dec_layers = 2
    if args.init_ckpt and args.resume_joint_ckpt:
        parser.error("--init_ckpt and --resume_joint_ckpt cannot be used together.")
    if args.skip_prepare_vae_data and args.force_prepare_vae_data:
        parser.error("--skip_prepare_vae_data and --force_prepare_vae_data cannot be used together.")
    args.vae_only = True
    args.freeze_vae = False
    args.scorer_only = False
    args.latent_regression_only = False
    return args


def build_model_kwargs(args: argparse.Namespace, geometry_stage: str) -> Dict[str, Any]:
    cfg = vars(args).copy()
    cfg["geometry_stage"] = geometry_stage
    for key in ["init_ckpt", "resume_joint_ckpt", "skip_prepare_vae_data", "force_prepare_vae_data",
                "output_dir", "precision", "stats_device", "stats_log_interval", "seed", "accelerator", "devices",
                "joint_save_top_k", "calibration_save_top_k", "joint_selection", "final_selection"]:
        cfg.pop(key, None)
    return cfg


def _load_checkpoint(path: str | Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def prepare_partial_vae_resume_checkpoint(model: nn.Module, checkpoint_path: str | Path,
                                          output_path: Path) -> str:
    """Create a Lightning checkpoint that resumes VAE optimization but ignores frozen architecture changes.

    The original optimizer/scheduler/loop state is preserved. The model state is
    rebuilt on the current model definition: every compatible tensor is copied
    from the checkpoint, while incompatible ``fm_decoder.*`` tensors and other
    frozen-module differences use the current initialization. All VAE tensors
    must exist with identical shapes, otherwise optimizer-state restoration would
    be unsafe and the function raises.
    """
    checkpoint = _load_checkpoint(checkpoint_path)
    old_state = checkpoint.get("state_dict")
    if not isinstance(old_state, dict):
        raise RuntimeError(f"Resume checkpoint has no Lightning state_dict: {checkpoint_path}")
    current_state = model.state_dict()
    merged_state: Dict[str, torch.Tensor] = {}
    copied, reset_fm, reset_other, ignored_old = [], [], [], []

    vae_prefixes = ("latent_encoder.", "latent_decoder.")
    missing_vae = []
    mismatched_vae = []
    for key, current_value in current_state.items():
        old_value = old_state.get(key)
        if old_value is not None and tuple(old_value.shape) == tuple(current_value.shape):
            merged_state[key] = old_value.detach().cpu()
            copied.append(key)
            continue
        merged_state[key] = current_value.detach().cpu().clone()
        if key.startswith(vae_prefixes):
            if old_value is None:
                missing_vae.append(key)
            else:
                mismatched_vae.append((key, tuple(old_value.shape), tuple(current_value.shape)))
        elif key.startswith("fm_decoder."):
            reset_fm.append(key)
        else:
            reset_other.append(key)

    ignored_old = sorted(key for key in old_state if key not in current_state)
    if missing_vae or mismatched_vae:
        raise RuntimeError(
            "Cannot restore the old optimizer safely because the VAE parameter structure changed. "
            f"missing VAE keys={missing_vae[:20]}, mismatched VAE keys={mismatched_vae[:20]}"
        )

    optimizer_states = checkpoint.get("optimizer_states", [])
    if not optimizer_states:
        raise RuntimeError("Resume checkpoint contains no optimizer_states.")
    trainable_count = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    checkpoint_param_count = sum(
        len(group.get("params", []))
        for group in optimizer_states[0].get("param_groups", [])
    )
    if checkpoint_param_count != trainable_count:
        raise RuntimeError(
            "Optimizer parameter count does not match the current trainable VAE. "
            f"checkpoint={checkpoint_param_count}, current={trainable_count}. "
            "The VAE architecture or trainable-parameter selection changed."
        )

    checkpoint["state_dict"] = merged_state
    # Start new ModelCheckpoint bookkeeping in the new output directory while
    # retaining optimizer, scheduler, epoch, global step and loop progress.
    checkpoint.pop("callbacks", None)
    checkpoint.setdefault("partial_resume_metadata", {}).update({
        "source_checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "resume_scope": "VAE parameters + optimizer/scheduler + trainer loop state",
        "copied_compatible_tensors": len(copied),
        "reset_fm_decoder_tensors": len(reset_fm),
        "reset_other_frozen_tensors": len(reset_other),
        "ignored_old_tensors": len(ignored_old),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"[Partial Resume] source: {Path(checkpoint_path).expanduser().resolve()}")
    print(f"[Partial Resume] sanitized checkpoint: {output_path.resolve()}")
    print(f"[Partial Resume] copied compatible tensors: {len(copied):,}")
    print(f"[Partial Resume] reset current fm_decoder tensors: {len(reset_fm):,}")
    print(f"[Partial Resume] reset other frozen/current-only tensors: {len(reset_other):,}")
    print(f"[Partial Resume] ignored checkpoint-only tensors: {len(ignored_old):,}")
    print(f"[Partial Resume] optimizer parameter tensors: {checkpoint_param_count:,}")
    return str(output_path.resolve())


def load_initial_weights(model: nn.Module, checkpoint_path: Optional[str]) -> None:
    if checkpoint_path is None:
        return
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = dict(checkpoint.get("state_dict", checkpoint))
    reset_anchor_keys = (
        "geometry_log_sensitivity_target", "geometry_anchor_log_sensitivity_sum",
        "geometry_anchor_sensitivity_sum", "geometry_anchor_sample_count",
        "geometry_anchor_batches_seen")
    removed_anchor_keys = [key for key in reset_anchor_keys if state_dict.pop(key, None) is not None]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if hasattr(model, "geometry_log_sensitivity_target"):
        model.geometry_log_sensitivity_target.zero_()
        model.geometry_anchor_log_sensitivity_sum.zero_()
        model.geometry_anchor_sample_count.zero_()
        model.geometry_anchor_batches_seen.zero_()
    print(f"[Init] loaded compatible weights from: {checkpoint_path}")
    print(f"[Init] reset E[log sensitivity] anchor state; removed keys: {removed_anchor_keys}")
    print(f"[Init] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def build_trainer(args: argparse.Namespace, stage_name: str, max_epochs: int,
                  callbacks: Sequence[ModelCheckpoint]) -> pl.Trainer:
    logger = TensorBoardLogger(save_dir=args.output_dir, name=stage_name)
    strategy = DDPStrategy(find_unused_parameters=False) if args.devices > 1 else "auto"
    all_callbacks = list(callbacks) + [LearningRateMonitor(logging_interval="epoch")]
    return pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy=strategy,
                      precision=args.precision, max_epochs=max_epochs, accumulate_grad_batches=1,
                      callbacks=all_callbacks, logger=logger, log_every_n_steps=20)


def move_batch_to_device(data: Any, device: torch.device):
    if isinstance(data, dict):
        return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in data.items()}
    if isinstance(data, (Batch, HeteroData)):
        return data.to(device)
    raise TypeError(f"Unsupported batch type: {type(data)}")


def vae_cache_state(cache_dir: Path) -> Tuple[bool, bool, int]:
    if not cache_dir.is_dir():
        return False, False, 0
    success = (cache_dir / "_SUCCESS").is_file()
    count = sum(1 for path in cache_dir.rglob("*") if path.is_file() and path.name != "_SUCCESS")
    return count > 0, success, count


def prepare_vae_cache_if_needed(datamodule: ArgoverseV2DataModule, args: argparse.Namespace) -> None:
    cache_dir = Path(args.vae_processed_dir)
    has_files, has_success, count = vae_cache_state(cache_dir)
    if args.skip_prepare_vae_data:
        print("[Data] skipped VAE preprocessing.")
        return
    if has_files and not args.force_prepare_vae_data:
        print(f"[Data] existing VAE cache found ({count:,} files, _SUCCESS={has_success}); skipping: {cache_dir}")
        return
    if args.force_prepare_vae_data and cache_dir.exists():
        print(f"[Data] deleting old VAE cache: {cache_dir}")
        shutil.rmtree(cache_dir)
    print(f"[Data] preparing VAE cache: {cache_dir}")
    datamodule.prepare_vae_data()
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "_SUCCESS").touch()


@torch.no_grad()
def compute_exact_latent_stats(model: StandardizedGeometryAwareQCNetFM, train_loader: Iterable,
                               device: torch.device, log_interval: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    model.to(device)
    model.eval()
    latent_sum = torch.zeros(model.vae_num_intents, model.latent_dim, dtype=torch.float64)
    latent_sq_sum = torch.zeros_like(latent_sum)
    count = 0
    for batch_idx, data in enumerate(train_loader):
        data = move_batch_to_device(data, device)
        target, predict_mask = model._extract_target_mask(data)
        valid_agents = predict_mask.any(dim=-1)
        if valid_agents.any():
            mu, _, _ = model.latent_encoder.vae.encode(target, predict_mask=predict_mask)
            values = mu.transpose(0, 1).contiguous()[valid_agents].double().cpu()
            latent_sum += values.sum(dim=0)
            latent_sq_sum += values.pow(2).sum(dim=0)
            count += int(values.size(0))
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


def checkpoint_score(callback: ModelCheckpoint) -> Optional[float]:
    score = callback.best_model_score
    return None if score is None else float(score.detach().cpu().item())


def select_checkpoint(selection: str, total: ModelCheckpoint, reconstruction: ModelCheckpoint,
                      geometry: ModelCheckpoint, resume: ModelCheckpoint) -> str:
    paths = {"total": total.best_model_path, "reconstruction": reconstruction.best_model_path,
             "geometry": geometry.best_model_path, "last": resume.last_model_path}
    selected = paths[selection]
    if not selected:
        selected = resume.last_model_path or total.best_model_path or reconstruction.best_model_path or geometry.best_model_path
    if not selected:
        raise RuntimeError(f"No checkpoint available. Paths: {paths}")
    return selected


def make_stage_callbacks(stage_dir: Path, save_top_k: int) -> Tuple[ModelCheckpoint, ModelCheckpoint, ModelCheckpoint, ModelCheckpoint]:
    resume = ModelCheckpoint(dirpath=stage_dir / "resume", filename="resume-{epoch:03d}-{step}", monitor=None,
                             save_top_k=1, save_last=True, save_weights_only=False, every_n_epochs=1,
                             save_on_train_epoch_end=True)
    total = ModelCheckpoint(dirpath=stage_dir / "best_total",
                            filename="total-{epoch:03d}-{val_selection_score:.6f}", monitor="val_selection_score",
                            mode="min", save_top_k=max(1, save_top_k), save_last=False, save_weights_only=False)
    reconstruction = ModelCheckpoint(dirpath=stage_dir / "best_reconstruction",
                                     filename="reconstruction-{epoch:03d}-{val_vae_recon:.6f}",
                                     monitor="val_vae_recon", mode="min", save_top_k=max(1, save_top_k),
                                     save_last=False, save_weights_only=False)
    geometry = ModelCheckpoint(dirpath=stage_dir / "best_geometry",
                               filename="geometry-{epoch:03d}-{val_geo_loss:.6f}", monitor="val_geo_loss",
                               mode="min", save_top_k=max(1, save_top_k), save_last=False, save_weights_only=False)
    return resume, total, reconstruction, geometry


def export_vae_weights(checkpoint_path: str, output_path: Path, metadata: Dict[str, Any]) -> str:
    checkpoint = _load_checkpoint(checkpoint_path)
    state = checkpoint.get("state_dict", checkpoint)
    latent_encoder = {k.removeprefix("latent_encoder."): v.detach().cpu()
                      for k, v in state.items() if k.startswith("latent_encoder.")}
    if not latent_encoder or "z_mean" not in state or "z_std" not in state:
        raise RuntimeError(f"Checkpoint lacks latent_encoder or z statistics: {checkpoint_path}")
    payload = {"latent_encoder": latent_encoder, "z_mean": state["z_mean"].detach().cpu(),
               "z_std": state["z_std"].detach().cpu(), "source_checkpoint": str(Path(checkpoint_path).resolve()),
               **metadata}
    torch.save(payload, output_path)
    return str(output_path.resolve())


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    print(f"[Config] num_dec_layers={args.num_dec_layers}, geometry anchor=E[log sensitivity]")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datamodule = ArgoverseV2DataModule(**vars(args))
    datamodule.setup(stage="fit")
    prepare_vae_cache_if_needed(datamodule, args)
    train_loader = datamodule.vae_train_dataloader()
    val_loader = datamodule.val_dataloader()

    print("\n========== Stage A+B: warm-up + standardized-geometry joint training ==========")
    joint_model = StandardizedGeometryAwareQCNetFM(**build_model_kwargs(args, "joint"))
    resume_ckpt_path = None
    if args.resume_joint_ckpt is None:
        load_initial_weights(joint_model, args.init_ckpt)
    else:
        resume_ckpt_path = prepare_partial_vae_resume_checkpoint(
            joint_model, args.resume_joint_ckpt,
            output_dir / "_partial_resume" / "vae_optimizer_resume.ckpt")
    joint_resume, joint_total, joint_recon, joint_geo = make_stage_callbacks(
        output_dir / "joint", args.joint_save_top_k)
    joint_trainer = build_trainer(args, "joint", args.pretrain_epochs + args.joint_epochs,
                                  [joint_resume, joint_total, joint_recon, joint_geo])
    joint_trainer.fit(joint_model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                      ckpt_path=resume_ckpt_path)
    if not joint_trainer.is_global_zero:
        return
    joint_selected = select_checkpoint(args.joint_selection, joint_total, joint_recon, joint_geo, joint_resume)
    print(f"[Joint] selected {args.joint_selection}: {joint_selected}")
    joint_payload = _load_checkpoint(joint_selected)
    joint_model.load_state_dict(joint_payload["state_dict"], strict=True)

    print("\n========== Stage C: exact full-train-set latent statistics ==========")
    exact_mean, exact_std, count = compute_exact_latent_stats(
        joint_model, train_loader, torch.device(args.stats_device), args.stats_log_interval)
    stats_path = output_dir / "exact_latent_stats.json"
    anchor_initialized = bool(joint_model._geometry_anchor_initialized())
    anchor_target_log = float(joint_model.geometry_log_sensitivity_target.detach().cpu().item())
    anchor_target_geometric_mean_sensitivity = float(torch.exp(
        joint_model.geometry_log_sensitivity_target.detach().cpu()).item())
    stats_path.write_text(json.dumps({"geometry_space": "standardized",
                                      "geometry_anchor_type": "expected_log_sensitivity",
                                      "num_dec_layers": int(args.num_dec_layers),
                                      "num_valid_agents": count,
                                      "mean": exact_mean.tolist(), "std": exact_std.tolist(),
                                      "geometry_anchor_initialized": anchor_initialized,
                                      "geometry_anchor_target_log_sensitivity": anchor_target_log,
                                      "geometry_anchor_target_geometric_mean_sensitivity": anchor_target_geometric_mean_sensitivity},
                                     indent=2, ensure_ascii=False), encoding="utf-8")
    joint_payload["state_dict"] = {k: v.detach().cpu() for k, v in joint_model.state_dict().items()}
    joint_payload.setdefault("geometry_metadata", {}).update({
        "geometry_space": "standardized", "uses_decoder_calibration": True,
        "geometry_anchor_type": "expected_log_sensitivity",
        "num_dec_layers": int(args.num_dec_layers),
        "joint_selection": args.joint_selection, "source_checkpoint": str(Path(joint_selected).resolve()),
        "geometry_anchor_initialized": anchor_initialized,
        "geometry_anchor_target_log_sensitivity": anchor_target_log,
        "geometry_anchor_target_geometric_mean_sensitivity": anchor_target_geometric_mean_sensitivity,
        "exact_latent_statistics": {"num_valid_agents": count, "mean": exact_mean.tolist(),
                                    "std": exact_std.tolist()}})
    joint_exact_path = output_dir / "joint_selected_with_exact_stats.ckpt"
    torch.save(joint_payload, joint_exact_path)
    joint_state = joint_payload["state_dict"]
    del joint_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n========== Stage D: frozen-encoder decoder calibration ==========")
    calibration_model = StandardizedGeometryAwareQCNetFM(**build_model_kwargs(args, "calibration"))
    calibration_model.load_state_dict(joint_state, strict=True)
    calibration_model.z_second_moment.copy_(
        (exact_std.pow(2) + exact_mean.pow(2)).unsqueeze(0).to(calibration_model.z_second_moment))
    calibration_model.z_stats_initialized = True
    trainable_names = [name for name, p in calibration_model.named_parameters() if p.requires_grad]
    print(f"[Calibration] trainable tensors: {len(trainable_names)}")
    for name in trainable_names[:30]:
        print(f"  - {name}")
    calibration_resume, calibration_total, calibration_recon, calibration_geo = make_stage_callbacks(
        output_dir / "calibration", args.calibration_save_top_k)
    calibration_trainer = build_trainer(args, "calibration", args.calibration_epochs,
                                        [calibration_resume, calibration_total, calibration_recon, calibration_geo])
    calibration_trainer.fit(calibration_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    final_source = select_checkpoint(args.final_selection, calibration_total, calibration_recon,
                                     calibration_geo, calibration_resume)
    print(f"[Calibration] selected {args.final_selection}: {final_source}")

    aliases = {
        "best_total": (calibration_total.best_model_path, output_dir / "geometry_aware_vae_best_total.ckpt"),
        "best_reconstruction": (calibration_recon.best_model_path, output_dir / "geometry_aware_vae_best_reconstruction.ckpt"),
        "best_geometry": (calibration_geo.best_model_path, output_dir / "geometry_aware_vae_best_geometry.ckpt"),
        "last": (calibration_resume.last_model_path, output_dir / "geometry_aware_vae_last.ckpt")}
    alias_paths: Dict[str, Optional[str]] = {}
    weight_paths: Dict[str, Optional[str]] = {}
    metadata = {"geometry_space": "standardized", "uses_decoder_calibration": True,
                "num_dec_layers": int(args.num_dec_layers),
                "geometry_anchor_type": "expected_log_sensitivity",
                "geometry_detach_latent_center": bool(args.geometry_detach_latent_center),
                "endpoint_loss_weight": float(args.endpoint_loss_weight),
                "geometry_anchor_weight": float(args.geometry_anchor_weight),
                "geometry_anchor_initialized": anchor_initialized,
                "geometry_anchor_target_log_sensitivity": anchor_target_log,
                "geometry_anchor_target_geometric_mean_sensitivity": anchor_target_geometric_mean_sensitivity}
    for label, (source, destination) in aliases.items():
        if source:
            shutil.copy2(source, destination)
            alias_paths[label] = str(destination.resolve())
            weight_paths[label] = export_vae_weights(
                source, output_dir / f"geometry_aware_vae_{label}_weights.pt", metadata)
        else:
            alias_paths[label] = None
            weight_paths[label] = None
    final_path = output_dir / "geometry_aware_vae_final.ckpt"
    shutil.copy2(final_source, final_path)
    default_weights = output_dir / "geometry_aware_vae_weights.pt"
    export_vae_weights(final_source, default_weights, metadata)

    manifest = {
        "pipeline": "3-layer decoder + E[log sensitivity] geometry joint training + exact statistics + frozen-encoder decoder calibration",
        "geometry_space": "standardized", "uses_decoder_calibration": True,
        "num_dec_layers": int(args.num_dec_layers),
        "geometry_anchor_type": "expected_log_sensitivity",
        "joint_selection": args.joint_selection, "joint_selected_checkpoint": str(Path(joint_selected).resolve()),
        "joint_exact_checkpoint": str(joint_exact_path.resolve()),
        "final_selection": args.final_selection, "formal_final_checkpoint": str(final_path.resolve()),
        "default_vae_only_weights": str(default_weights.resolve()), "latent_statistics": str(stats_path.resolve()),
        "endpoint_loss_weight": float(args.endpoint_loss_weight), "geometry_weight": float(args.geometry_weight),
        "geometry_detach_latent_center": bool(args.geometry_detach_latent_center),
        "geometry_anchor_weight": float(args.geometry_anchor_weight),
        "geometry_anchor_init_batches": int(args.geometry_anchor_init_batches),
        "geometry_anchor_initialized": anchor_initialized,
        "geometry_anchor_target_log_sensitivity": anchor_target_log,
        "geometry_anchor_target_geometric_mean_sensitivity": anchor_target_geometric_mean_sensitivity,
        "joint": {"best_total": {"score": checkpoint_score(joint_total), "path": joint_total.best_model_path},
                  "best_reconstruction": {"score": checkpoint_score(joint_recon), "path": joint_recon.best_model_path},
                  "best_geometry": {"score": checkpoint_score(joint_geo), "path": joint_geo.best_model_path},
                  "last": joint_resume.last_model_path},
        "calibration": {"aliases": alias_paths, "weights": weight_paths,
                        "best_total_score": checkpoint_score(calibration_total),
                        "best_reconstruction_score": checkpoint_score(calibration_recon),
                        "best_geometry_score": checkpoint_score(calibration_geo)},
        "exact_latent_statistics": {"num_valid_agents": count, "mean": exact_mean.tolist(), "std": exact_std.tolist()}}
    manifest_path = output_dir / "standardized_geometry_checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n========== Final outputs ==========")
    print(f"[Final] checkpoint      : {final_path}")
    print(f"[Final] VAE-only weights: {default_weights}")
    print(f"[Final] latent stats    : {stats_path}")
    print(f"[Final] manifest        : {manifest_path}")
    print(f"[Resume joint] --resume_joint_ckpt {joint_resume.last_model_path or '<joint/resume/last.ckpt>'}")


if __name__ == "__main__":
    main()