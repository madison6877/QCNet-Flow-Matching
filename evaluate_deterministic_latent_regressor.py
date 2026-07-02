#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估确定性潜变量回归头（centered-raw latent）。

评估内容
--------
1. 潜变量回归：
   - mean per-agent squared error（与训练 val_latent_reg_loss 一致）
   - element-wise RMSE / MAE
   - residual energy ratio
   - explained energy fraction
   - 逐维 target / prediction / residual 统计和 Pearson 相关系数

2. 单轨迹预测：
   - 回归头预测轨迹 vs 原始 GT：ADE / FDE / MR
   - VAE target-center 重构 vs 原始 GT：衡量 VAE 解码上限
   - centered-raw 零潜变量轨迹 vs 原始 GT：均值 latent 基线
   - 回归头轨迹 vs VAE target-center 轨迹：隔离回归头误差

3. 输出：
   - summary.json
   - per_dimension.csv

缓存约定
--------
当前项目虽然沿用缓存键名 ``z_target_std``，但该字段应实际保存：

    z_target_centered_raw = z_raw - z_mean

不能再除以 z_std。

示例
----
python evaluate_deterministic_latent_regressor.py \
  --ckpt /root/autodl-tmp/checkpoints/latent_reg_best.ckpt \
  --cache_dir /root/autodl-tmp/latent_reg_cache_B \
  --split val \
  --device cuda:0 \
  --precision bf16 \
  --eval_batch_size 2048
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from predictors import QCNetFM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估 centered-raw 确定性潜变量回归头。"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="训练完成的确定性回归头 Lightning checkpoint。",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help=(
            "缓存根目录；通常包含 manifest.json、train/ 和 val/。"
            "也兼容直接传入含 shard_*.pt 的 split 目录。"
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["32", "bf16", "fp16"],
        help="模型前向精度。最终统计始终使用 float64 累积。",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2048,
        help="每次从 shard 中送入 GPU 的 agent 数量。",
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=0,
        help="最多评估多少个 shard；0 表示全部。",
    )
    parser.add_argument(
        "--miss_threshold",
        type=float,
        default=2.0,
        help="MR 使用的 FDE 阈值，单位为米。",
    )
    parser.add_argument(
        "--trajectory_scale",
        type=float,
        default=None,
        help="轨迹从网络尺度恢复到米的倍率；默认读取 manifest，缺省为 10。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="结果目录；默认保存到 cache_dir/eval_<split>_<checkpoint_stem>。",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="每多少个 shard 打印一次进度；0 表示不打印。",
    )
    return parser.parse_args()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def as_plain_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "items"):
        return dict(obj.items())
    if hasattr(obj, "__dict__"):
        return vars(obj)
    raise TypeError(f"无法将对象转换为 dict：{type(obj)}")


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "32":
        return contextlib.nullcontext()

    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "当前 GPU 不支持 bf16，请改用 --precision 32 或 fp16。"
            )
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        raise ValueError(f"未知 precision：{precision}")

    return torch.autocast(device_type="cuda", dtype=dtype)


def resolve_cache_paths(
    cache_dir: Path,
    split: str,
) -> Tuple[Path, Optional[Path]]:
    """
    支持两种输入：
      1. cache_dir/train、cache_dir/val
      2. cache_dir 本身直接包含 shard_*.pt
    """
    nested_split_dir = cache_dir / split

    if nested_split_dir.is_dir():
        split_dir = nested_split_dir
        manifest_path = cache_dir / "manifest.json"
    elif cache_dir.is_dir() and any(cache_dir.glob("shard_*.pt")):
        split_dir = cache_dir
        manifest_path = cache_dir.parent / "manifest.json"
    else:
        raise FileNotFoundError(
            "没有找到缓存 shard。\n"
            f"检查过：{nested_split_dir}/shard_*.pt\n"
            f"以及：{cache_dir}/shard_*.pt\n"
            "建议 --cache_dir 传入同时包含 train/ 和 val/ 的缓存根目录。"
        )

    if not manifest_path.is_file():
        manifest_path = None

    return split_dir, manifest_path


def read_manifest(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest_coordinate_system(manifest: Dict[str, Any]) -> None:
    if not manifest:
        print("[警告] 未找到 manifest.json，无法从元数据确认 latent 坐标系。")
        return

    coordinate_system = str(
        manifest.get("latent_coordinate_system", "")
    ).strip().lower()
    divided_by_std = manifest.get("latent_target_divided_by_std", None)

    standardized_names = {
        "standardized",
        "standardized_latent",
        "zscore",
        "z_score",
    }

    if coordinate_system in standardized_names or divided_by_std is True:
        raise RuntimeError(
            "manifest 表明该缓存是 standardized latent，但当前回归头与评估脚本"
            "要求 centered-raw latent：z_raw - z_mean。请重新生成缓存。"
        )

    if coordinate_system:
        print(f"cache latent coordinate system: {coordinate_system}")
    else:
        print(
            "[提示] manifest 未明确记录 latent_coordinate_system；"
            "脚本将通过最终逐维 std 与 checkpoint z_std 对照检查。"
        )


def load_model(
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[QCNetFM, Dict[str, Any]]:
    checkpoint = torch_load(ckpt_path)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint 必须是 dict，实际为 {type(checkpoint)}")

    hparams = as_plain_dict(checkpoint.get("hyper_parameters", {}))
    if not hparams:
        raise RuntimeError(
            "checkpoint 中缺少 hyper_parameters，无法自动重建 QCNetFM。"
        )

    # 当前 QCNetFM 只在 latent_regression_only=True 时构造 latent_regressor。
    # 评估时强制构造回归头，同时关闭其他互斥训练模式。
    hparams["latent_regression_only"] = True
    hparams["vae_only"] = False
    hparams["scorer_only"] = False

    model = QCNetFM(**hparams)
    state_dict = checkpoint.get("state_dict", checkpoint)

    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "严格加载后仍存在不兼容参数："
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    if getattr(model, "latent_regressor", None) is None:
        raise RuntimeError("模型没有构造 latent_regressor。")

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, hparams


def find_latent_target_key(shard: Dict[str, torch.Tensor]) -> str:
    candidates = (
        "z_target_centered_raw",
        "z_target_std",  # 旧键名；当前缓存中实际应为 centered-raw。
        "z_target",
    )
    for key in candidates:
        if key in shard:
            return key
    raise KeyError(
        "缓存中没有 latent target。支持的键为："
        + ", ".join(candidates)
    )


def ensure_trajectory_shape(
    trajectory: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """
    期望输出 [N, T, D]。
    对 [N, 1, T, D] 的单模态输出自动 squeeze。
    """
    if trajectory.ndim == 4 and trajectory.size(1) == 1:
        trajectory = trajectory[:, 0]

    if trajectory.ndim != 3:
        raise ValueError(
            f"{name} 应为 [N,T,D] 或 [N,1,T,D]，"
            f"实际为 {tuple(trajectory.shape)}"
        )

    return trajectory


def masked_trajectory_errors(
    prediction_m: torch.Tensor,
    target_m: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回每个 agent 的 ADE 和 FDE，单位为米。
    """
    prediction_m = prediction_m.float()
    target_m = target_m.float()
    valid_mask = valid_mask.bool()

    if prediction_m.shape != target_m.shape:
        raise ValueError(
            f"预测与目标形状不一致：pred={tuple(prediction_m.shape)}, "
            f"target={tuple(target_m.shape)}"
        )

    if valid_mask.shape != prediction_m.shape[:2]:
        raise ValueError(
            f"mask 形状不一致：mask={tuple(valid_mask.shape)}, "
            f"expected={tuple(prediction_m.shape[:2])}"
        )

    valid_counts = valid_mask.sum(dim=-1)
    if (valid_counts == 0).any():
        raise ValueError("轨迹评估样本中存在未来时间步全部无效的 agent。")

    displacement = torch.linalg.vector_norm(
        prediction_m - target_m,
        dim=-1,
    )
    mask_f = valid_mask.to(displacement.dtype)

    ade = (
        (displacement * mask_f).sum(dim=-1)
        / valid_counts.to(displacement.dtype)
    )

    time_index = torch.arange(
        valid_mask.size(1),
        device=valid_mask.device,
    ).view(1, -1)
    last_valid_index = (
        time_index.expand_as(valid_mask)
        .masked_fill(~valid_mask, -1)
        .max(dim=-1)
        .values
    )

    agent_index = torch.arange(
        prediction_m.size(0),
        device=prediction_m.device,
    )
    fde = displacement[agent_index, last_valid_index]

    return ade, fde


def finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) else None


def tensor_quantiles(values: torch.Tensor) -> Dict[str, float]:
    values = values.double().flatten()
    if values.numel() == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "q90": float("nan"),
            "q95": float("nan"),
        }

    return {
        "mean": values.mean().item(),
        "median": values.median().item(),
        "q90": torch.quantile(values, 0.90).item(),
        "q95": torch.quantile(values, 0.95).item(),
    }


class LatentAccumulator:
    def __init__(self, latent_dim: int) -> None:
        self.latent_dim = latent_dim
        self.agent_count = 0
        self.element_count = 0
        self.vector_count = 0

        self.sse = 0.0
        self.sae = 0.0
        self.target_energy = 0.0
        self.prediction_energy = 0.0

        self.sum_target = torch.zeros(latent_dim, dtype=torch.float64)
        self.sum_prediction = torch.zeros(latent_dim, dtype=torch.float64)
        self.sum_residual = torch.zeros(latent_dim, dtype=torch.float64)

        self.sum2_target = torch.zeros(latent_dim, dtype=torch.float64)
        self.sum2_prediction = torch.zeros(latent_dim, dtype=torch.float64)
        self.sum2_residual = torch.zeros(latent_dim, dtype=torch.float64)

        self.sum_abs_residual = torch.zeros(latent_dim, dtype=torch.float64)
        self.sum_target_prediction = torch.zeros(
            latent_dim, dtype=torch.float64
        )

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        prediction = prediction.detach().double().cpu()
        target = target.detach().double().cpu()

        if prediction.shape != target.shape:
            raise ValueError(
                f"latent prediction/target 形状不一致："
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        if prediction.ndim != 3:
            raise ValueError(
                f"latent 应为 [N,K,D]，实际为 {tuple(prediction.shape)}"
            )
        if prediction.size(-1) != self.latent_dim:
            raise ValueError(
                f"latent_dim={prediction.size(-1)}，预期={self.latent_dim}"
            )

        residual = target - prediction

        self.agent_count += prediction.size(0)
        self.element_count += prediction.numel()

        target_flat = target.reshape(-1, self.latent_dim)
        prediction_flat = prediction.reshape(-1, self.latent_dim)
        residual_flat = residual.reshape(-1, self.latent_dim)
        self.vector_count += target_flat.size(0)

        self.sse += residual.pow(2).sum().item()
        self.sae += residual.abs().sum().item()
        self.target_energy += target.pow(2).sum().item()
        self.prediction_energy += prediction.pow(2).sum().item()

        self.sum_target += target_flat.sum(dim=0)
        self.sum_prediction += prediction_flat.sum(dim=0)
        self.sum_residual += residual_flat.sum(dim=0)

        self.sum2_target += target_flat.pow(2).sum(dim=0)
        self.sum2_prediction += prediction_flat.pow(2).sum(dim=0)
        self.sum2_residual += residual_flat.pow(2).sum(dim=0)

        self.sum_abs_residual += residual_flat.abs().sum(dim=0)
        self.sum_target_prediction += (
            target_flat * prediction_flat
        ).sum(dim=0)

    @staticmethod
    def _mean_std(
        value_sum: torch.Tensor,
        value_sum2: torch.Tensor,
        count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = value_sum / max(count, 1)
        variance = (
            value_sum2 / max(count, 1) - mean.pow(2)
        ).clamp_min(0.0)
        return mean, variance.sqrt()

    def compute(
        self,
        checkpoint_z_std: torch.Tensor,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if self.agent_count == 0:
            raise RuntimeError("没有累计到任何 latent 样本。")

        target_mean, target_std = self._mean_std(
            self.sum_target,
            self.sum2_target,
            self.vector_count,
        )
        pred_mean, pred_std = self._mean_std(
            self.sum_prediction,
            self.sum2_prediction,
            self.vector_count,
        )
        residual_mean, residual_std = self._mean_std(
            self.sum_residual,
            self.sum2_residual,
            self.vector_count,
        )

        target_var = target_std.pow(2)
        pred_var = pred_std.pow(2)
        covariance = (
            self.sum_target_prediction / self.vector_count
            - target_mean * pred_mean
        )
        pearson = covariance / torch.sqrt(
            (target_var * pred_var).clamp_min(1e-24)
        )
        pearson = torch.where(
            (target_var > 1e-12) & (pred_var > 1e-12),
            pearson,
            torch.full_like(pearson, float("nan")),
        )

        per_dim_residual_mse = self.sum2_residual / self.vector_count
        per_dim_target_energy = self.sum2_target / self.vector_count
        per_dim_energy_ratio = per_dim_residual_mse / (
            per_dim_target_energy.clamp_min(1e-24)
        )
        per_dim_explained = 1.0 - per_dim_energy_ratio

        checkpoint_z_std = (
            checkpoint_z_std.detach().double().cpu().view(-1)
        )

        per_dimension: List[Dict[str, Any]] = []
        for dim in range(self.latent_dim):
            per_dimension.append(
                {
                    "dimension": dim,
                    "checkpoint_z_std": checkpoint_z_std[dim].item(),
                    "target_mean": target_mean[dim].item(),
                    "target_std": target_std[dim].item(),
                    "prediction_mean": pred_mean[dim].item(),
                    "prediction_std": pred_std[dim].item(),
                    "residual_mean": residual_mean[dim].item(),
                    "residual_std": residual_std[dim].item(),
                    "residual_rmse": math.sqrt(
                        per_dim_residual_mse[dim].item()
                    ),
                    "residual_mae": (
                        self.sum_abs_residual[dim] / self.vector_count
                    ).item(),
                    "pearson_target_prediction": finite_or_none(
                        pearson[dim].item()
                    ),
                    "residual_energy_ratio": (
                        per_dim_energy_ratio[dim].item()
                    ),
                    "explained_energy_fraction": (
                        per_dim_explained[dim].item()
                    ),
                }
            )

        residual_energy_ratio = self.sse / max(
            self.target_energy, 1e-24
        )

        summary = {
            "num_agents": self.agent_count,
            "num_latent_vectors": self.vector_count,
            "num_latent_elements": self.element_count,
            "mean_per_agent_squared_error": (
                self.sse / self.agent_count
            ),
            "elementwise_mse": self.sse / self.element_count,
            "elementwise_rmse": math.sqrt(
                self.sse / self.element_count
            ),
            "elementwise_mae": self.sae / self.element_count,
            "zero_center_baseline_rmse": math.sqrt(
                self.target_energy / self.element_count
            ),
            "target_energy_per_element": (
                self.target_energy / self.element_count
            ),
            "prediction_energy_per_element": (
                self.prediction_energy / self.element_count
            ),
            "residual_energy_per_element": (
                self.sse / self.element_count
            ),
            "residual_energy_ratio": residual_energy_ratio,
            "explained_energy_fraction": (
                1.0 - residual_energy_ratio
            ),
            "target_mean_per_dimension": target_mean.tolist(),
            "target_std_per_dimension": target_std.tolist(),
            "prediction_mean_per_dimension": pred_mean.tolist(),
            "prediction_std_per_dimension": pred_std.tolist(),
            "residual_mean_per_dimension": residual_mean.tolist(),
            "residual_std_per_dimension": residual_std.tolist(),
        }

        return summary, per_dimension


class TrajectoryAccumulator:
    def __init__(self, miss_threshold: float) -> None:
        self.miss_threshold = miss_threshold
        self.ade: List[torch.Tensor] = []
        self.fde: List[torch.Tensor] = []

    def update(
        self,
        ade: torch.Tensor,
        fde: torch.Tensor,
    ) -> None:
        self.ade.append(ade.detach().double().cpu())
        self.fde.append(fde.detach().double().cpu())

    def compute(self) -> Dict[str, Any]:
        if not self.ade:
            return {
                "num_agents": 0,
                "ADE_m": {},
                "FDE_m": {},
                "MR": float("nan"),
                "miss_threshold_m": self.miss_threshold,
            }

        ade = torch.cat(self.ade)
        fde = torch.cat(self.fde)

        return {
            "num_agents": int(ade.numel()),
            "ADE_m": tensor_quantiles(ade),
            "FDE_m": tensor_quantiles(fde),
            "MR": (fde > self.miss_threshold).double().mean().item(),
            "miss_threshold_m": self.miss_threshold,
        }


def print_running_status(
    shard_index: int,
    num_shards: int,
    latent_accumulator: LatentAccumulator,
    pred_gt_accumulator: TrajectoryAccumulator,
) -> None:
    latent_rmse = math.sqrt(
        latent_accumulator.sse
        / max(latent_accumulator.element_count, 1)
    )
    energy_ratio = (
        latent_accumulator.sse
        / max(latent_accumulator.target_energy, 1e-24)
    )

    if pred_gt_accumulator.ade:
        ade = torch.cat(pred_gt_accumulator.ade)
        fde = torch.cat(pred_gt_accumulator.fde)
        ade_mean = ade.mean().item()
        fde_mean = fde.mean().item()
    else:
        ade_mean = float("nan")
        fde_mean = float("nan")

    print(
        f"[{shard_index}/{num_shards}] "
        f"latent_agents={latent_accumulator.agent_count:,} "
        f"focal_agents={sum(x.numel() for x in pred_gt_accumulator.ade):,} "
        f"latent_RMSE={latent_rmse:.6f} "
        f"residual_energy_ratio={energy_ratio:.6f} "
        f"ADE1={ade_mean:.6f}m "
        f"FDE1={fde_mean:.6f}m"
    )


@torch.inference_mode()
def evaluate(
    model: QCNetFM,
    shard_paths: Iterable[Path],
    device: torch.device,
    precision: str,
    eval_batch_size: int,
    trajectory_scale: float,
    miss_threshold: float,
    log_interval: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if eval_batch_size <= 0:
        raise ValueError("--eval_batch_size 必须大于 0。")

    latent_accumulator = LatentAccumulator(
        latent_dim=int(model.latent_dim)
    )

    prediction_vs_gt = TrajectoryAccumulator(miss_threshold)
    vae_center_vs_gt = TrajectoryAccumulator(miss_threshold)
    zero_latent_vs_gt = TrajectoryAccumulator(miss_threshold)
    prediction_vs_vae_center = TrajectoryAccumulator(miss_threshold)

    shard_paths = list(shard_paths)
    num_shards = len(shard_paths)

    target_key_used: Optional[str] = None
    num_chunks = 0

    for shard_index, shard_path in enumerate(shard_paths, start=1):
        shard = torch_load(shard_path)
        if not isinstance(shard, dict):
            raise TypeError(
                f"{shard_path} 内容不是 dict：{type(shard)}"
            )

        required_keys = {
            "x_m",
            "target",
            "predict_mask",
            "eval_mask",
        }
        missing_keys = required_keys - set(shard)
        if missing_keys:
            raise KeyError(
                f"{shard_path} 缺少字段：{sorted(missing_keys)}"
            )

        latent_target_key = find_latent_target_key(shard)
        if target_key_used is None:
            target_key_used = latent_target_key
        elif target_key_used != latent_target_key:
            raise RuntimeError(
                "不同 shard 使用了不同 latent target 键："
                f"{target_key_used} vs {latent_target_key}"
            )

        x_m_all = shard["x_m"]
        z_target_all = shard[latent_target_key]
        target_all = shard["target"]
        predict_mask_all = shard["predict_mask"].bool()
        eval_mask_all = shard["eval_mask"].bool()

        num_agents = x_m_all.size(0)
        for start in range(0, num_agents, eval_batch_size):
            end = min(start + eval_batch_size, num_agents)
            num_chunks += 1

            x_m = x_m_all[start:end].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            z_target = z_target_all[start:end].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            target = target_all[start:end].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            predict_mask = predict_mask_all[start:end].to(
                device=device,
                non_blocking=True,
            )
            eval_mask = eval_mask_all[start:end].to(
                device=device,
                non_blocking=True,
            )

            if x_m.ndim != 2 or x_m.size(-1) != model.hidden_dim:
                raise ValueError(
                    f"x_m 形状错误：{tuple(x_m.shape)}；"
                    f"预期 [N,{model.hidden_dim}]"
                )

            expected_latent_shape = (
                x_m.size(0),
                model.vae_num_intents,
                model.latent_dim,
            )
            if tuple(z_target.shape) != expected_latent_shape:
                raise ValueError(
                    f"latent target 形状错误：{tuple(z_target.shape)}；"
                    f"预期 {expected_latent_shape}"
                )

            with autocast_context(device, precision):
                z_prediction = model.latent_regressor(x_m)

                trajectory_prediction = ensure_trajectory_shape(
                    model.latent_decoder(z_prediction),
                    "trajectory_prediction",
                )
                trajectory_vae_center = ensure_trajectory_shape(
                    model.latent_decoder(z_target),
                    "trajectory_vae_center",
                )
                trajectory_zero_latent = ensure_trajectory_shape(
                    model.latent_decoder(
                        torch.zeros_like(z_prediction)
                    ),
                    "trajectory_zero_latent",
                )

            latent_accumulator.update(
                prediction=z_prediction.float(),
                target=z_target.float(),
            )

            valid_eval = eval_mask & predict_mask.any(dim=-1)
            if valid_eval.any():
                prediction_m = (
                    trajectory_prediction[valid_eval].float()
                    * trajectory_scale
                )
                vae_center_m = (
                    trajectory_vae_center[valid_eval].float()
                    * trajectory_scale
                )
                zero_latent_m = (
                    trajectory_zero_latent[valid_eval].float()
                    * trajectory_scale
                )
                target_m = (
                    target[valid_eval, :, : model.output_dim].float()
                    * trajectory_scale
                )
                mask_eval = predict_mask[valid_eval]

                if prediction_m.size(-1) != model.output_dim:
                    prediction_m = prediction_m[..., : model.output_dim]
                if vae_center_m.size(-1) != model.output_dim:
                    vae_center_m = vae_center_m[..., : model.output_dim]
                if zero_latent_m.size(-1) != model.output_dim:
                    zero_latent_m = zero_latent_m[..., : model.output_dim]

                ade, fde = masked_trajectory_errors(
                    prediction_m,
                    target_m,
                    mask_eval,
                )
                prediction_vs_gt.update(ade, fde)

                ade, fde = masked_trajectory_errors(
                    vae_center_m,
                    target_m,
                    mask_eval,
                )
                vae_center_vs_gt.update(ade, fde)

                ade, fde = masked_trajectory_errors(
                    zero_latent_m,
                    target_m,
                    mask_eval,
                )
                zero_latent_vs_gt.update(ade, fde)

                ade, fde = masked_trajectory_errors(
                    prediction_m,
                    vae_center_m,
                    mask_eval,
                )
                prediction_vs_vae_center.update(ade, fde)

            del (
                x_m,
                z_target,
                target,
                predict_mask,
                eval_mask,
                z_prediction,
                trajectory_prediction,
                trajectory_vae_center,
                trajectory_zero_latent,
            )

        if log_interval > 0 and (
            shard_index == 1
            or shard_index % log_interval == 0
            or shard_index == num_shards
        ):
            print_running_status(
                shard_index=shard_index,
                num_shards=num_shards,
                latent_accumulator=latent_accumulator,
                pred_gt_accumulator=prediction_vs_gt,
            )

    latent_summary, per_dimension = latent_accumulator.compute(
        checkpoint_z_std=model.z_std,
    )

    trajectory_summary = {
        "prediction_vs_ground_truth": prediction_vs_gt.compute(),
        "vae_target_center_vs_ground_truth": vae_center_vs_gt.compute(),
        "zero_centered_latent_vs_ground_truth": zero_latent_vs_gt.compute(),
        "prediction_vs_vae_target_center": (
            prediction_vs_vae_center.compute()
        ),
    }

    pred_gt_ade = trajectory_summary[
        "prediction_vs_ground_truth"
    ].get("ADE_m", {}).get("mean")
    zero_gt_ade = trajectory_summary[
        "zero_centered_latent_vs_ground_truth"
    ].get("ADE_m", {}).get("mean")
    pred_gt_fde = trajectory_summary[
        "prediction_vs_ground_truth"
    ].get("FDE_m", {}).get("mean")
    zero_gt_fde = trajectory_summary[
        "zero_centered_latent_vs_ground_truth"
    ].get("FDE_m", {}).get("mean")

    if (
        isinstance(pred_gt_ade, (int, float))
        and isinstance(zero_gt_ade, (int, float))
        and math.isfinite(pred_gt_ade)
        and math.isfinite(zero_gt_ade)
    ):
        trajectory_summary["improvement_over_zero_latent"] = {
            "ADE_absolute_m": zero_gt_ade - pred_gt_ade,
            "ADE_relative": (
                (zero_gt_ade - pred_gt_ade)
                / max(zero_gt_ade, 1e-12)
            ),
            "FDE_absolute_m": zero_gt_fde - pred_gt_fde,
            "FDE_relative": (
                (zero_gt_fde - pred_gt_fde)
                / max(zero_gt_fde, 1e-12)
            ),
        }

    summary = {
        "latent_target_cache_key": target_key_used,
        "num_shards": num_shards,
        "num_chunks": num_chunks,
        "latent": latent_summary,
        "trajectory": trajectory_summary,
    }

    return summary, per_dimension


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    if not rows:
        return

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def print_final_summary(summary: Dict[str, Any]) -> None:
    latent = summary["results"]["latent"]
    trajectory = summary["results"]["trajectory"]

    print("\n" + "=" * 92)
    print("Deterministic latent regressor evaluation")
    print("=" * 92)
    print(f"checkpoint: {summary['checkpoint']}")
    print(f"cache:      {summary['cache_split_dir']}")
    print(f"split:      {summary['split']}")
    print(f"coordinate: centered_raw = z_raw - z_mean")
    print("-" * 92)

    print("[Latent]")
    print(
        f"agents                         : "
        f"{latent['num_agents']:,}"
    )
    print(
        f"mean per-agent squared error   : "
        f"{latent['mean_per_agent_squared_error']:.6f}"
    )
    print(
        f"element-wise RMSE              : "
        f"{latent['elementwise_rmse']:.6f}"
    )
    print(
        f"element-wise MAE               : "
        f"{latent['elementwise_mae']:.6f}"
    )
    print(
        f"zero-center baseline RMSE      : "
        f"{latent['zero_center_baseline_rmse']:.6f}"
    )
    print(
        f"residual energy ratio          : "
        f"{latent['residual_energy_ratio']:.6f}"
    )
    print(
        f"explained energy fraction      : "
        f"{latent['explained_energy_fraction']:.6f}"
    )
    print(
        "residual std per dimension    : "
        + str(
            [
                round(value, 6)
                for value in latent["residual_std_per_dimension"]
            ]
        )
    )

    print("-" * 92)
    for name, label in (
        ("prediction_vs_ground_truth", "Regressor vs GT"),
        (
            "vae_target_center_vs_ground_truth",
            "VAE target-center vs GT",
        ),
        (
            "zero_centered_latent_vs_ground_truth",
            "Zero latent vs GT",
        ),
        (
            "prediction_vs_vae_target_center",
            "Regressor vs VAE center",
        ),
    ):
        metrics = trajectory[name]
        if metrics["num_agents"] == 0:
            print(f"[{label}] no valid focal agents")
            continue
        print(
            f"[{label}] "
            f"N={metrics['num_agents']:,} | "
            f"ADE={metrics['ADE_m']['mean']:.6f}m | "
            f"FDE={metrics['FDE_m']['mean']:.6f}m | "
            f"MR@{metrics['miss_threshold_m']:.1f}m="
            f"{metrics['MR']:.6f}"
        )

    improvement = trajectory.get("improvement_over_zero_latent")
    if improvement:
        print(
            "[Improvement over zero latent] "
            f"ADE={improvement['ADE_absolute_m']:+.6f}m "
            f"({improvement['ADE_relative']:+.2%}), "
            f"FDE={improvement['FDE_absolute_m']:+.6f}m "
            f"({improvement['FDE_relative']:+.2%})"
        )

    print("=" * 92)


def main() -> None:
    args = parse_args()

    torch.set_float32_matmul_precision("high")

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境不可用。")

    split_dir, manifest_path = resolve_cache_paths(
        cache_dir=cache_dir,
        split=args.split,
    )
    manifest = read_manifest(manifest_path)
    validate_manifest_coordinate_system(manifest)

    shard_paths = sorted(split_dir.glob("shard_*.pt"))
    if args.max_shards > 0:
        shard_paths = shard_paths[: args.max_shards]

    if not shard_paths:
        raise FileNotFoundError(
            f"在 {split_dir} 中没有找到 shard_*.pt"
        )

    print(f"checkpoint:  {ckpt_path}")
    print(f"cache split: {split_dir}")
    print(f"num shards:  {len(shard_paths)}")
    print(f"device:      {device}")
    print(f"precision:   {args.precision}")

    model, hparams = load_model(
        ckpt_path=ckpt_path,
        device=device,
    )

    trajectory_scale = args.trajectory_scale
    if trajectory_scale is None:
        trajectory_scale = float(
            manifest.get(
                "trajectory_scale",
                getattr(model, "trajectory_scale", 10.0),
            )
        )

    print(
        f"model hidden_dim={model.hidden_dim}, "
        f"latent_dim={model.latent_dim}, "
        f"num_intents={model.vae_num_intents}"
    )
    print(
        "checkpoint z_std:",
        [
            round(float(value), 6)
            for value in model.z_std.detach().cpu().view(-1)
        ],
    )
    print(f"trajectory scale: {trajectory_scale}")

    results, per_dimension = evaluate(
        model=model,
        shard_paths=shard_paths,
        device=device,
        precision=args.precision,
        eval_batch_size=args.eval_batch_size,
        trajectory_scale=trajectory_scale,
        miss_threshold=args.miss_threshold,
        log_interval=args.log_interval,
    )

    if args.output_dir is None:
        output_dir = (
            cache_dir
            / f"eval_{args.split}_{ckpt_path.stem}"
        )
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    full_summary = {
        "checkpoint": str(ckpt_path),
        "cache_split_dir": str(split_dir),
        "manifest": str(manifest_path) if manifest_path else None,
        "split": args.split,
        "device": str(device),
        "precision": args.precision,
        "eval_batch_size": args.eval_batch_size,
        "trajectory_scale": trajectory_scale,
        "miss_threshold_m": args.miss_threshold,
        "latent_coordinate_system": "centered_raw",
        "latent_target_formula": "z_raw - z_mean",
        "model": {
            "hidden_dim": int(model.hidden_dim),
            "latent_dim": int(model.latent_dim),
            "num_intents": int(model.vae_num_intents),
            "num_future_steps": int(model.num_future_steps),
            "output_dim": int(model.output_dim),
            "z_mean": (
                model.z_mean.detach().cpu().view(-1).tolist()
            ),
            "z_std": (
                model.z_std.detach().cpu().view(-1).tolist()
            ),
        },
        "results": results,
        "per_dimension": per_dimension,
    }

    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "per_dimension.csv"

    summary_path.write_text(
        json.dumps(
            full_summary,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    write_csv(csv_path, per_dimension)

    print_final_summary(full_summary)
    print(f"\nsummary JSON: {summary_path}")
    print(f"per-dim CSV:  {csv_path}")


if __name__ == "__main__":
    main()