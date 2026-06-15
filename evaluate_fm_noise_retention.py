from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch_geometric.data import Batch

from datamodules import ArgoverseV2DataModule
from predictors import QCNetFM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "统计 latent FM 从初始高斯噪声 z0 到最终 latent z1 的噪声保留率。"
            "除逐维 Pearson 相关外，还计算场景内中心化相关、线性保留系数、"
            "完整跨维线性 R²、方差保留率和 z1-z0 位移。"
        )
    )
    parser.add_argument("--ckpt", type=str, required=True)

    # 数据路径；未提供时尝试从 checkpoint hyper_parameters 读取。
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--val_raw_dir", type=str, default=None)
    parser.add_argument("--val_processed_dir", type=str, default=None)
    parser.add_argument("--train_raw_dir", type=str, default=None)
    parser.add_argument("--train_processed_dir", type=str, default=None)
    parser.add_argument("--test_raw_dir", type=str, default=None)
    parser.add_argument("--test_processed_dir", type=str, default=None)

    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--precision",
        choices=["32", "bf16", "fp16"],
        default="32",
    )
    parser.add_argument(
        "--num_modes",
        type=int,
        default=32,
        help=(
            "每个场景的初始噪声数。场景内统计至少需要 2；"
            "建议 16 或 32，结果更稳定。"
        ),
    )
    parser.add_argument(
        "--fm_num_steps",
        type=int,
        default=None,
        help="Heun 步数；默认读取 checkpoint。",
    )
    parser.add_argument(
        "--agent_scope",
        choices=["focal", "all_valid"],
        default="focal",
        help=(
            "focal：只统计 AV2 category==3；"
            "all_valid：统计所有未来有效 agent。"
        ),
    )
    parser.add_argument(
        "--record_fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help=(
            "记录 ODE 过程中的相对时间位置。会映射到最接近的 Heun step。"
            "例如 --record_fractions 0 0.5 1"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2027],
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="0 表示完整验证集。",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-6,
        help="完整线性映射求解时的 ridge 正则。",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--allow_shape_mismatch",
        action="store_true",
        help="危险诊断选项；正式评估不要使用。",
    )
    return parser.parse_args()


def torch_load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint 内容不是 dict：{type(obj)}")
    return obj


def as_plain_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "items"):
        return dict(obj.items())
    if hasattr(obj, "__dict__"):
        return vars(obj)
    raise TypeError(f"无法将 hyper_parameters 转为 dict：{type(obj)}")


def load_model(
    ckpt_path: Path,
    device: torch.device,
    allow_shape_mismatch: bool,
) -> Tuple[QCNetFM, Dict[str, Any]]:
    checkpoint = torch_load_checkpoint(ckpt_path)
    hparams = as_plain_dict(checkpoint.get("hyper_parameters", {}))

    if not hparams:
        raise RuntimeError("checkpoint 中缺少 hyper_parameters。")

    model = QCNetFM(**hparams)
    state_dict = checkpoint.get("state_dict", checkpoint)

    if not allow_shape_mismatch:
        model.load_state_dict(state_dict, strict=True)
    else:
        current = model.state_dict()
        compatible: Dict[str, torch.Tensor] = {}
        mismatched = []
        unexpected = []

        for key, value in state_dict.items():
            if key not in current:
                unexpected.append(key)
                continue
            if tuple(value.shape) != tuple(current[key].shape):
                mismatched.append(
                    {
                        "key": key,
                        "checkpoint": tuple(value.shape),
                        "current": tuple(current[key].shape),
                    }
                )
                continue
            compatible[key] = value

        missing, extra = model.load_state_dict(compatible, strict=False)

        print("\n[警告] 已开启 --allow_shape_mismatch")
        print(f"兼容参数数：{len(compatible):,}")
        print(f"尺寸不匹配参数数：{len(mismatched):,}")
        for item in mismatched[:20]:
            print(
                f"  skip {item['key']}: "
                f"ckpt={item['checkpoint']} current={item['current']}"
            )
        print(f"missing keys：{len(missing)}")
        print(f"unexpected keys：{len(extra) + len(unexpected)}\n")

    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, hparams


def build_datamodule(
    hparams: Dict[str, Any],
    args: argparse.Namespace,
) -> ArgoverseV2DataModule:
    cfg = dict(hparams)
    overrides = {
        "root": args.root,
        "val_raw_dir": args.val_raw_dir,
        "val_processed_dir": args.val_processed_dir,
        "train_raw_dir": args.train_raw_dir,
        "train_processed_dir": args.train_processed_dir,
        "test_raw_dir": args.test_raw_dir,
        "test_processed_dir": args.test_processed_dir,
        "val_batch_size": args.val_batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
    }

    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    cfg.setdefault("dataset", "argoverse_v2")
    cfg.setdefault("train_batch_size", 1)
    cfg.setdefault("val_batch_size", 1)
    cfg.setdefault("test_batch_size", 1)
    cfg.setdefault("shuffle", False)
    cfg.setdefault("num_workers", 4)
    cfg.setdefault("pin_memory", torch.cuda.is_available())
    cfg.setdefault("persistent_workers", False)

    if not cfg.get("root"):
        raise ValueError(
            "没有找到数据集 root。请通过 --root 指定，"
            "或确认 checkpoint hyper_parameters 中保存了 root。"
        )

    datamodule = ArgoverseV2DataModule(**cfg)
    datamodule.setup(stage="fit")
    return datamodule


def autocast_context(device: torch.device, precision: str):
    if precision == "32":
        return contextlib.nullcontext()

    if device.type != "cuda":
        print(
            f"[警告] device={device} 时不启用 {precision} autocast，改用 float32。",
            file=sys.stderr,
        )
        return contextlib.nullcontext()

    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 GPU 不支持 bfloat16。")
        return torch.autocast("cuda", dtype=torch.bfloat16)

    if precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)

    raise ValueError(f"未知 precision：{precision}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_record_fractions(
    fractions: Sequence[float],
    num_steps: int,
) -> Dict[int, float]:
    """
    将相对时间映射到 rollout 状态索引：
      0       -> 初始 z0
      num_steps -> 最终 z1
    """
    if num_steps <= 0:
        raise ValueError("num_steps 必须大于 0。")

    mapping: Dict[int, float] = {}
    for fraction in fractions:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                f"record_fractions 必须位于 [0,1]，收到 {fraction}"
            )
        step_index = int(round(float(fraction) * num_steps))
        step_index = min(max(step_index, 0), num_steps)
        mapping[step_index] = step_index / num_steps

    mapping[0] = 0.0
    mapping[num_steps] = 1.0
    return dict(sorted(mapping.items()))


@torch.no_grad()
def heun_rollout_with_snapshots(
    decoder,
    ctx: Mapping[str, torch.Tensor],
    initial_noises: torch.Tensor,
    num_steps: int,
    record_steps: Mapping[int, float],
) -> Dict[int, torch.Tensor]:
    """
    Args:
        initial_noises: [M, N, I, D]

    Returns:
        snapshots[step]: [N, M, I, D]

    所有 mode 在相同场景条件下独立积分。
    """
    if initial_noises.ndim != 4:
        raise ValueError(
            f"initial_noises 应为 [M,N,I,D]，实际 {tuple(initial_noises.shape)}"
        )

    num_modes, n_agents, _, _ = initial_noises.shape
    device = initial_noises.device
    dt = 1.0 / num_steps

    time_grid = torch.linspace(
        0.0,
        1.0 - dt,
        num_steps,
        device=device,
    )

    # 为减少显存，逐 mode rollout；只保存要求的时刻。
    per_step_outputs: Dict[int, List[torch.Tensor]] = {
        step: [] for step in record_steps
    }

    t_cur = torch.empty(n_agents, device=device)
    t_next = torch.empty(n_agents, device=device)

    for mode_idx in range(num_modes):
        x_t = initial_noises[mode_idx].clone()

        if 0 in per_step_outputs:
            per_step_outputs[0].append(x_t.clone())

        for step_idx, t_value in enumerate(time_grid, start=1):
            t_cur.fill_(t_value)
            t_next.fill_(t_value + dt)

            v1 = decoder._forward_core(ctx, x_t, t_cur)
            x_euler = x_t + v1 * dt
            v2 = decoder._forward_core(ctx, x_euler, t_next)
            x_t = x_t + 0.5 * (v1 + v2) * dt

            if step_idx in per_step_outputs:
                per_step_outputs[step_idx].append(x_t.clone())

    return {
        step: torch.stack(mode_outputs, dim=1)
        for step, mode_outputs in per_step_outputs.items()
    }


def create_moment_accumulator(
    feature_dim: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    同时累计 raw/global 与 within-scene-centered 统计。

    完整线性映射使用 centered 统计：
        Y_centered ≈ X_centered @ A
    """
    zeros_d = lambda: torch.zeros(
        feature_dim, device=device, dtype=torch.float64
    )
    zeros_dd = lambda: torch.zeros(
        feature_dim, feature_dim, device=device, dtype=torch.float64
    )

    return {
        # raw/global sufficient statistics
        "raw_count": torch.zeros((), device=device, dtype=torch.float64),
        "raw_sum_x": zeros_d(),
        "raw_sum_y": zeros_d(),
        "raw_sum_x2": zeros_d(),
        "raw_sum_y2": zeros_d(),
        "raw_sum_xy_diag": zeros_d(),
        "raw_xtx": zeros_dd(),
        "raw_xty": zeros_dd(),

        # scene-wise centered sufficient statistics
        "centered_count": torch.zeros((), device=device, dtype=torch.float64),
        "centered_sum_x2": zeros_d(),
        "centered_sum_y2": zeros_d(),
        "centered_sum_xy_diag": zeros_d(),
        "centered_xtx": zeros_dd(),
        "centered_xty": zeros_dd(),
        "centered_yty": zeros_dd(),

        # scalar geometry
        "agent_count": torch.zeros((), device=device, dtype=torch.float64),
        "z0_within_var_sum": torch.zeros((), device=device, dtype=torch.float64),
        "zt_within_var_sum": torch.zeros((), device=device, dtype=torch.float64),
        "delta_sq_sum": torch.zeros((), device=device, dtype=torch.float64),
        "cosine_sum": torch.zeros((), device=device, dtype=torch.float64),
        "norm_z0_sum": torch.zeros((), device=device, dtype=torch.float64),
        "norm_zt_sum": torch.zeros((), device=device, dtype=torch.float64),
    }


def update_moment_accumulator(
    accumulator: Dict[str, torch.Tensor],
    z0: torch.Tensor,
    zt: torch.Tensor,
) -> None:
    """
    z0, zt: [N, M, I, D]

    I×D 被展平为 feature_dim。
    场景内中心化在 mode 维 M 上完成。
    """
    if z0.shape != zt.shape:
        raise ValueError(
            f"z0/zt 形状不一致：{tuple(z0.shape)} vs {tuple(zt.shape)}"
        )
    if z0.ndim != 4:
        raise ValueError(f"输入应为 [N,M,I,D]，实际 {tuple(z0.shape)}")

    n_agents, num_modes, num_intents, latent_dim = z0.shape
    feature_dim = num_intents * latent_dim

    x = z0.double().reshape(n_agents, num_modes, feature_dim)
    y = zt.double().reshape(n_agents, num_modes, feature_dim)

    # ---------- raw/global ----------
    x_flat = x.reshape(-1, feature_dim)
    y_flat = y.reshape(-1, feature_dim)

    accumulator["raw_count"] += x_flat.size(0)
    accumulator["raw_sum_x"] += x_flat.sum(dim=0)
    accumulator["raw_sum_y"] += y_flat.sum(dim=0)
    accumulator["raw_sum_x2"] += x_flat.square().sum(dim=0)
    accumulator["raw_sum_y2"] += y_flat.square().sum(dim=0)
    accumulator["raw_sum_xy_diag"] += (x_flat * y_flat).sum(dim=0)
    accumulator["raw_xtx"] += x_flat.T @ x_flat
    accumulator["raw_xty"] += x_flat.T @ y_flat

    # ---------- within-scene centered ----------
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)

    xc = x_centered.reshape(-1, feature_dim)
    yc = y_centered.reshape(-1, feature_dim)

    accumulator["centered_count"] += xc.size(0)
    accumulator["centered_sum_x2"] += xc.square().sum(dim=0)
    accumulator["centered_sum_y2"] += yc.square().sum(dim=0)
    accumulator["centered_sum_xy_diag"] += (xc * yc).sum(dim=0)
    accumulator["centered_xtx"] += xc.T @ xc
    accumulator["centered_xty"] += xc.T @ yc
    accumulator["centered_yty"] += yc.T @ yc

    # 每个 agent 的场景内总方差；先在 mode 上取均值，再 sum feature。
    z0_within_var_per_agent = x_centered.square().sum(dim=-1).mean(dim=1)
    zt_within_var_per_agent = y_centered.square().sum(dim=-1).mean(dim=1)

    # 每个 agent、mode 的 zt-z0；对 mode 取均值。
    delta_sq_per_agent = (
        (y - x).square().sum(dim=-1).mean(dim=1)
    )

    # 向量 cosine：对每个 agent/mode 计算，再对 mode 平均。
    eps = 1e-12
    dot = (x * y).sum(dim=-1)
    norm_x = x.square().sum(dim=-1).sqrt()
    norm_y = y.square().sum(dim=-1).sqrt()
    cosine = dot / (norm_x * norm_y).clamp_min(eps)

    accumulator["agent_count"] += n_agents
    accumulator["z0_within_var_sum"] += z0_within_var_per_agent.sum()
    accumulator["zt_within_var_sum"] += zt_within_var_per_agent.sum()
    accumulator["delta_sq_sum"] += delta_sq_per_agent.sum()
    accumulator["cosine_sum"] += cosine.mean(dim=1).sum()
    accumulator["norm_z0_sum"] += norm_x.mean(dim=1).sum()
    accumulator["norm_zt_sum"] += norm_y.mean(dim=1).sum()


def safe_corr(
    covariance: torch.Tensor,
    variance_x: torch.Tensor,
    variance_y: torch.Tensor,
) -> torch.Tensor:
    denom = (variance_x * variance_y).clamp_min(0).sqrt()
    corr = covariance / denom.clamp_min(1e-12)
    return corr.clamp(-1.0, 1.0)


def finalize_moment_accumulator(
    accumulator: Dict[str, torch.Tensor],
    feature_dim: int,
    ridge: float,
) -> Dict[str, Any]:
    raw_count = accumulator["raw_count"].clamp_min(1.0)
    centered_count = accumulator["centered_count"].clamp_min(1.0)
    agent_count = accumulator["agent_count"].clamp_min(1.0)

    # ---------- raw/global per-dim moments ----------
    mean_x = accumulator["raw_sum_x"] / raw_count
    mean_y = accumulator["raw_sum_y"] / raw_count

    var_x_raw = (
        accumulator["raw_sum_x2"] / raw_count - mean_x.square()
    ).clamp_min(0)
    var_y_raw = (
        accumulator["raw_sum_y2"] / raw_count - mean_y.square()
    ).clamp_min(0)
    cov_raw = (
        accumulator["raw_sum_xy_diag"] / raw_count - mean_x * mean_y
    )
    corr_raw = safe_corr(cov_raw, var_x_raw, var_y_raw)
    slope_raw = cov_raw / var_x_raw.clamp_min(1e-12)

    # raw cross-correlation matrix
    exy = accumulator["raw_xty"] / raw_count
    cov_xy_matrix_raw = exy - mean_x[:, None] * mean_y[None, :]
    denom_matrix_raw = (
        var_x_raw[:, None] * var_y_raw[None, :]
    ).clamp_min(0).sqrt()
    cross_corr_matrix_raw = (
        cov_xy_matrix_raw / denom_matrix_raw.clamp_min(1e-12)
    ).clamp(-1.0, 1.0)

    # ---------- scene-wise centered ----------
    var_x_centered = (
        accumulator["centered_sum_x2"] / centered_count
    ).clamp_min(0)
    var_y_centered = (
        accumulator["centered_sum_y2"] / centered_count
    ).clamp_min(0)
    cov_centered = (
        accumulator["centered_sum_xy_diag"] / centered_count
    )
    corr_centered = safe_corr(
        cov_centered,
        var_x_centered,
        var_y_centered,
    )
    slope_centered = (
        cov_centered / var_x_centered.clamp_min(1e-12)
    )

    cov_xy_matrix_centered = (
        accumulator["centered_xty"] / centered_count
    )
    denom_matrix_centered = (
        var_x_centered[:, None] * var_y_centered[None, :]
    ).clamp_min(0).sqrt()
    cross_corr_matrix_centered = (
        cov_xy_matrix_centered
        / denom_matrix_centered.clamp_min(1e-12)
    ).clamp(-1.0, 1.0)

    # ---------- full cross-dim linear predictability ----------
    xtx = accumulator["centered_xtx"]
    xty = accumulator["centered_xty"]
    yty = accumulator["centered_yty"]

    eye = torch.eye(
        feature_dim,
        device=xtx.device,
        dtype=xtx.dtype,
    )
    scale = torch.trace(xtx) / max(feature_dim, 1)
    regularizer = max(float(ridge), 0.0) * scale.clamp_min(1e-12)

    try:
        linear_map = torch.linalg.solve(
            xtx + regularizer * eye,
            xty,
        )
    except RuntimeError:
        linear_map = torch.linalg.pinv(
            xtx + regularizer * eye
        ) @ xty

    # SSE = tr(Y'Y - 2 A'X'Y + A'X'X A)
    sse_matrix = (
        yty
        - 2.0 * linear_map.T @ xty
        + linear_map.T @ xtx @ linear_map
    )
    sse = torch.trace(sse_matrix).clamp_min(0)
    sst = torch.trace(yty).clamp_min(1e-12)
    linear_r2 = (1.0 - sse / sst).item()

    # 每个输出维度的 R²
    sse_per_output = torch.diagonal(sse_matrix).clamp_min(0)
    sst_per_output = torch.diagonal(yty).clamp_min(1e-12)
    linear_r2_per_output = 1.0 - sse_per_output / sst_per_output

    # ---------- scalar geometry ----------
    z0_within_var = (
        accumulator["z0_within_var_sum"] / agent_count
    ).item()
    zt_within_var = (
        accumulator["zt_within_var_sum"] / agent_count
    ).item()

    variance_retention_ratio = (
        zt_within_var / max(z0_within_var, 1e-12)
    )

    delta_loss = (
        accumulator["delta_sq_sum"] / agent_count
    ).item()
    delta_rmse = (delta_loss / feature_dim) ** 0.5

    return {
        "num_agents": int(agent_count.item()),
        "num_pairs": int(raw_count.item()),
        "feature_dim": int(feature_dim),

        "z0_mean_per_dim": mean_x.cpu().tolist(),
        "zt_mean_per_dim": mean_y.cpu().tolist(),
        "z0_std_raw_per_dim": var_x_raw.sqrt().cpu().tolist(),
        "zt_std_raw_per_dim": var_y_raw.sqrt().cpu().tolist(),

        "raw_corr_per_dim": corr_raw.cpu().tolist(),
        "raw_slope_per_dim": slope_raw.cpu().tolist(),
        "raw_cross_corr_matrix": cross_corr_matrix_raw.cpu().tolist(),

        # 这是最重要的逐维噪声保留指标：
        # 先对每个场景的 modes 去均值，再计算 z0 与 zt 的关系。
        "within_centered_corr_per_dim": corr_centered.cpu().tolist(),
        "within_centered_slope_per_dim": slope_centered.cpu().tolist(),
        "within_centered_cross_corr_matrix": (
            cross_corr_matrix_centered.cpu().tolist()
        ),

        # 允许跨维旋转/混合的完整线性噪声可预测率。
        "within_full_linear_R2": float(linear_r2),
        "within_full_linear_R2_per_output_dim": (
            linear_r2_per_output.cpu().tolist()
        ),
        "within_linear_map_z0_to_zt": linear_map.cpu().tolist(),

        "z0_within_scene_variance": float(z0_within_var),
        "zt_within_scene_variance": float(zt_within_var),
        "within_variance_retention_ratio": float(
            variance_retention_ratio
        ),
        "z0_within_scene_std_per_element": float(
            (z0_within_var / feature_dim) ** 0.5
        ),
        "zt_within_scene_std_per_element": float(
            (zt_within_var / feature_dim) ** 0.5
        ),

        "zt_minus_z0_loss": float(delta_loss),
        "zt_minus_z0_RMSE": float(delta_rmse),
        "mean_cosine_z0_zt": float(
            (accumulator["cosine_sum"] / agent_count).item()
        ),
        "mean_norm_z0": float(
            (accumulator["norm_z0_sum"] / agent_count).item()
        ),
        "mean_norm_zt": float(
            (accumulator["norm_zt_sum"] / agent_count).item()
        ),
    }


@torch.inference_mode()
def evaluate_once(
    model: QCNetFM,
    val_loader: Iterable,
    device: torch.device,
    precision: str,
    num_modes: int,
    fm_num_steps: int,
    agent_scope: str,
    record_steps: Mapping[int, float],
    seed: int,
    max_batches: int,
    log_interval: int,
    ridge: float,
) -> Dict[str, Any]:
    set_seed(seed)
    model.eval()

    feature_dim = (
        int(model.fm_decoder.num_intents)
        * int(model.fm_decoder.latent_dim)
    )

    accumulators = {
        step: create_moment_accumulator(feature_dim, device)
        for step in record_steps
    }

    processed_batches = 0
    selected_agent_count = 0

    for batch_idx, data in enumerate(val_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        if isinstance(data, Batch):
            data["agent"]["av_index"] += data["agent"]["ptr"][:-1]

        data = data.to(device)

        predict_mask = data["agent"]["predict_mask"][
            :,
            model.num_historical_steps:,
        ].bool()
        valid_future = predict_mask.any(dim=-1)

        if agent_scope == "focal":
            if model.dataset != "argoverse_v2":
                raise ValueError(
                    "agent_scope=focal 当前仅实现 AV2 category==3。"
                )
            selected_mask = (
                (data["agent"]["category"] == 3)
                & valid_future
            )
        elif agent_scope == "all_valid":
            selected_mask = valid_future
        else:
            raise ValueError(f"未知 agent_scope：{agent_scope}")

        if not selected_mask.any():
            processed_batches += 1
            continue

        with autocast_context(device, precision):
            scene_enc = model.encoder(data)
            ctx = model.fm_decoder._build_graph_context(data, scene_enc)

        n_agents = int(ctx["pos_m"].size(0))

        # 所有记录时刻都对应同一组 z0。
        initial_noises = torch.randn(
            num_modes,
            n_agents,
            model.fm_decoder.num_intents,
            model.fm_decoder.latent_dim,
            device=device,
        )

        with autocast_context(device, precision):
            snapshots = heun_rollout_with_snapshots(
                decoder=model.fm_decoder,
                ctx=ctx,
                initial_noises=initial_noises,
                num_steps=fm_num_steps,
                record_steps=record_steps,
            )

        # [N, M, I, D]
        z0_all = snapshots[0].float()
        z0_selected = z0_all[selected_mask]

        for step, zt_all in snapshots.items():
            zt_selected = zt_all.float()[selected_mask]
            update_moment_accumulator(
                accumulators[step],
                z0_selected,
                zt_selected,
            )

        batch_selected = int(selected_mask.sum().item())
        selected_agent_count += batch_selected
        processed_batches += 1

        if (
            log_interval > 0
            and (
                processed_batches == 1
                or processed_batches % log_interval == 0
            )
        ):
            endpoint_step = max(record_steps)
            endpoint_partial = finalize_moment_accumulator(
                accumulators[endpoint_step],
                feature_dim=feature_dim,
                ridge=ridge,
            )
            print(
                f"[seed={seed}] batch={processed_batches:>5d} "
                f"agents={selected_agent_count:>7d} "
                f"endpoint withinCorrMean="
                f"{sum(endpoint_partial['within_centered_corr_per_dim']) / feature_dim:.6f} "
                f"fullLinearR2="
                f"{endpoint_partial['within_full_linear_R2']:.6f} "
                f"varianceRatio="
                f"{endpoint_partial['within_variance_retention_ratio']:.6f} "
                f"deltaRMSE="
                f"{endpoint_partial['zt_minus_z0_RMSE']:.6f}"
            )

        del (
            scene_enc,
            ctx,
            initial_noises,
            snapshots,
            z0_all,
            z0_selected,
        )

    if selected_agent_count == 0:
        raise RuntimeError("没有选中任何可统计的 agent。")

    by_time: Dict[str, Any] = {}
    for step, fraction in record_steps.items():
        by_time[f"{fraction:.6f}"] = {
            "step": int(step),
            "fraction": float(fraction),
            **finalize_moment_accumulator(
                accumulators[step],
                feature_dim=feature_dim,
                ridge=ridge,
            ),
        }

    return {
        "seed": int(seed),
        "num_batches": int(processed_batches),
        "num_agents": int(selected_agent_count),
        "num_modes": int(num_modes),
        "fm_num_steps": int(fm_num_steps),
        "agent_scope": agent_scope,
        "by_time": by_time,
    }


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    x = torch.tensor(values, dtype=torch.float64)
    return (
        x.mean().item(),
        x.std(unbiased=True).item() if x.numel() > 1 else 0.0,
    )


def summarize_runs(
    runs: Sequence[Dict[str, Any]],
    time_keys: Sequence[str],
) -> Dict[str, Any]:
    scalar_metrics = (
        "within_full_linear_R2",
        "z0_within_scene_variance",
        "zt_within_scene_variance",
        "within_variance_retention_ratio",
        "z0_within_scene_std_per_element",
        "zt_within_scene_std_per_element",
        "zt_minus_z0_loss",
        "zt_minus_z0_RMSE",
        "mean_cosine_z0_zt",
        "mean_norm_z0",
        "mean_norm_zt",
    )

    summary: Dict[str, Any] = {}

    for time_key in time_keys:
        item: Dict[str, Any] = {}
        for metric in scalar_metrics:
            mean, std = mean_std(
                [
                    run["by_time"][time_key][metric]
                    for run in runs
                ]
            )
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std

        # 向量/矩阵保留每个 seed 的原始结果。
        for metric in (
            "raw_corr_per_dim",
            "raw_slope_per_dim",
            "raw_cross_corr_matrix",
            "within_centered_corr_per_dim",
            "within_centered_slope_per_dim",
            "within_centered_cross_corr_matrix",
            "within_full_linear_R2_per_output_dim",
            "within_linear_map_z0_to_zt",
            "z0_mean_per_dim",
            "zt_mean_per_dim",
            "z0_std_raw_per_dim",
            "zt_std_raw_per_dim",
        ):
            item[f"{metric}_runs"] = [
                run["by_time"][time_key][metric]
                for run in runs
            ]

        summary[time_key] = item

    return summary


def format_vector(values: Sequence[float], precision: int = 4) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def print_run_details(run: Dict[str, Any]) -> None:
    print("\n单次 seed 的噪声保留曲线")
    print(
        f"{'t':>7s} | {'corrMean':>9s} | {'|corr|Mean':>10s} | "
        f"{'linearR2':>9s} | {'varRatio':>9s} | "
        f"{'z_t-z_0 RMSE':>13s} | {'cos(z0,zt)':>10s}"
    )
    print("-" * 91)

    for time_key, item in run["by_time"].items():
        corr = item["within_centered_corr_per_dim"]
        corr_mean = sum(corr) / len(corr)
        abs_corr_mean = sum(abs(value) for value in corr) / len(corr)

        print(
            f"{float(time_key):7.3f} | "
            f"{corr_mean:9.5f} | "
            f"{abs_corr_mean:10.5f} | "
            f"{item['within_full_linear_R2']:9.5f} | "
            f"{item['within_variance_retention_ratio']:9.5f} | "
            f"{item['zt_minus_z0_RMSE']:13.6f} | "
            f"{item['mean_cosine_z0_zt']:10.5f}"
        )

    endpoint = run["by_time"][f"{1.0:.6f}"]
    print("\n最终时刻逐维统计")
    print(
        "within-centered corr:  "
        + format_vector(endpoint["within_centered_corr_per_dim"])
    )
    print(
        "within-centered slope: "
        + format_vector(endpoint["within_centered_slope_per_dim"])
    )
    print(
        "full linear R²/output: "
        + format_vector(
            endpoint["within_full_linear_R2_per_output_dim"]
        )
    )
    print(
        "z0 raw std:            "
        + format_vector(endpoint["z0_std_raw_per_dim"])
    )
    print(
        "z1 raw std:            "
        + format_vector(endpoint["zt_std_raw_per_dim"])
    )
    print("\nwithin-centered z0→z1 cross-correlation matrix:")
    for row in endpoint["within_centered_cross_corr_matrix"]:
        print("  " + format_vector(row))


def print_summary(
    summary: Dict[str, Any],
    time_keys: Sequence[str],
) -> None:
    print("\n" + "=" * 100)
    print("多 seed 汇总")
    print("=" * 100)
    print(
        f"{'t':>7s} | {'linearR2':>18s} | {'varRatio':>18s} | "
        f"{'deltaRMSE':>18s} | {'cos(z0,zt)':>18s}"
    )
    print("-" * 92)

    for time_key in time_keys:
        item = summary[time_key]
        print(
            f"{float(time_key):7.3f} | "
            f"{item['within_full_linear_R2_mean']:.6f} ± "
            f"{item['within_full_linear_R2_std']:.6f} | "
            f"{item['within_variance_retention_ratio_mean']:.6f} ± "
            f"{item['within_variance_retention_ratio_std']:.6f} | "
            f"{item['zt_minus_z0_RMSE_mean']:.6f} ± "
            f"{item['zt_minus_z0_RMSE_std']:.6f} | "
            f"{item['mean_cosine_z0_zt_mean']:.6f} ± "
            f"{item['mean_cosine_z0_zt_std']:.6f}"
        )


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt_path}")

    if args.num_modes < 2:
        raise ValueError(
            "--num_modes 至少为 2；建议使用 16 或 32。"
        )
    if args.num_modes < 8:
        print(
            "[警告] num_modes 较小，场景内中心化统计可能不稳定；"
            "建议正式评估使用 16 或 32。",
            file=sys.stderr,
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境不可用。")

    model, hparams = load_model(
        ckpt_path=ckpt_path,
        device=device,
        allow_shape_mismatch=args.allow_shape_mismatch,
    )

    datamodule = build_datamodule(hparams, args)
    val_loader = datamodule.val_dataloader()

    fm_num_steps = (
        int(args.fm_num_steps)
        if args.fm_num_steps is not None
        else int(model.fm_num_steps)
    )
    if fm_num_steps <= 0:
        raise ValueError("--fm_num_steps 必须大于 0。")

    record_steps = normalize_record_fractions(
        args.record_fractions,
        fm_num_steps,
    )

    print(f"checkpoint:       {ckpt_path}")
    print(f"device:           {device}")
    print(f"precision:        {args.precision}")
    print(f"num_modes:        {args.num_modes}")
    print(f"fm_num_steps:     {fm_num_steps}")
    print(f"agent_scope:      {args.agent_scope}")
    print(f"noise seeds:      {args.seeds}")
    print(f"record steps:     {record_steps}")
    print(f"ridge:            {args.ridge}")
    if args.max_batches > 0:
        print(f"max_batches:      {args.max_batches}")

    all_runs: List[Dict[str, Any]] = []

    for seed in args.seeds:
        print("\n" + "=" * 100)
        print(f"开始评估 seed={seed}")
        print("=" * 100)

        run = evaluate_once(
            model=model,
            val_loader=val_loader,
            device=device,
            precision=args.precision,
            num_modes=args.num_modes,
            fm_num_steps=fm_num_steps,
            agent_scope=args.agent_scope,
            record_steps=record_steps,
            seed=seed,
            max_batches=args.max_batches,
            log_interval=args.log_interval,
            ridge=args.ridge,
        )
        all_runs.append(run)
        print_run_details(run)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    time_keys = [f"{fraction:.6f}" for fraction in record_steps.values()]
    summary = summarize_runs(all_runs, time_keys)
    print_summary(summary, time_keys)

    output = {
        "checkpoint": str(ckpt_path.resolve()),
        "precision": args.precision,
        "num_modes": int(args.num_modes),
        "fm_num_steps": int(fm_num_steps),
        "agent_scope": args.agent_scope,
        "seeds": list(args.seeds),
        "record_steps": {
            str(step): fraction
            for step, fraction in record_steps.items()
        },
        "runs": all_runs,
        "summary": summary,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n结果已保存到：{output_path.resolve()}")


if __name__ == "__main__":
    main()