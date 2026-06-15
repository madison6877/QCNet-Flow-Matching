from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Batch

from datamodules import ArgoverseV2DataModule
from predictors import QCNetFM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在完全相同的 AV2 validation focal-agent 子集上，逐维比较："
            "target latent 分布、确定性回归 residual、FM 场景内方差、"
            "FM 分布中心误差及方差校准比。"
        )
    )

    parser.add_argument(
        "--fm_ckpt",
        type=str,
        required=True,
        help="普通 latent FM 的 Lightning checkpoint。",
    )
    parser.add_argument(
        "--reg_ckpt",
        type=str,
        required=True,
        help=(
            "包含训练完成 latent_regressor 的确定性回归 Lightning checkpoint。"
        ),
    )

    # 数据路径。若不传，则尝试读取 FM checkpoint 的 hyper_parameters。
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
            "每个场景生成的 FM latent 数量。"
            "逐维方差与中心校正建议至少使用 16，优先使用 32。"
        ),
    )
    parser.add_argument(
        "--fm_num_steps",
        type=int,
        default=None,
        help="Heun 步数；默认使用 FM checkpoint 中的设置。",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2027],
        help="FM 初始噪声随机种子。",
    )
    parser.add_argument(
        "--reg_feature_source",
        choices=["reg", "fm"],
        default="reg",
        help=(
            "确定性回归头输入 x_m 的来源。"
            "'reg' 使用回归 checkpoint 内冻结的 encoder（默认，复现回归训练）；"
            "'fm' 使用 FM checkpoint encoder（只适合两个 encoder 完全一致时）。"
        ),
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="0 表示完整 validation set。",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--compatibility_tol",
        type=float,
        default=1e-6,
        help="VAE latent encoder、z_mean、z_std 一致性检查容差。",
    )
    parser.add_argument(
        "--allow_latent_mismatch",
        action="store_true",
        help=(
            "危险选项：允许 FM 与回归 checkpoint 使用不同 VAE/标准化统计。"
            "此时逐维对比通常没有意义，正式实验不要开启。"
        ),
    )
    parser.add_argument(
        "--allow_shape_mismatch",
        action="store_true",
        help=(
            "危险选项：加载 checkpoint 时跳过尺寸不匹配参数。"
            "正式实验不要开启。"
        ),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
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

    raise TypeError(
        f"无法将 hyper_parameters 转换为普通 dict：{type(obj)}"
    )


def load_qcnetfm(
    ckpt_path: Path,
    device: torch.device,
    allow_shape_mismatch: bool,
    label: str,
) -> Tuple[QCNetFM, Dict[str, Any]]:
    checkpoint = torch_load_checkpoint(ckpt_path)
    hparams = as_plain_dict(
        checkpoint.get("hyper_parameters", {})
    )

    if not hparams:
        raise RuntimeError(
            f"{label} checkpoint 中缺少 hyper_parameters：{ckpt_path}"
        )

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

        missing, extra = model.load_state_dict(
            compatible,
            strict=False,
        )

        print(
            f"\n[警告] {label} 开启了 --allow_shape_mismatch"
        )
        print(f"兼容加载参数：{len(compatible):,}")
        print(f"尺寸不匹配：{len(mismatched):,}")

        for item in mismatched[:20]:
            print(
                f"  skip {item['key']}: "
                f"ckpt={item['checkpoint']} "
                f"current={item['current']}"
            )

        if len(mismatched) > 20:
            print(
                f"  ... 另有 {len(mismatched) - 20} 项"
            )

        print(f"missing keys：{len(missing)}")
        print(
            f"unexpected keys："
            f"{len(extra) + len(unexpected)}\n"
        )

    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, hparams


def build_datamodule(
    fm_hparams: Dict[str, Any],
    args: argparse.Namespace,
) -> ArgoverseV2DataModule:
    cfg = dict(fm_hparams)

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
            "或确认 FM checkpoint hyper_parameters 中保存了 root。"
        )

    datamodule = ArgoverseV2DataModule(**cfg)
    datamodule.setup(stage="fit")

    return datamodule


def autocast_context(
    device: torch.device,
    precision: str,
):
    if precision == "32":
        return contextlib.nullcontext()

    if device.type != "cuda":
        print(
            f"[警告] device={device} 时不启用 "
            f"{precision} autocast，改用 float32。",
            file=sys.stderr,
        )
        return contextlib.nullcontext()

    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "当前 GPU 不支持 bfloat16。"
            )
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    if precision == "fp16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    raise ValueError(f"未知 precision：{precision}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def module_max_abs_diff(
    module_a: torch.nn.Module,
    module_b: torch.nn.Module,
) -> Tuple[float, List[str]]:
    state_a = module_a.state_dict()
    state_b = module_b.state_dict()

    problems: List[str] = []
    common_keys = sorted(
        set(state_a.keys()) & set(state_b.keys())
    )

    missing_a = sorted(set(state_b.keys()) - set(state_a.keys()))
    missing_b = sorted(set(state_a.keys()) - set(state_b.keys()))

    for key in missing_a[:10]:
        problems.append(f"A 缺少 key：{key}")
    for key in missing_b[:10]:
        problems.append(f"B 缺少 key：{key}")

    max_diff = 0.0

    for key in common_keys:
        value_a = state_a[key]
        value_b = state_b[key]

        if tuple(value_a.shape) != tuple(value_b.shape):
            problems.append(
                f"{key} shape 不一致："
                f"{tuple(value_a.shape)} vs {tuple(value_b.shape)}"
            )
            continue

        if value_a.numel() == 0:
            continue

        if not (
            torch.is_floating_point(value_a)
            or torch.is_complex(value_a)
        ):
            if not torch.equal(
                value_a.detach().cpu(),
                value_b.detach().cpu(),
            ):
                problems.append(f"{key} 非浮点值不一致")
            continue

        diff = (
            value_a.detach().float().cpu()
            - value_b.detach().float().cpu()
        ).abs().max().item()

        max_diff = max(max_diff, float(diff))

    return max_diff, problems


def verify_latent_compatibility(
    fm_model: QCNetFM,
    reg_model: QCNetFM,
    tolerance: float,
    allow_mismatch: bool,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    fm_num_intents = int(fm_model.vae_num_intents)
    reg_num_intents = int(reg_model.vae_num_intents)
    fm_latent_dim = int(fm_model.latent_dim)
    reg_latent_dim = int(reg_model.latent_dim)

    checks["fm_num_intents"] = fm_num_intents
    checks["reg_num_intents"] = reg_num_intents
    checks["fm_latent_dim"] = fm_latent_dim
    checks["reg_latent_dim"] = reg_latent_dim

    shape_ok = (
        fm_num_intents == reg_num_intents
        and fm_latent_dim == reg_latent_dim
    )

    if not shape_ok:
        message = (
            "FM 与回归 checkpoint 的 latent shape 不一致："
            f"FM=({fm_num_intents},{fm_latent_dim}), "
            f"REG=({reg_num_intents},{reg_latent_dim})"
        )

        if not allow_mismatch:
            raise RuntimeError(message)
        print(f"[危险警告] {message}")

    z_mean_diff = (
        fm_model.z_mean.detach().float().cpu()
        - reg_model.z_mean.detach().float().cpu()
    ).abs().max().item()

    z_std_diff = (
        fm_model.z_std.detach().float().cpu()
        - reg_model.z_std.detach().float().cpu()
    ).abs().max().item()

    latent_encoder_diff, latent_encoder_problems = (
        module_max_abs_diff(
            fm_model.latent_encoder,
            reg_model.latent_encoder,
        )
    )

    encoder_diff, encoder_problems = module_max_abs_diff(
        fm_model.encoder,
        reg_model.encoder,
    )

    checks.update(
        {
            "z_mean_max_abs_diff": float(z_mean_diff),
            "z_std_max_abs_diff": float(z_std_diff),
            "latent_encoder_max_abs_diff": float(
                latent_encoder_diff
            ),
            "latent_encoder_problems": (
                latent_encoder_problems
            ),
            "scene_encoder_max_abs_diff": float(
                encoder_diff
            ),
            "scene_encoder_problems": encoder_problems,
        }
    )

    latent_ok = (
        shape_ok
        and z_mean_diff <= tolerance
        and z_std_diff <= tolerance
        and latent_encoder_diff <= tolerance
        and not latent_encoder_problems
    )

    checks["latent_compatible"] = bool(latent_ok)
    checks["scene_encoder_identical_within_tol"] = bool(
        encoder_diff <= tolerance
        and not encoder_problems
    )

    if not latent_ok:
        message = (
            "FM 与回归 checkpoint 不在同一 latent 坐标系："
            f"z_mean_diff={z_mean_diff:.3e}, "
            f"z_std_diff={z_std_diff:.3e}, "
            f"latent_encoder_diff={latent_encoder_diff:.3e}, "
            f"problems={latent_encoder_problems[:5]}"
        )

        if not allow_mismatch:
            raise RuntimeError(
                message
                + "\n请使用同一 VAE、同一 z_mean/z_std "
                "训练得到的两个 checkpoint。"
            )

        print(f"[危险警告] {message}")

    print("\nCheckpoint 一致性检查")
    print(
        f"latent compatible:        {checks['latent_compatible']}"
    )
    print(
        f"z_mean max abs diff:      {z_mean_diff:.6e}"
    )
    print(
        f"z_std max abs diff:       {z_std_diff:.6e}"
    )
    print(
        f"latent encoder max diff:  "
        f"{latent_encoder_diff:.6e}"
    )
    print(
        f"scene encoder max diff:   {encoder_diff:.6e}"
    )
    print(
        "scene encoder identical: "
        f"{checks['scene_encoder_identical_within_tol']}"
    )

    return checks


def get_latent_regressor(
    reg_model: QCNetFM,
) -> torch.nn.Module:
    if not hasattr(reg_model, "latent_regressor"):
        raise AttributeError(
            "回归 checkpoint 重建出的 QCNetFM 没有 "
            "latent_regressor 属性。"
        )

    regressor = getattr(
        reg_model,
        "latent_regressor",
    )

    if regressor is None:
        raise RuntimeError(
            "reg_model.latent_regressor 为 None。"
            "请确认 --reg_ckpt 是确定性 latent 回归实验的 checkpoint。"
        )

    return regressor


def normalize_regressor_output(
    output: torch.Tensor,
    num_intents: int,
    latent_dim: int,
) -> torch.Tensor:
    """
    将回归头输出统一为 [N, I, D]。
    """
    if output.ndim == 3:
        expected = (
            output.size(0),
            num_intents,
            latent_dim,
        )

        if tuple(output.shape) != expected:
            raise ValueError(
                "回归头三维输出形状错误："
                f"实际={tuple(output.shape)}, "
                f"期望={expected}"
            )

        return output

    if output.ndim == 2:
        if output.size(-1) == latent_dim and num_intents == 1:
            return output.unsqueeze(1)

        if output.size(-1) == num_intents * latent_dim:
            return output.view(
                output.size(0),
                num_intents,
                latent_dim,
            )

    raise ValueError(
        "无法识别 latent_regressor 输出形状："
        f"{tuple(output.shape)}；"
        f"num_intents={num_intents}, latent_dim={latent_dim}"
    )


@torch.no_grad()
def heun_sample_latents(
    decoder,
    ctx: Mapping[str, torch.Tensor],
    num_modes: int,
    num_steps: int,
) -> torch.Tensor:
    """
    Returns:
        [N_agents, num_modes, num_intents, latent_dim]
    """
    n_agents = int(ctx["pos_m"].size(0))
    device = ctx["pos_m"].device

    dt = 1.0 / num_steps
    time_grid = torch.linspace(
        0.0,
        1.0 - dt,
        num_steps,
        device=device,
    )

    t_cur = torch.empty(
        n_agents,
        device=device,
    )
    t_next = torch.empty(
        n_agents,
        device=device,
    )

    outputs: List[torch.Tensor] = []

    for _ in range(num_modes):
        x_t = torch.randn(
            n_agents,
            decoder.num_intents,
            decoder.latent_dim,
            device=device,
        )

        for t_value in time_grid:
            t_cur.fill_(t_value)
            t_next.fill_(t_value + dt)

            v1 = decoder._forward_core(
                ctx,
                x_t,
                t_cur,
            )

            x_euler = x_t + v1 * dt

            v2 = decoder._forward_core(
                ctx,
                x_euler,
                t_next,
            )

            x_t = (
                x_t
                + 0.5 * (v1 + v2) * dt
            )

        outputs.append(x_t)

    return torch.stack(
        outputs,
        dim=1,
    )


def create_accumulator(
    feature_dim: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    def zeros() -> torch.Tensor:
        return torch.zeros(
            feature_dim,
            device=device,
            dtype=torch.float64,
        )

    return {
        "agent_count": torch.zeros(
            (),
            device=device,
            dtype=torch.float64,
        ),

        # target latent moments
        "target_sum": zeros(),
        "target_sq_sum": zeros(),

        # deterministic regression residual
        "reg_residual_sq_sum": zeros(),

        # FM statistics
        "fm_within_var_sum": zeros(),
        "fm_center_raw_sq_sum": zeros(),
        "fm_total_sample_sq_sum": zeros(),
        "fm_mode0_sq_sum": zeros(),

        # Additional diagnostics
        "fm_mean_minus_reg_sq_sum": zeros(),
        "fm_sample_sum": zeros(),
        "fm_sample_sq_sum": zeros(),

        "fm_sample_count": torch.zeros(
            (),
            device=device,
            dtype=torch.float64,
        ),
    }


def flatten_latent(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    [N, I, D] -> [N, I*D]
    """
    if tensor.ndim != 3:
        raise ValueError(
            f"期望 [N,I,D]，实际 {tuple(tensor.shape)}"
        )

    return tensor.reshape(
        tensor.size(0),
        -1,
    )


def update_accumulator(
    accumulator: Dict[str, torch.Tensor],
    target: torch.Tensor,
    reg_prediction: torch.Tensor,
    fm_samples: torch.Tensor,
) -> None:
    """
    target:         [N,I,D]
    reg_prediction: [N,I,D]
    fm_samples:     [N,K,I,D]
    """
    if fm_samples.ndim != 4:
        raise ValueError(
            f"fm_samples 应为 [N,K,I,D]，实际 {tuple(fm_samples.shape)}"
        )

    n_agents, num_modes, num_intents, latent_dim = (
        fm_samples.shape
    )

    if target.shape != (
        n_agents,
        num_intents,
        latent_dim,
    ):
        raise ValueError(
            "target 与 FM sample 形状不匹配："
            f"target={tuple(target.shape)}, "
            f"samples={tuple(fm_samples.shape)}"
        )

    if reg_prediction.shape != target.shape:
        raise ValueError(
            "reg_prediction 与 target 形状不匹配："
            f"reg={tuple(reg_prediction.shape)}, "
            f"target={tuple(target.shape)}"
        )

    if num_modes < 2:
        raise ValueError(
            "FM within variance 需要 num_modes >= 2。"
        )

    target_flat = flatten_latent(
        target.double()
    )
    reg_flat = flatten_latent(
        reg_prediction.double()
    )

    samples_flat = fm_samples.double().reshape(
        n_agents,
        num_modes,
        -1,
    )

    fm_mean = samples_flat.mean(
        dim=1,
    )

    centered = (
        samples_flat
        - fm_mean.unsqueeze(1)
    )

    # unbiased sample variance for each scene/agent/dimension
    within_var_per_agent_dim = (
        centered.square().sum(dim=1)
        / (num_modes - 1)
    )

    center_raw_sq_per_agent_dim = (
        fm_mean - target_flat
    ).square()

    total_sample_sq_per_agent_dim = (
        samples_flat
        - target_flat.unsqueeze(1)
    ).square().mean(dim=1)

    mode0_sq_per_agent_dim = (
        samples_flat[:, 0]
        - target_flat
    ).square()

    reg_residual_sq_per_agent_dim = (
        reg_flat - target_flat
    ).square()

    fm_mean_minus_reg_sq_per_agent_dim = (
        fm_mean - reg_flat
    ).square()

    accumulator["agent_count"] += n_agents

    accumulator["target_sum"] += (
        target_flat.sum(dim=0)
    )
    accumulator["target_sq_sum"] += (
        target_flat.square().sum(dim=0)
    )

    accumulator["reg_residual_sq_sum"] += (
        reg_residual_sq_per_agent_dim.sum(dim=0)
    )

    accumulator["fm_within_var_sum"] += (
        within_var_per_agent_dim.sum(dim=0)
    )
    accumulator["fm_center_raw_sq_sum"] += (
        center_raw_sq_per_agent_dim.sum(dim=0)
    )
    accumulator["fm_total_sample_sq_sum"] += (
        total_sample_sq_per_agent_dim.sum(dim=0)
    )
    accumulator["fm_mode0_sq_sum"] += (
        mode0_sq_per_agent_dim.sum(dim=0)
    )

    accumulator["fm_mean_minus_reg_sq_sum"] += (
        fm_mean_minus_reg_sq_per_agent_dim.sum(dim=0)
    )

    all_samples_flat = samples_flat.reshape(
        n_agents * num_modes,
        -1,
    )

    accumulator["fm_sample_sum"] += (
        all_samples_flat.sum(dim=0)
    )
    accumulator["fm_sample_sq_sum"] += (
        all_samples_flat.square().sum(dim=0)
    )
    accumulator["fm_sample_count"] += (
        n_agents * num_modes
    )


def safe_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    return numerator / denominator.clamp_min(
        1e-12
    )


def finalize_accumulator(
    accumulator: Dict[str, torch.Tensor],
    num_modes: int,
    num_intents: int,
    latent_dim: int,
) -> Dict[str, Any]:
    count = accumulator["agent_count"].clamp_min(
        1.0
    )

    sample_count = accumulator[
        "fm_sample_count"
    ].clamp_min(1.0)

    target_mean = (
        accumulator["target_sum"] / count
    )
    target_var = (
        accumulator["target_sq_sum"] / count
        - target_mean.square()
    ).clamp_min(0.0)

    reg_residual_mse = (
        accumulator["reg_residual_sq_sum"]
        / count
    )

    fm_within_var = (
        accumulator["fm_within_var_sum"]
        / count
    )

    fm_center_raw_mse = (
        accumulator["fm_center_raw_sq_sum"]
        / count
    )

    # E[(sample_mean - target)^2]
    # = true_center_error + Var(samples|scene)/K
    #
    # 因而用 unbiased within variance 做有限 K 校正。
    fm_center_corrected_mse = (
        fm_center_raw_mse
        - fm_within_var / num_modes
    )

    fm_center_corrected_mse_clamped = (
        fm_center_corrected_mse.clamp_min(0.0)
    )

    fm_total_sample_mse = (
        accumulator["fm_total_sample_sq_sum"]
        / count
    )

    fm_mode0_mse = (
        accumulator["fm_mode0_sq_sum"]
        / count
    )

    fm_mean_minus_reg_mse = (
        accumulator["fm_mean_minus_reg_sq_sum"]
        / count
    )

    fm_marginal_mean = (
        accumulator["fm_sample_sum"]
        / sample_count
    )

    fm_marginal_var = (
        accumulator["fm_sample_sq_sum"]
        / sample_count
        - fm_marginal_mean.square()
    ).clamp_min(0.0)

    within_over_reg = safe_ratio(
        fm_within_var,
        reg_residual_mse,
    )

    center_over_reg = safe_ratio(
        fm_center_corrected_mse_clamped,
        reg_residual_mse,
    )

    total_over_reg = safe_ratio(
        fm_total_sample_mse,
        reg_residual_mse,
    )

    labels = [
        f"intent_{intent_idx}/dim_{dim_idx}"
        for intent_idx in range(num_intents)
        for dim_idx in range(latent_dim)
    ]

    per_dim = []

    for index, label in enumerate(labels):
        per_dim.append(
            {
                "label": label,
                "target_mean": target_mean[index].item(),
                "target_variance": target_var[index].item(),
                "target_std": target_var[index].sqrt().item(),

                "deterministic_residual_mse": (
                    reg_residual_mse[index].item()
                ),
                "deterministic_residual_rmse": (
                    reg_residual_mse[index]
                    .clamp_min(0.0)
                    .sqrt()
                    .item()
                ),

                "fm_within_variance_unbiased": (
                    fm_within_var[index].item()
                ),
                "fm_within_std": (
                    fm_within_var[index]
                    .clamp_min(0.0)
                    .sqrt()
                    .item()
                ),

                "fm_center_mse_raw": (
                    fm_center_raw_mse[index].item()
                ),
                "fm_center_mse_corrected": (
                    fm_center_corrected_mse[index].item()
                ),
                "fm_center_mse_corrected_clamped": (
                    fm_center_corrected_mse_clamped[
                        index
                    ].item()
                ),
                "fm_center_rmse_corrected": (
                    fm_center_corrected_mse_clamped[
                        index
                    ].sqrt().item()
                ),

                "fm_total_sample_mse": (
                    fm_total_sample_mse[index].item()
                ),
                "fm_mode0_mse": (
                    fm_mode0_mse[index].item()
                ),

                "fm_mean_vs_reg_mse": (
                    fm_mean_minus_reg_mse[index].item()
                ),

                "fm_marginal_mean": (
                    fm_marginal_mean[index].item()
                ),
                "fm_marginal_variance": (
                    fm_marginal_var[index].item()
                ),
                "fm_marginal_std": (
                    fm_marginal_var[index].sqrt().item()
                ),

                "within_variance_over_reg_residual": (
                    within_over_reg[index].item()
                ),
                "center_mse_over_reg_residual": (
                    center_over_reg[index].item()
                ),
                "total_mse_over_reg_residual": (
                    total_over_reg[index].item()
                ),
            }
        )

    feature_dim = num_intents * latent_dim

    totals = {
        "num_agents": int(count.item()),
        "num_modes": int(num_modes),
        "feature_dim": int(feature_dim),

        "target_variance_sum": (
            target_var.sum().item()
        ),
        "target_std_per_element": (
            target_var.mean().sqrt().item()
        ),

        # 与 val_latent_reg_loss 同尺度：
        # 对 latent 维求和，再对 agent 平均。
        "deterministic_residual_loss": (
            reg_residual_mse.sum().item()
        ),
        "deterministic_residual_rmse": (
            reg_residual_mse.mean().sqrt().item()
        ),

        "fm_within_variance_sum": (
            fm_within_var.sum().item()
        ),
        "fm_within_std_per_element": (
            fm_within_var.mean().sqrt().item()
        ),

        "fm_center_loss_raw": (
            fm_center_raw_mse.sum().item()
        ),
        "fm_center_loss_corrected": (
            fm_center_corrected_mse.sum().item()
        ),
        "fm_center_loss_corrected_clamped": (
            fm_center_corrected_mse_clamped.sum().item()
        ),
        "fm_center_rmse_corrected": (
            fm_center_corrected_mse_clamped
            .mean()
            .sqrt()
            .item()
        ),

        "fm_total_sample_loss": (
            fm_total_sample_mse.sum().item()
        ),
        "fm_total_sample_rmse": (
            fm_total_sample_mse.mean().sqrt().item()
        ),

        "fm_mode0_loss": (
            fm_mode0_mse.sum().item()
        ),
        "fm_mode0_rmse": (
            fm_mode0_mse.mean().sqrt().item()
        ),

        "fm_mean_vs_reg_loss": (
            fm_mean_minus_reg_mse.sum().item()
        ),
        "fm_mean_vs_reg_rmse": (
            fm_mean_minus_reg_mse.mean().sqrt().item()
        ),

        "fm_marginal_variance_sum": (
            fm_marginal_var.sum().item()
        ),
        "fm_marginal_std_per_element": (
            fm_marginal_var.mean().sqrt().item()
        ),

        "within_variance_over_reg_residual_total": (
            fm_within_var.sum()
            / reg_residual_mse.sum().clamp_min(1e-12)
        ).item(),

        "center_loss_over_reg_residual_total": (
            fm_center_corrected_mse_clamped.sum()
            / reg_residual_mse.sum().clamp_min(1e-12)
        ).item(),

        "total_sample_loss_over_reg_residual_total": (
            fm_total_sample_mse.sum()
            / reg_residual_mse.sum().clamp_min(1e-12)
        ).item(),

        # 分解自检：
        # total sample MSE
        # ≈ corrected center MSE + within variance
        "decomposition_max_abs_per_dim": (
            fm_total_sample_mse
            - fm_center_corrected_mse
            - fm_within_var
        ).abs().max().item(),
    }

    return {
        "per_dim": per_dim,
        "totals": totals,
    }


@torch.inference_mode()
def evaluate_once(
    fm_model: QCNetFM,
    reg_model: QCNetFM,
    val_loader: Iterable,
    device: torch.device,
    precision: str,
    num_modes: int,
    fm_num_steps: int,
    reg_feature_source: str,
    seed: int,
    max_batches: int,
    log_interval: int,
) -> Dict[str, Any]:
    set_seed(seed)

    fm_model.eval()
    reg_model.eval()

    regressor = get_latent_regressor(
        reg_model
    )

    num_intents = int(
        fm_model.vae_num_intents
    )
    latent_dim = int(
        fm_model.latent_dim
    )
    feature_dim = (
        num_intents * latent_dim
    )

    accumulator = create_accumulator(
        feature_dim=feature_dim,
        device=device,
    )

    processed_batches = 0
    first_batch_latent_target_diff: Optional[
        float
    ] = None

    for batch_idx, data in enumerate(val_loader):
        if (
            max_batches > 0
            and batch_idx >= max_batches
        ):
            break

        if isinstance(data, Batch):
            data["agent"]["av_index"] += (
                data["agent"]["ptr"][:-1]
            )

        data = data.to(device)

        predict_mask = data["agent"][
            "predict_mask"
        ][
            :,
            fm_model.num_historical_steps:,
        ].bool()

        if fm_model.dataset != "argoverse_v2":
            raise ValueError(
                "当前脚本仅实现 AV2 focal agent："
                f"dataset={fm_model.dataset}"
            )

        focal_mask = (
            (data["agent"]["category"] == 3)
            & predict_mask.any(dim=-1)
        )

        if not focal_mask.any():
            processed_batches += 1
            continue

        target_normalized = (
            data["agent"]["target"][
                ...,
                :fm_model.output_dim,
            ]
            / 10.0
        )

        with autocast_context(
            device,
            precision,
        ):
            fm_scene_enc = fm_model.encoder(
                data
            )

            if reg_feature_source == "reg":
                reg_scene_enc = reg_model.encoder(
                    data
                )
            elif reg_feature_source == "fm":
                reg_scene_enc = fm_scene_enc
            else:
                raise ValueError(
                    f"未知 reg_feature_source："
                    f"{reg_feature_source}"
                )

            target_raw_fm = (
                fm_model.latent_encoder.encode(
                    target_normalized,
                    predict_mask=predict_mask,
                )
            )

            # 动态检查两份 VAE 对相同轨迹产生的 latent
            # 是否真的完全一致。
            if first_batch_latent_target_diff is None:
                target_raw_reg = (
                    reg_model.latent_encoder.encode(
                        target_normalized,
                        predict_mask=predict_mask,
                    )
                )

                target_std_reg = (
                    target_raw_reg.float()
                    - reg_model.z_mean.float()
                ) / (
                    reg_model.z_std.float()
                    + 1e-6
                )

            x_m_reg = (
                reg_scene_enc["x_a"][
                    :,
                    -1,
                    :,
                ]
            )

            reg_output = regressor(
                x_m_reg
            )

            fm_ctx = (
                fm_model.fm_decoder
                ._build_graph_context(
                    data,
                    fm_scene_enc,
                )
            )

            fm_samples_all = (
                heun_sample_latents(
                    decoder=fm_model.fm_decoder,
                    ctx=fm_ctx,
                    num_modes=num_modes,
                    num_steps=fm_num_steps,
                )
            )

        target_std_fm = (
            target_raw_fm.float()
            - fm_model.z_mean.float()
        ) / (
            fm_model.z_std.float()
            + 1e-6
        )

        if first_batch_latent_target_diff is None:
            first_batch_latent_target_diff = (
                target_std_fm
                - target_std_reg
            ).abs().max().item()

            print(
                "首个有效 batch 的 standardized target "
                "max abs diff（FM vs REG）："
                f"{first_batch_latent_target_diff:.6e}"
            )

        reg_prediction_all = (
            normalize_regressor_output(
                reg_output.float(),
                num_intents=num_intents,
                latent_dim=latent_dim,
            )
        )

        target_focal = (
            target_std_fm[focal_mask]
        )
        reg_focal = (
            reg_prediction_all[focal_mask]
        )
        fm_samples_focal = (
            fm_samples_all.float()[focal_mask]
        )

        update_accumulator(
            accumulator=accumulator,
            target=target_focal,
            reg_prediction=reg_focal,
            fm_samples=fm_samples_focal,
        )

        processed_batches += 1

        if (
            log_interval > 0
            and (
                processed_batches == 1
                or processed_batches
                % log_interval
                == 0
            )
        ):
            partial = finalize_accumulator(
                accumulator=accumulator,
                num_modes=num_modes,
                num_intents=num_intents,
                latent_dim=latent_dim,
            )

            total = partial["totals"]

            print(
                f"[seed={seed}] "
                f"batch={processed_batches:>5d} "
                f"agents={total['num_agents']:>7d} "
                f"regLoss="
                f"{total['deterministic_residual_loss']:.6f} "
                f"fmWithin="
                f"{total['fm_within_variance_sum']:.6f} "
                f"fmCenterCorr="
                f"{total['fm_center_loss_corrected']:.6f} "
                f"fmTotal="
                f"{total['fm_total_sample_loss']:.6f} "
                f"within/reg="
                f"{total['within_variance_over_reg_residual_total']:.6f}"
            )

        del (
            fm_scene_enc,
            reg_scene_enc,
            target_raw_fm,
            reg_output,
            fm_ctx,
            fm_samples_all,
            target_std_fm,
            target_focal,
            reg_focal,
            fm_samples_focal,
        )

    if accumulator["agent_count"].item() <= 0:
        raise RuntimeError(
            "没有评估到任何 category==3 且未来有效的 focal agent。"
        )

    result = finalize_accumulator(
        accumulator=accumulator,
        num_modes=num_modes,
        num_intents=num_intents,
        latent_dim=latent_dim,
    )

    result.update(
        {
            "seed": int(seed),
            "num_batches": int(
                processed_batches
            ),
            "num_modes": int(num_modes),
            "fm_num_steps": int(
                fm_num_steps
            ),
            "reg_feature_source": (
                reg_feature_source
            ),
            "first_batch_standardized_target_max_abs_diff": (
                first_batch_latent_target_diff
            ),
        }
    )

    return result


def format_float(
    value: float,
    width: int = 10,
) -> str:
    return f"{value:{width}.5f}"


def print_result_table(
    result: Dict[str, Any],
) -> None:
    print("\n" + "=" * 166)
    print(
        "同一 validation focal subset 的逐维分布校准对照"
    )
    print("=" * 166)

    header = (
        f"{'dimension':>15s} | "
        f"{'targetStd':>9s} | "
        f"{'regMSE':>9s} | "
        f"{'FMwithin':>9s} | "
        f"{'FMcenter':>9s} | "
        f"{'FMtotal':>9s} | "
        f"{'within/reg':>10s} | "
        f"{'center/reg':>10s} | "
        f"{'FMmargStd':>10s} | "
        f"{'FMmean-reg':>10s}"
    )

    print(header)
    print("-" * len(header))

    for item in result["per_dim"]:
        print(
            f"{item['label']:>15s} | "
            f"{format_float(item['target_std'], 9)} | "
            f"{format_float(item['deterministic_residual_mse'], 9)} | "
            f"{format_float(item['fm_within_variance_unbiased'], 9)} | "
            f"{format_float(item['fm_center_mse_corrected'], 9)} | "
            f"{format_float(item['fm_total_sample_mse'], 9)} | "
            f"{format_float(item['within_variance_over_reg_residual'], 10)} | "
            f"{format_float(item['center_mse_over_reg_residual'], 10)} | "
            f"{format_float(item['fm_marginal_std'], 10)} | "
            f"{format_float(item['fm_mean_vs_reg_mse'], 10)}"
        )

    totals = result["totals"]

    print("\n" + "=" * 96)
    print("总量统计（对 latent 维求和）")
    print("=" * 96)

    print(
        "agents:                         "
        f"{totals['num_agents']:,}"
    )
    print(
        "target variance sum:             "
        f"{totals['target_variance_sum']:.6f}"
    )
    print(
        "deterministic residual loss/RMSE:"
        f" {totals['deterministic_residual_loss']:.6f}"
        f" / {totals['deterministic_residual_rmse']:.6f}"
    )
    print(
        "FM within variance/std:          "
        f"{totals['fm_within_variance_sum']:.6f}"
        f" / {totals['fm_within_std_per_element']:.6f}"
    )
    print(
        "FM center raw loss:              "
        f"{totals['fm_center_loss_raw']:.6f}"
    )
    print(
        "FM center corrected loss/RMSE:   "
        f"{totals['fm_center_loss_corrected']:.6f}"
        f" / {totals['fm_center_rmse_corrected']:.6f}"
    )
    print(
        "FM total sample loss/RMSE:       "
        f"{totals['fm_total_sample_loss']:.6f}"
        f" / {totals['fm_total_sample_rmse']:.6f}"
    )
    print(
        "FM mode0 loss/RMSE:              "
        f"{totals['fm_mode0_loss']:.6f}"
        f" / {totals['fm_mode0_rmse']:.6f}"
    )
    print(
        "FM mean vs reg loss/RMSE:        "
        f"{totals['fm_mean_vs_reg_loss']:.6f}"
        f" / {totals['fm_mean_vs_reg_rmse']:.6f}"
    )
    print(
        "FM marginal variance/std:        "
        f"{totals['fm_marginal_variance_sum']:.6f}"
        f" / {totals['fm_marginal_std_per_element']:.6f}"
    )
    print(
        "within variance / reg residual:  "
        f"{totals['within_variance_over_reg_residual_total']:.6f}"
    )
    print(
        "center loss / reg residual:      "
        f"{totals['center_loss_over_reg_residual_total']:.6f}"
    )
    print(
        "total loss / reg residual:       "
        f"{totals['total_sample_loss_over_reg_residual_total']:.6f}"
    )
    print(
        "decomposition max abs error:     "
        f"{totals['decomposition_max_abs_per_dim']:.6e}"
    )

    print(
        "\n说明：FMcenter 使用有限 K 校正："
        "raw_center_MSE - unbiased_within_variance / K。"
    )
    print(
        "within/reg > 1 表示 FM 的场景内随机方差"
        "大于确定性回归 residual；"
        "它是校准诊断，不是严格的真实条件方差估计。"
    )


def aggregate_seed_results(
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if not results:
        raise ValueError("results 不能为空。")

    if len(results) == 1:
        return results[0]

    # target 与 deterministic regression 对所有 seed 完全相同；
    # FM 相关量对 seeds 做简单均值。
    num_dims = len(results[0]["per_dim"])
    per_dim: List[Dict[str, Any]] = []

    for dim_index in range(num_dims):
        merged: Dict[str, Any] = {
            "label": results[0]["per_dim"][
                dim_index
            ]["label"]
        }

        keys = [
            key
            for key, value in results[0][
                "per_dim"
            ][dim_index].items()
            if key != "label"
            and isinstance(value, (int, float))
        ]

        for key in keys:
            values = torch.tensor(
                [
                    run["per_dim"][
                        dim_index
                    ][key]
                    for run in results
                ],
                dtype=torch.float64,
            )

            merged[key] = values.mean().item()
            merged[f"{key}_seed_std"] = (
                values.std(unbiased=True).item()
            )

        per_dim.append(merged)

    totals: Dict[str, Any] = {}

    for key, value in results[0]["totals"].items():
        if isinstance(value, bool):
            totals[key] = value
            continue

        if isinstance(value, int):
            # counts/configs 不跨 seed 相加。
            totals[key] = value
            continue

        if isinstance(value, float):
            values = torch.tensor(
                [
                    run["totals"][key]
                    for run in results
                ],
                dtype=torch.float64,
            )

            totals[key] = values.mean().item()
            totals[f"{key}_seed_std"] = (
                values.std(unbiased=True).item()
            )

    return {
        "per_dim": per_dim,
        "totals": totals,
        "seed": "mean",
        "num_seeds": len(results),
        "num_modes": results[0]["num_modes"],
        "fm_num_steps": results[0][
            "fm_num_steps"
        ],
        "reg_feature_source": results[0][
            "reg_feature_source"
        ],
    }


def main() -> None:
    args = parse_args()

    torch.set_float32_matmul_precision(
        "high"
    )

    if args.num_modes < 2:
        raise ValueError(
            "--num_modes 至少为 2；"
            "建议正式实验使用 16 或 32。"
        )

    fm_ckpt_path = Path(
        args.fm_ckpt
    )
    reg_ckpt_path = Path(
        args.reg_ckpt
    )

    if not fm_ckpt_path.is_file():
        raise FileNotFoundError(
            f"FM checkpoint 不存在："
            f"{fm_ckpt_path}"
        )

    if not reg_ckpt_path.is_file():
        raise FileNotFoundError(
            f"回归 checkpoint 不存在："
            f"{reg_ckpt_path}"
        )

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "指定了 CUDA，"
            "但 torch.cuda.is_available() 为 False。"
        )

    print(f"FM checkpoint:   {fm_ckpt_path}")
    print(f"REG checkpoint:  {reg_ckpt_path}")
    print(f"device:          {device}")
    print(f"precision:       {args.precision}")
    print(f"num_modes:       {args.num_modes}")
    print(f"seeds:           {args.seeds}")
    print(
        f"reg x_m source:  "
        f"{args.reg_feature_source}"
    )

    fm_model, fm_hparams = load_qcnetfm(
        ckpt_path=fm_ckpt_path,
        device=device,
        allow_shape_mismatch=(
            args.allow_shape_mismatch
        ),
        label="FM",
    )

    reg_model, _ = load_qcnetfm(
        ckpt_path=reg_ckpt_path,
        device=device,
        allow_shape_mismatch=(
            args.allow_shape_mismatch
        ),
        label="REG",
    )

    # 确认 checkpoint 中确实存在回归头。
    get_latent_regressor(reg_model)

    compatibility = verify_latent_compatibility(
        fm_model=fm_model,
        reg_model=reg_model,
        tolerance=args.compatibility_tol,
        allow_mismatch=(
            args.allow_latent_mismatch
        ),
    )

    if (
        args.reg_feature_source == "fm"
        and not compatibility[
            "scene_encoder_identical_within_tol"
        ]
    ):
        print(
            "\n[警告] 你选择了 "
            "--reg_feature_source fm，"
            "但 FM 与回归 checkpoint 的 encoder 不一致。"
            "回归头接收到的 x_m 分布可能与训练时不同。\n"
        )

    datamodule = build_datamodule(
        fm_hparams=fm_hparams,
        args=args,
    )

    val_loader = (
        datamodule.val_dataloader()
    )

    fm_num_steps = (
        int(args.fm_num_steps)
        if args.fm_num_steps is not None
        else int(fm_model.fm_num_steps)
    )

    if fm_num_steps <= 0:
        raise ValueError(
            "--fm_num_steps 必须大于 0。"
        )

    print(
        f"fm_num_steps:    {fm_num_steps}"
    )

    if args.max_batches > 0:
        print(
            f"max_batches:     {args.max_batches}"
        )

    runs: List[Dict[str, Any]] = []

    for seed in args.seeds:
        print("\n" + "=" * 110)
        print(f"开始评估 seed={seed}")
        print("=" * 110)

        result = evaluate_once(
            fm_model=fm_model,
            reg_model=reg_model,
            val_loader=val_loader,
            device=device,
            precision=args.precision,
            num_modes=args.num_modes,
            fm_num_steps=fm_num_steps,
            reg_feature_source=(
                args.reg_feature_source
            ),
            seed=seed,
            max_batches=args.max_batches,
            log_interval=args.log_interval,
        )

        runs.append(result)
        print_result_table(result)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = aggregate_seed_results(
        runs
    )

    if len(runs) > 1:
        print("\n" + "#" * 110)
        print("多 seed 平均结果")
        print("#" * 110)
        print_result_table(aggregate)

    output = {
        "fm_checkpoint": str(
            fm_ckpt_path.resolve()
        ),
        "reg_checkpoint": str(
            reg_ckpt_path.resolve()
        ),
        "precision": args.precision,
        "num_modes": int(
            args.num_modes
        ),
        "fm_num_steps": int(
            fm_num_steps
        ),
        "seeds": list(args.seeds),
        "reg_feature_source": (
            args.reg_feature_source
        ),
        "compatibility": compatibility,
        "runs": runs,
        "aggregate": aggregate,
    }

    if args.output_json:
        output_path = Path(
            args.output_json
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                output,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"\n结果已保存到："
            f"{output_path.resolve()}"
        )


if __name__ == "__main__":
    main()