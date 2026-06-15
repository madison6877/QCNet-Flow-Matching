from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch_geometric.data import Batch

from datamodules import ArgoverseV2DataModule
from predictors import QCNetFM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估 QCNetFM 的轨迹指标与 FM endpoint latent 指标。"
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Lightning checkpoint 路径")

    # 数据路径。若不传，尝试从 checkpoint hyper_parameters 读取。
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
        help="例如 cuda:0 或 cpu",
    )
    parser.add_argument(
        "--precision",
        choices=["32", "bf16", "fp16"],
        default="32",
        help="推理精度；要与训练验证速度接近可使用 bf16",
    )

    parser.add_argument(
        "--num_modes",
        type=int,
        default=None,
        help="采样轨迹数量；默认使用 checkpoint 中的 num_modes",
    )
    parser.add_argument(
        "--max_guesses",
        type=int,
        default=6,
        help="参与 minADE/minFDE 的最多轨迹数，AV2 常用 6",
    )
    parser.add_argument(
        "--fm_num_steps",
        type=int,
        default=None,
        help="Heun ODE 步数；默认使用 checkpoint 中的 fm_num_steps",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2026],
        help="可传多个种子，例如 --seeds 2026 2027 2028",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="最多评估多少个 validation batch；0 表示完整验证集",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="每隔多少个 batch 打印一次累计指标",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="可选：将结果保存为 JSON",
    )
    parser.add_argument(
        "--allow_shape_mismatch",
        action="store_true",
        help=(
            "危险选项：跳过 checkpoint 中尺寸不匹配的权重。"
            "只用于诊断；正式比较实验时不要开启。"
        ),
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
    raise TypeError(f"无法将 hyper_parameters 转换为 dict：{type(obj)}")


def load_model(
    ckpt_path: Path,
    device: torch.device,
    allow_shape_mismatch: bool,
) -> Tuple[QCNetFM, Dict[str, Any], Dict[str, Any]]:
    checkpoint = torch_load_checkpoint(ckpt_path)
    hparams = as_plain_dict(checkpoint.get("hyper_parameters", {}))

    if not hparams:
        raise RuntimeError(
            "checkpoint 中没有 hyper_parameters，无法自动重建 QCNetFM。"
            "请使用 Lightning 保存的完整 .ckpt。"
        )

    model = QCNetFM(**hparams)
    state_dict = checkpoint.get("state_dict", checkpoint)

    if not allow_shape_mismatch:
        # 正式评估必须严格加载，防止漏载 velocity head 等关键模块。
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

        print("\n[警告] 已开启 --allow_shape_mismatch。")
        print(f"兼容加载参数数：{len(compatible):,}")
        print(f"尺寸不匹配参数数：{len(mismatched):,}")
        for item in mismatched[:20]:
            print(
                f"  skip {item['key']}: "
                f"ckpt={item['checkpoint']} current={item['current']}"
            )
        if len(mismatched) > 20:
            print(f"  ... 另有 {len(mismatched) - 20} 项")
        print(f"missing keys：{len(missing)}")
        print(f"unexpected keys：{len(extra) + len(unexpected)}")
        print(
            "[警告] 跳过关键输出头权重会使指标无效；"
            "该模式只适合定位兼容性问题。\n"
        )

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model, hparams, checkpoint


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
            raise RuntimeError("当前 GPU 不支持 bfloat16。请使用 --precision 32 或 fp16。")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    raise ValueError(f"未知 precision：{precision}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




@torch.no_grad()
def sample_trajectories_and_latents(
    model: QCNetFM,
    data,
    scene_enc: Dict[str, torch.Tensor],
    num_modes: int,
    num_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在同一次 Heun rollout 中同时返回轨迹和最终标准化 latent。

    Returns:
        trajectories:   [N_a, M, T_f, output_dim]，模型内部归一化尺度（/10）
        latent_samples: [N_a, M, I, latent_dim]，标准化 latent endpoint
        probabilities:  [N_a, M]

    这里逐行复现 QCNetFMDecoder.sample()，唯一差别是保留解码前的 x_t。
    因此 latent 与轨迹严格一一对应，不会发生二次采样导致的随机噪声不一致。
    """
    decoder = model.fm_decoder
    ctx = decoder._build_graph_context(data, scene_enc)
    n_agents = ctx["pos_m"].size(0)
    device = ctx["pos_m"].device

    dt = 1.0 / num_steps
    t_grid = torch.linspace(0.0, 1.0 - dt, num_steps, device=device)

    all_latents = []
    all_trajectories = []
    t_cur = torch.empty(n_agents, device=device)
    t_next = torch.empty(n_agents, device=device)

    for _ in range(num_modes):
        x_t = torch.randn(
            n_agents,
            decoder.num_intents,
            decoder.latent_dim,
            device=device,
        )

        for t_value in t_grid:
            t_cur.fill_(t_value)
            t_next.fill_(t_value + dt)

            v1 = decoder._forward_core(ctx, x_t, t_cur)
            x_euler = x_t + v1 * dt
            v2 = decoder._forward_core(ctx, x_euler, t_next)
            x_t = x_t + 0.5 * (v1 + v2) * dt

        # x_t 仍位于标准化 latent 空间。
        all_latents.append(x_t)
        all_trajectories.append(model.latent_decoder(x_t))

    latent_samples = torch.stack(all_latents, dim=1)
    trajectories = torch.stack(all_trajectories, dim=1)

    agent_context = scene_enc["x_a"][:, -1, :]
    logits = decoder.scorer(agent_context, trajectories)
    probabilities = torch.softmax(logits, dim=-1)

    return trajectories, latent_samples, probabilities


def select_topk_candidates(
    trajectories: torch.Tensor,
    latent_samples: torch.Tensor,
    probabilities: Optional[torch.Tensor],
    max_guesses: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """对轨迹和对应 latent 使用完全相同的候选索引。

    当生成数 M 大于 max_guesses 时，按 scorer 概率保留 top-k；
    否则全部保留，并保持原始采样顺序。
    """
    n, modes, t, d = trajectories.shape
    if latent_samples.shape[:2] != (n, modes):
        raise ValueError(
            "trajectory/latent 候选维度不匹配："
            f"traj={tuple(trajectories.shape)}, latent={tuple(latent_samples.shape)}"
        )

    keep = min(max_guesses, modes)
    if keep == modes:
        indices = torch.arange(modes, device=trajectories.device).unsqueeze(0).expand(n, -1)
        return trajectories, latent_samples, probabilities, indices

    if probabilities is None:
        indices = torch.arange(keep, device=trajectories.device).unsqueeze(0).expand(n, -1)
    else:
        indices = probabilities.topk(
            k=keep,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices

    traj_idx = indices[:, :, None, None].expand(n, keep, t, d)
    trajectories_kept = trajectories.gather(dim=1, index=traj_idx)

    _, _, num_intents, latent_dim = latent_samples.shape
    latent_idx = indices[:, :, None, None].expand(n, keep, num_intents, latent_dim)
    latent_kept = latent_samples.gather(dim=1, index=latent_idx)

    probabilities_kept = (
        probabilities.gather(dim=1, index=indices)
        if probabilities is not None
        else None
    )
    return trajectories_kept, latent_kept, probabilities_kept, indices


def latent_error_per_mode(
    latent_samples: torch.Tensor,
    latent_target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算每个候选的 endpoint latent 误差。

    Args:
        latent_samples: [N, M, I, D]，标准化 latent
        latent_target:  [N, I, D]，标准化 VAE posterior mean

    Returns:
        loss_per_mode: [N, M]
            在 intent 与 latent 维上求平方和，和 val_latent_reg_loss 同尺度。
        mse_per_mode: [N, M]
            在 intent 与 latent 维上求平均；其全局平方根与
            val_latent_reg_rmse 可直接比较。
    """
    if latent_samples.ndim != 4:
        raise ValueError(
            f"latent_samples 应为 [N,M,I,D]，实际 {tuple(latent_samples.shape)}"
        )
    if latent_target.ndim != 3:
        raise ValueError(
            f"latent_target 应为 [N,I,D]，实际 {tuple(latent_target.shape)}"
        )
    if latent_samples.shape[0] != latent_target.shape[0] or latent_samples.shape[2:] != latent_target.shape[1:]:
        raise ValueError(
            "sample/target latent 形状不匹配："
            f"samples={tuple(latent_samples.shape)}, target={tuple(latent_target.shape)}"
        )

    squared_error = (
        latent_samples.float() - latent_target.float().unsqueeze(1)
    ).pow(2)
    loss_per_mode = squared_error.sum(dim=(-1, -2))
    mse_per_mode = squared_error.mean(dim=(-1, -2))
    return loss_per_mode, mse_per_mode

def batch_minade_minfde(
    trajectories: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    trajectories: [N, K, T, D]，单位：米
    target:       [N, T, D]，单位：米
    valid_mask:   [N, T]，bool

    返回：
      minade_ade_per_agent: [N]
          直接选择 ADE 最小的模态。

      minade_fde_per_agent: [N]
          先选择 FDE 最小的模态，再计算该模态的 ADE。
          该定义与项目 minADE Metric 的默认 min_criterion='FDE' 一致，
          因而应使用它与训练日志 val_minADE 直接比较。

      minfde_per_agent: [N]
          所有候选模态中的最小 FDE。
    """
    if trajectories.ndim != 4:
        raise ValueError(
            f"trajectories 应为 [N,K,T,D]，实际 {tuple(trajectories.shape)}"
        )
    if target.ndim != 3:
        raise ValueError(f"target 应为 [N,T,D]，实际 {tuple(target.shape)}")
    if valid_mask.ndim != 2:
        raise ValueError(f"valid_mask 应为 [N,T]，实际 {tuple(valid_mask.shape)}")

    trajectories = trajectories.float()
    target = target.float()
    valid_mask = valid_mask.bool()

    n, k, t, d = trajectories.shape
    if target.shape != (n, t, d):
        raise ValueError(
            f"pred/target 形状不匹配：pred={tuple(trajectories.shape)}, "
            f"target={tuple(target.shape)}"
        )
    if valid_mask.shape != (n, t):
        raise ValueError(
            f"mask 形状不匹配：mask={tuple(valid_mask.shape)}, expected={(n, t)}"
        )

    valid_counts = valid_mask.sum(dim=-1)
    if (valid_counts == 0).any():
        raise ValueError(
            "传入 batch_minade_minfde 的 agent 中存在未来全无效样本。"
        )

    # 每个时间步的欧氏距离：[N, K, T]
    displacement = torch.linalg.vector_norm(
        trajectories - target.unsqueeze(1),
        dim=-1,
    )
    mask_f = valid_mask[:, None, :].to(displacement.dtype)

    # 每个候选模态的 ADE：[N, K]
    ade_per_mode = (
        (displacement * mask_f).sum(dim=-1)
        / valid_counts[:, None].to(displacement.dtype)
    )

    # 指标 1：直接按 ADE 选择最佳候选。
    minade_ade_per_agent = ade_per_mode.min(dim=1).values

    # 每个 agent 最后一个有效未来时间步。
    # 与项目 minADE/minFDE Metric 中 inds_last 的含义一致。
    time_indices = torch.arange(
        1,
        t + 1,
        device=valid_mask.device,
        dtype=torch.long,
    )
    last_valid_idx = (valid_mask.long() * time_indices).argmax(dim=-1)

    pred_end_idx = last_valid_idx[:, None, None, None].expand(n, k, 1, d)
    pred_end = trajectories.gather(dim=2, index=pred_end_idx).squeeze(2)

    gt_end_idx = last_valid_idx[:, None, None].expand(n, 1, d)
    gt_end = target.gather(dim=1, index=gt_end_idx).squeeze(1)

    # 每个候选模态的 FDE：[N, K]
    fde_per_mode = torch.linalg.vector_norm(
        pred_end - gt_end.unsqueeze(1),
        dim=-1,
    )

    # 指标 3：所有候选中最小 FDE。
    minfde_per_agent, best_fde_idx = fde_per_mode.min(dim=1)

    # 指标 2：使用 FDE 最小的候选对应的 ADE。
    # 这是训练日志默认 minADE 的定义。
    minade_fde_per_agent = ade_per_mode.gather(
        dim=1,
        index=best_fde_idx.unsqueeze(1),
    ).squeeze(1)

    return minade_ade_per_agent, minade_fde_per_agent, minfde_per_agent


@torch.inference_mode()
def evaluate_once(
    model: QCNetFM,
    val_loader: Iterable,
    device: torch.device,
    precision: str,
    num_modes: int,
    max_guesses: int,
    fm_num_steps: int,
    seed: int,
    max_batches: int,
    log_interval: int,
) -> Dict[str, float]:
    set_seed(seed)
    model.eval()

    minade_ade_sum = torch.zeros((), device=device, dtype=torch.float64)
    minade_fde_sum = torch.zeros((), device=device, dtype=torch.float64)
    minfde_sum = torch.zeros((), device=device, dtype=torch.float64)

    # 以下都先累计“每个 agent 的 loss/MSE”，最后统一平均。
    latent_mode0_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    latent_mode0_mse_sum = torch.zeros((), device=device, dtype=torch.float64)
    latent_scorer_top1_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    latent_scorer_top1_mse_sum = torch.zeros((), device=device, dtype=torch.float64)
    latent_min_all_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    latent_min_all_mse_sum = torch.zeros((), device=device, dtype=torch.float64)
    latent_min_kept_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    latent_min_kept_mse_sum = torch.zeros((), device=device, dtype=torch.float64)

    # 条件生成分布分解：
    # E_k ||z_k - z_gt||^2
    #   = ||mean_k(z_k) - z_gt||^2
    #   + E_k ||z_k - mean_k(z_k)||^2
    latent_sample_mean_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    latent_within_scene_var_sum = torch.zeros((), device=device, dtype=torch.float64)
    latent_total_sample_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    latent_decomposition_max_abs = torch.zeros((), device=device, dtype=torch.float64)

    agent_count = 0
    processed_batches = 0

    for batch_idx, data in enumerate(val_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        if isinstance(data, Batch):
            data["agent"]["av_index"] += data["agent"]["ptr"][:-1]

        data = data.to(device)
        predict_mask = data["agent"]["predict_mask"][
            :, model.num_historical_steps:
        ].bool()

        if model.dataset != "argoverse_v2":
            raise ValueError(f"当前脚本仅实现 AV2，实际 dataset={model.dataset}")

        eval_mask = (
            (data["agent"]["category"] == 3)
            & predict_mask.any(dim=-1)
        )

        if not eval_mask.any():
            processed_batches += 1
            continue

        target_normalized = (
            data["agent"]["target"][..., :model.output_dim] / 10.0
        )

        with autocast_context(device, precision):
            scene_enc = model.encoder(data)

            # FM 的监督目标就是 VAE posterior mean，而不是 sampled z。
            latent_target_raw = model.latent_encoder.encode(
                target_normalized,
                predict_mask=predict_mask,
            )

            trajectories_normalized, latent_samples, probabilities = (
                sample_trajectories_and_latents(
                    model=model,
                    data=data,
                    scene_enc=scene_enc,
                    num_modes=num_modes,
                    num_steps=fm_num_steps,
                )
            )

        latent_target_std = (
            latent_target_raw.float() - model.z_mean.float()
        ) / (model.z_std.float() + 1e-6)

        trajectories_m = trajectories_normalized.float() * 10.0
        target_m = target_normalized.float() * 10.0

        trajectories_eval_all = trajectories_m[eval_mask]
        latent_eval_all = latent_samples.float()[eval_mask]
        probabilities_eval = (
            probabilities.float()[eval_mask]
            if probabilities is not None
            else None
        )
        latent_target_eval = latent_target_std[eval_mask]
        target_eval = target_m[eval_mask]
        valid_mask_eval = predict_mask[eval_mask]

        # ------------------------------------------------------------
        # 条件生成分布的“中心误差 + 场景内方差”分解。
        # 这里必须使用筛选前的全部 num_modes 个候选。
        # latent_eval_all:   [N, K, num_intents, latent_dim]
        # latent_target_eval:[N, num_intents, latent_dim]
        # ------------------------------------------------------------
        sample_mean = latent_eval_all.mean(dim=1)

        mean_error_per_agent = (
            sample_mean - latent_target_eval
        ).pow(2).sum(dim=(-1, -2))

        within_var_per_agent = (
            latent_eval_all - sample_mean[:, None]
        ).pow(2).sum(dim=(-1, -2)).mean(dim=1)

        total_sample_loss_per_agent = (
            latent_eval_all - latent_target_eval[:, None]
        ).pow(2).sum(dim=(-1, -2)).mean(dim=1)

        decomposition_abs_per_agent = (
            total_sample_loss_per_agent
            - mean_error_per_agent
            - within_var_per_agent
        ).abs()

        # 所有生成候选上的 latent oracle 指标。
        latent_loss_all, latent_mse_all = latent_error_per_mode(
            latent_eval_all,
            latent_target_eval,
        )

        mode0_loss = latent_loss_all[:, 0]
        mode0_mse = latent_mse_all[:, 0]

        if probabilities_eval is None:
            scorer_top1_idx = torch.zeros(
                latent_eval_all.size(0),
                dtype=torch.long,
                device=device,
            )
        else:
            scorer_top1_idx = probabilities_eval.argmax(dim=-1)

        scorer_top1_loss = latent_loss_all.gather(
            dim=1,
            index=scorer_top1_idx.unsqueeze(1),
        ).squeeze(1)
        scorer_top1_mse = latent_mse_all.gather(
            dim=1,
            index=scorer_top1_idx.unsqueeze(1),
        ).squeeze(1)

        min_all_loss = latent_loss_all.min(dim=1).values
        min_all_mse = latent_mse_all.min(dim=1).values

        # 与轨迹评估使用同一 scorer top-k 候选集合。
        trajectories_eval, latent_eval_kept, _, _ = select_topk_candidates(
            trajectories=trajectories_eval_all,
            latent_samples=latent_eval_all,
            probabilities=probabilities_eval,
            max_guesses=max_guesses,
        )

        latent_loss_kept, latent_mse_kept = latent_error_per_mode(
            latent_eval_kept,
            latent_target_eval,
        )
        min_kept_loss = latent_loss_kept.min(dim=1).values
        min_kept_mse = latent_mse_kept.min(dim=1).values

        minade_ade, minade_fde, minfde = batch_minade_minfde(
            trajectories=trajectories_eval,
            target=target_eval,
            valid_mask=valid_mask_eval,
        )

        minade_ade_sum += minade_ade.double().sum()
        minade_fde_sum += minade_fde.double().sum()
        minfde_sum += minfde.double().sum()

        latent_mode0_loss_sum += mode0_loss.double().sum()
        latent_mode0_mse_sum += mode0_mse.double().sum()
        latent_scorer_top1_loss_sum += scorer_top1_loss.double().sum()
        latent_scorer_top1_mse_sum += scorer_top1_mse.double().sum()
        latent_min_all_loss_sum += min_all_loss.double().sum()
        latent_min_all_mse_sum += min_all_mse.double().sum()
        latent_min_kept_loss_sum += min_kept_loss.double().sum()
        latent_min_kept_mse_sum += min_kept_mse.double().sum()

        latent_sample_mean_loss_sum += mean_error_per_agent.double().sum()
        latent_within_scene_var_sum += within_var_per_agent.double().sum()
        latent_total_sample_loss_sum += total_sample_loss_per_agent.double().sum()
        latent_decomposition_max_abs = torch.maximum(
            latent_decomposition_max_abs,
            decomposition_abs_per_agent.double().max(),
        )

        agent_count += int(minade_ade.numel())
        processed_batches += 1

        if log_interval > 0 and (
            processed_batches == 1 or processed_batches % log_interval == 0
        ):
            denom = max(agent_count, 1)
            running_minade_ade = (minade_ade_sum / denom).item()
            running_minade_fde = (minade_fde_sum / denom).item()
            running_minfde = (minfde_sum / denom).item()
            running_mode0_rmse = torch.sqrt(
                latent_mode0_mse_sum / denom
            ).item()
            running_top1_rmse = torch.sqrt(
                latent_scorer_top1_mse_sum / denom
            ).item()
            running_min_all_rmse = torch.sqrt(
                latent_min_all_mse_sum / denom
            ).item()
            running_min_kept_rmse = torch.sqrt(
                latent_min_kept_mse_sum / denom
            ).item()
            running_sample_mean_loss = (
                latent_sample_mean_loss_sum / denom
            ).item()
            running_within_var = (
                latent_within_scene_var_sum / denom
            ).item()
            running_total_sample_loss = (
                latent_total_sample_loss_sum / denom
            ).item()

            print(
                f"[seed={seed}] batch={processed_batches:>5d} "
                f"agents={agent_count:>7d} "
                f"minADE_ADE={running_minade_ade:.6f} "
                f"minADE_FDE={running_minade_fde:.6f} "
                f"minFDE={running_minfde:.6f} "
                f"latent_RMSE_mode0={running_mode0_rmse:.6f} "
                f"latent_RMSE_top1={running_top1_rmse:.6f} "
                f"minLatentRMSE_all={running_min_all_rmse:.6f} "
                f"minLatentRMSE_kept={running_min_kept_rmse:.6f} "
                f"sampleMeanLoss={running_sample_mean_loss:.6f} "
                f"withinVar={running_within_var:.6f} "
                f"totalSampleLoss={running_total_sample_loss:.6f}"
            )

        del (
            scene_enc,
            latent_target_raw,
            latent_target_std,
            trajectories_normalized,
            latent_samples,
            probabilities,
            trajectories_m,
            target_m,
        )

    if agent_count == 0:
        raise RuntimeError(
            "没有评估到任何 category == 3 且未来有效的 agent。"
            "请检查数据集、category 和 predict_mask。"
        )

    mode0_loss = (latent_mode0_loss_sum / agent_count).item()
    mode0_rmse = torch.sqrt(latent_mode0_mse_sum / agent_count).item()
    top1_loss = (latent_scorer_top1_loss_sum / agent_count).item()
    top1_rmse = torch.sqrt(latent_scorer_top1_mse_sum / agent_count).item()
    min_all_loss = (latent_min_all_loss_sum / agent_count).item()
    min_all_rmse = torch.sqrt(latent_min_all_mse_sum / agent_count).item()
    min_kept_loss = (latent_min_kept_loss_sum / agent_count).item()
    min_kept_rmse = torch.sqrt(latent_min_kept_mse_sum / agent_count).item()

    sample_mean_loss = (latent_sample_mean_loss_sum / agent_count).item()
    within_scene_variance = (latent_within_scene_var_sum / agent_count).item()
    total_sample_loss = (latent_total_sample_loss_sum / agent_count).item()
    decomposition_max_abs = latent_decomposition_max_abs.item()

    latent_elements = int(model.vae_num_intents * model.latent_dim)
    sample_mean_rmse = (sample_mean_loss / latent_elements) ** 0.5
    within_scene_std = (within_scene_variance / latent_elements) ** 0.5
    total_sample_rmse = (total_sample_loss / latent_elements) ** 0.5

    return {
        "seed": int(seed),
        "minADE_ADE": (minade_ade_sum / agent_count).item(),
        "minADE_FDE": (minade_fde_sum / agent_count).item(),
        "minFDE": (minfde_sum / agent_count).item(),
        # 与 deterministic latent regression loss/RMSE 直接可比。
        "latent_endpoint_loss_mode0": mode0_loss,
        "latent_endpoint_RMSE_mode0": mode0_rmse,
        "latent_endpoint_loss_scorer_top1": top1_loss,
        "latent_endpoint_RMSE_scorer_top1": top1_rmse,
        # all：在全部 num_modes 个生成候选中 oracle 选 latent 最近者。
        "min_latent_endpoint_loss_all": min_all_loss,
        "min_latent_endpoint_RMSE_all": min_all_rmse,
        # kept：先按 scorer 截到 max_guesses，再 oracle 选 latent 最近者。
        "min_latent_endpoint_loss_kept": min_kept_loss,
        "min_latent_endpoint_RMSE_kept": min_kept_rmse,
        # 条件分布分解。loss/variance 都是在 intent×latent_dim 上求和。
        "latent_sample_mean_loss": sample_mean_loss,
        "latent_sample_mean_RMSE": sample_mean_rmse,
        "latent_within_scene_variance": within_scene_variance,
        "latent_within_scene_std": within_scene_std,
        "latent_total_sample_loss": total_sample_loss,
        "latent_total_sample_RMSE": total_sample_rmse,
        "latent_decomposition_max_abs": decomposition_max_abs,
        "num_agents": int(agent_count),
        "num_batches": int(processed_batches),
        "num_modes": int(num_modes),
        "max_guesses": int(min(max_guesses, num_modes)),
        "fm_num_steps": int(fm_num_steps),
    }


def mean_std(values):
    x = torch.tensor(values, dtype=torch.float64)
    mean = x.mean().item()
    std = x.std(unbiased=True).item() if x.numel() > 1 else 0.0
    return mean, std


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但 torch.cuda.is_available() 为 False。")

    print(f"checkpoint: {ckpt_path}")
    print(f"device:     {device}")
    print(f"precision:  {args.precision}")

    model, hparams, _ = load_model(
        ckpt_path=ckpt_path,
        device=device,
        allow_shape_mismatch=args.allow_shape_mismatch,
    )

    datamodule = build_datamodule(hparams, args)
    val_loader = datamodule.val_dataloader()

    num_modes = (
        int(args.num_modes)
        if args.num_modes is not None
        else int(model.num_modes)
    )
    fm_num_steps = (
        int(args.fm_num_steps)
        if args.fm_num_steps is not None
        else int(model.fm_num_steps)
    )

    if num_modes <= 0:
        raise ValueError("--num_modes 必须大于 0")
    if args.max_guesses <= 0:
        raise ValueError("--max_guesses 必须大于 0")
    if fm_num_steps <= 0:
        raise ValueError("--fm_num_steps 必须大于 0")

    print(f"num_modes:    {num_modes}")
    print(f"max_guesses:  {min(args.max_guesses, num_modes)}")
    print(f"fm_num_steps: {fm_num_steps}")
    print(f"seeds:        {args.seeds}")
    print(
        "latent target: standardized VAE posterior mean "
        "(与 FM 训练目标一致)"
    )
    if args.max_batches > 0:
        print(f"max_batches:  {args.max_batches}")

    all_results = []
    for seed in args.seeds:
        print("\n" + "=" * 80)
        print(f"开始评估 seed={seed}")
        print("=" * 80)

        result = evaluate_once(
            model=model,
            val_loader=val_loader,
            device=device,
            precision=args.precision,
            num_modes=num_modes,
            max_guesses=args.max_guesses,
            fm_num_steps=fm_num_steps,
            seed=seed,
            max_batches=args.max_batches,
            log_interval=args.log_interval,
        )
        all_results.append(result)

        print(
            f"\nseed={seed} 完成："
            f"minADE_ADE={result['minADE_ADE']:.6f}, "
            f"minADE_FDE={result['minADE_FDE']:.6f}, "
            f"minFDE={result['minFDE']:.6f}, "
            f"latent_RMSE_mode0={result['latent_endpoint_RMSE_mode0']:.6f}, "
            f"latent_RMSE_top1={result['latent_endpoint_RMSE_scorer_top1']:.6f}, "
            f"minLatentRMSE_all={result['min_latent_endpoint_RMSE_all']:.6f}, "
            f"minLatentRMSE_kept={result['min_latent_endpoint_RMSE_kept']:.6f}, "
            f"sampleMeanLoss={result['latent_sample_mean_loss']:.6f}, "
            f"withinVar={result['latent_within_scene_variance']:.6f}, "
            f"totalSampleLoss={result['latent_total_sample_loss']:.6f}, "
            f"agents={result['num_agents']:,}"
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    metric_names = [
        "minADE_ADE",
        "minADE_FDE",
        "minFDE",
        "latent_endpoint_loss_mode0",
        "latent_endpoint_RMSE_mode0",
        "latent_endpoint_loss_scorer_top1",
        "latent_endpoint_RMSE_scorer_top1",
        "min_latent_endpoint_loss_all",
        "min_latent_endpoint_RMSE_all",
        "min_latent_endpoint_loss_kept",
        "min_latent_endpoint_RMSE_kept",
        "latent_sample_mean_loss",
        "latent_sample_mean_RMSE",
        "latent_within_scene_variance",
        "latent_within_scene_std",
        "latent_total_sample_loss",
        "latent_total_sample_RMSE",
        "latent_decomposition_max_abs",
    ]

    summary_metrics: Dict[str, float] = {}
    for name in metric_names:
        mean, std = mean_std([r[name] for r in all_results])
        summary_metrics[f"{name}_mean"] = mean
        summary_metrics[f"{name}_std"] = std

    summary = {
        "checkpoint": str(ckpt_path.resolve()),
        "precision": args.precision,
        "seeds": list(args.seeds),
        "num_modes": num_modes,
        "max_guesses": min(args.max_guesses, num_modes),
        "fm_num_steps": fm_num_steps,
        "runs": all_results,
        "summary": summary_metrics,
    }

    print("\n" + "=" * 80)
    print("最终结果")
    print("=" * 80)
    print(
        f"minADE_ADE: {summary_metrics['minADE_ADE_mean']:.6f} "
        f"± {summary_metrics['minADE_ADE_std']:.6f} "
        "(直接按 ADE 选模态)"
    )
    print(
        f"minADE_FDE: {summary_metrics['minADE_FDE_mean']:.6f} "
        f"± {summary_metrics['minADE_FDE_std']:.6f} "
        "(与训练日志默认 minADE 一致)"
    )
    print(
        f"minFDE:     {summary_metrics['minFDE_mean']:.6f} "
        f"± {summary_metrics['minFDE_std']:.6f}"
    )

    print("\nEndpoint latent 指标（标准化 latent 空间）")
    print(
        "mode0 loss/RMSE: "
        f"{summary_metrics['latent_endpoint_loss_mode0_mean']:.6f} "
        f"± {summary_metrics['latent_endpoint_loss_mode0_std']:.6f} / "
        f"{summary_metrics['latent_endpoint_RMSE_mode0_mean']:.6f} "
        f"± {summary_metrics['latent_endpoint_RMSE_mode0_std']:.6f}"
    )
    print(
        "scorer top1 loss/RMSE: "
        f"{summary_metrics['latent_endpoint_loss_scorer_top1_mean']:.6f} "
        f"± {summary_metrics['latent_endpoint_loss_scorer_top1_std']:.6f} / "
        f"{summary_metrics['latent_endpoint_RMSE_scorer_top1_mean']:.6f} "
        f"± {summary_metrics['latent_endpoint_RMSE_scorer_top1_std']:.6f}"
    )
    print(
        f"oracle all-{num_modes} loss/RMSE: "
        f"{summary_metrics['min_latent_endpoint_loss_all_mean']:.6f} "
        f"± {summary_metrics['min_latent_endpoint_loss_all_std']:.6f} / "
        f"{summary_metrics['min_latent_endpoint_RMSE_all_mean']:.6f} "
        f"± {summary_metrics['min_latent_endpoint_RMSE_all_std']:.6f}"
    )
    print(
        f"oracle kept-{min(args.max_guesses, num_modes)} loss/RMSE: "
        f"{summary_metrics['min_latent_endpoint_loss_kept_mean']:.6f} "
        f"± {summary_metrics['min_latent_endpoint_loss_kept_std']:.6f} / "
        f"{summary_metrics['min_latent_endpoint_RMSE_kept_mean']:.6f} "
        f"± {summary_metrics['min_latent_endpoint_RMSE_kept_std']:.6f}"
    )
    print(
        "说明：loss 是 intent×latent_dim 上的平方和均值；"
        "RMSE 是逐 latent 元素 MSE 的平方根，可与 val_latent_reg_rmse 直接比较。"
    )

    print("\n条件生成分布分解（使用筛选前全部 num_modes）")
    print(
        "sample mean loss/RMSE: "
        f"{summary_metrics['latent_sample_mean_loss_mean']:.6f} "
        f"± {summary_metrics['latent_sample_mean_loss_std']:.6f} / "
        f"{summary_metrics['latent_sample_mean_RMSE_mean']:.6f} "
        f"± {summary_metrics['latent_sample_mean_RMSE_std']:.6f}"
    )
    print(
        "within-scene variance/std: "
        f"{summary_metrics['latent_within_scene_variance_mean']:.6f} "
        f"± {summary_metrics['latent_within_scene_variance_std']:.6f} / "
        f"{summary_metrics['latent_within_scene_std_mean']:.6f} "
        f"± {summary_metrics['latent_within_scene_std_std']:.6f}"
    )
    print(
        "total sample loss/RMSE: "
        f"{summary_metrics['latent_total_sample_loss_mean']:.6f} "
        f"± {summary_metrics['latent_total_sample_loss_std']:.6f} / "
        f"{summary_metrics['latent_total_sample_RMSE_mean']:.6f} "
        f"± {summary_metrics['latent_total_sample_RMSE_std']:.6f}"
    )
    print(
        "decomposition max abs error: "
        f"{summary_metrics['latent_decomposition_max_abs_mean']:.6e} "
        f"± {summary_metrics['latent_decomposition_max_abs_std']:.6e}"
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"结果已保存到：{output_path.resolve()}")


if __name__ == "__main__":
    main()