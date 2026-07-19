from __future__ import annotations

import argparse
import contextlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datamodules import ArgoverseV2DataModule
from predictors import QCNetFM
from transforms import TargetBuilder

LEGACY_RESIDUAL_KEYWORDS = (
    "residual_mean", "residual_transport_scale", "residual_scale_head",
    "residual_mean_scale_head",
)
LEGACY_RESIDUAL_HPARAMS = (
    "residual_mean_loss_weight", "residual_delta_std_scale",
    "residual_transport_scale_min", "residual_transport_scale_max",
    "residual_scale_loss_weight", "latent_terminal_loss_weight",
)
STRATEGY_NAMES = (
    "top6_center_only",
    "top3_center_plus_one_fm_each",
    "top1_center_plus_five_fm",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "32":
        return contextlib.nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported precision: {precision}")


def apply_runtime_hparam_defaults(hparams: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "num_selector_layers": 1,
        "num_hist_tokens": 2,
        "num_map_tokens": 4,
        "num_agent_tokens": 2,
        "selector_hard_ce_weight": 1.0,
        "selector_rank_loss_weight": 1.0,
        "selector_rank_margin": 0.2,
        "selector_latent_loss_weight": 0.5,
        "selector_num_hard_negatives": 12,
        "selector_temperature_init": 1.0,
        "selector_temperature_min": 0.05,
        "selector_temperature_max": 10.0,
        "selector_interaction_scale_init": 0.1,
        "prototype_sampling_topk": 0,
        "residual_samples_per_prototype": 1,
        "cfg_scene_dropout": 0.15,
        "cfg_guidance_scale": 1.0,
        "num_center_layers": 2,
        "residual_center_loss_weight": 1.0,
        "residual_source_scale": 0.7,
    }
    for key, value in defaults.items():
        hparams.setdefault(key, value)
    for key in LEGACY_RESIDUAL_HPARAMS:
        hparams.pop(key, None)
    return hparams


def load_model_state(
    model: QCNetFM,
    state_dict: Mapping[str, torch.Tensor],
    allow_shape_mismatch: bool,
) -> None:
    legacy_keys = [
        key for key in state_dict
        if any(token in key for token in LEGACY_RESIDUAL_KEYWORDS)
    ]
    if legacy_keys:
        raise RuntimeError(
            "checkpoint 仍包含旧 residual_mean / transport 参数，不是当前 centered-residual checkpoint。"
            "示例字段：" + ", ".join(legacy_keys[:8])
        )
    current = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    mismatch = []
    for key, value in state_dict.items():
        if key not in current:
            mismatch.append((key, "unexpected"))
        elif tuple(current[key].shape) != tuple(value.shape):
            mismatch.append(
                (key, f"shape ckpt={tuple(value.shape)} model={tuple(current[key].shape)}")
            )
        else:
            compatible[key] = value
    missing = [key for key in current if key not in compatible]
    if (mismatch or missing) and not allow_shape_mismatch:
        lines = [
            "checkpoint 与当前模型代码不完全匹配。该脚本依赖 selector、center 和 FM，不能忽略这些权重。"
        ]
        lines += [f"unexpected/mismatch: {key} ({reason})" for key, reason in mismatch[:30]]
        lines += [f"missing: {key}" for key in missing[:30]]
        raise RuntimeError("\n".join(lines))
    if mismatch or missing:
        print(
            f"[WARNING] diagnostic load: mismatched={len(mismatch)}, missing={len(missing)}；"
            "本次结果可能不可靠。"
        )
    model.load_state_dict(compatible, strict=False)
    print(f"loaded compatible parameters: {len(compatible)} / checkpoint {len(state_dict)}")


def build_model(args: argparse.Namespace) -> Tuple[QCNetFM, Dict[str, Any]]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hparams = dict(checkpoint.get("hyper_parameters", {}))
    if not hparams:
        raise RuntimeError("checkpoint 中没有 hyper_parameters，无法重建 QCNetFM。")
    hparams = apply_runtime_hparam_defaults(hparams)
    hparams["residual_fm"] = True
    if args.prototype_bank_path is not None:
        hparams["prototype_bank_path"] = str(Path(args.prototype_bank_path).expanduser().resolve())
    if not hparams.get("prototype_bank_path"):
        raise ValueError("需要通过 checkpoint 或 --prototype_bank_path 指定 prototype bank。")
    model = QCNetFM(**hparams)
    state_dict = checkpoint.get("state_dict", checkpoint)
    load_model_state(model, state_dict, args.allow_shape_mismatch)
    device = resolve_device(args.device)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.prototype_selector is None:
        raise RuntimeError("当前模型没有 prototype_selector，无法进行三组 selector 推理实验。")
    if not hasattr(model.fm_decoder, "predict_residual_center"):
        raise RuntimeError("当前 fm_decoder 没有 residual center head。")
    if int(model.prototype_latents_centered_raw.size(0)) < 6:
        raise RuntimeError("prototype 数量小于6，无法执行 Top-6 center-only。")
    print("========== Model Loaded ==========")
    print("checkpoint:", args.checkpoint)
    print("prototype_bank_path:", hparams["prototype_bank_path"])
    print("prototype_num:", int(model.prototype_latents_centered_raw.size(0)))
    print("device:", device)
    print("trajectory_scale:", float(model.trajectory_scale))
    print("residual_source_scale:", float(model.residual_source_scale))
    print("checkpoint cfg_guidance_scale:", float(getattr(model, "cfg_guidance_scale", 1.0)))
    return model, hparams


def build_val_loader(
    args: argparse.Namespace,
    hparams: Dict[str, Any],
    model: QCNetFM,
) -> DataLoader:
    cfg = dict(hparams)
    # 本脚本只使用 selector 推理，不依赖 prototype assignment sidecar。
    cfg.pop("prototype_assignment_dir", None)
    cfg.pop("prototype_assignment_strict", None)
    cfg.pop("assignment_cache_size", None)
    overrides = {
        "root": args.root,
        "train_batch_size": args.batch_size,
        "val_batch_size": args.batch_size,
        "test_batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers and args.num_workers > 0,
        "shuffle": False,
        "train_raw_dir": args.train_raw_dir,
        "val_raw_dir": args.val_raw_dir,
        "test_raw_dir": args.test_raw_dir,
        "train_processed_dir": args.train_processed_dir,
        "val_processed_dir": args.val_processed_dir,
        "test_processed_dir": args.test_processed_dir,
        "train_transform": TargetBuilder(model.num_historical_steps, model.num_future_steps),
        "val_transform": TargetBuilder(model.num_historical_steps, model.num_future_steps),
        "test_transform": None,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    if not cfg.get("root"):
        raise ValueError("缺少数据集 root，请传入 --root。")
    datamodule = ArgoverseV2DataModule(**cfg)
    datamodule.setup(stage="fit")
    if getattr(datamodule, "val_dataset", None) is None:
        raise RuntimeError("DataModule.setup(stage='fit') 后没有 val_dataset。")
    print("========== Validation Data ==========")
    print("root:", cfg["root"])
    print("val_processed_dir:", cfg.get("val_processed_dir"))
    print("val dataset processed_dir:", datamodule.val_dataset.processed_dir)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
    return DataLoader(datamodule.val_dataset, **loader_kwargs)


def squeeze_latent_decoder_output(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(1) if x.ndim == 4 and x.size(1) == 1 else x


def slice_scene_conditions(
    scene_conditions: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    n = int(mask.numel())
    result: Dict[str, torch.Tensor] = {}
    for key, value in scene_conditions.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == n:
            result[key] = value[mask]
        else:
            result[key] = value
    return result


def select_top6_prototypes(
    model: QCNetFM,
    scene_conditions: Mapping[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selector_outputs = model._compute_selector_outputs(scene_conditions=scene_conditions)
    logits = selector_outputs["logits"]
    top_scores, top_indices = torch.topk(logits, k=6, dim=-1)
    proto_bank = model.prototype_latents_centered_raw.to(
        device=logits.device, dtype=logits.dtype
    )
    prototype_modes = proto_bank[top_indices]
    if bool(getattr(model, "use_prototype_local_std", True)):
        std_bank = model.prototype_residual_std_population.to(
            device=logits.device, dtype=logits.dtype
        )
        std_modes = std_bank[top_indices].clamp_min(1e-4)
    else:
        global_std = model.z_std.to(device=logits.device, dtype=logits.dtype)
        global_std = global_std.view(1, 1, 1, -1)
        std_modes = global_std.expand_as(prototype_modes).contiguous()
    probabilities = torch.softmax(logits, dim=-1)
    top_probabilities = probabilities.gather(1, top_indices)
    return top_indices, prototype_modes, std_modes, top_probabilities


def predict_center_modes(
    model: QCNetFM,
    scene_conditions: Mapping[str, torch.Tensor],
    prototype_modes: torch.Tensor,
    prototype_indices: torch.Tensor,
) -> torch.Tensor:
    centers = model.fm_decoder.predict_residual_center(
        scene_conditions=scene_conditions,
        prototype_latent=prototype_modes,
        prototype_index=prototype_indices,
    )
    if centers.shape != prototype_modes.shape:
        raise RuntimeError(
            f"center/prototype shape 不一致：center={tuple(centers.shape)}, "
            f"prototype={tuple(prototype_modes.shape)}"
        )
    return centers


def decode_latent_modes(
    latent_decoder: torch.nn.Module,
    latent_modes: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    for mode_idx in range(latent_modes.size(1)):
        outputs.append(
            squeeze_latent_decoder_output(latent_decoder(latent_modes[:, mode_idx]))
        )
    return torch.stack(outputs, dim=1)


def make_null_scene_conditions(
    decoder: torch.nn.Module,
    scene_conditions: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    if not hasattr(decoder, "make_null_scene_conditions"):
        raise RuntimeError("当前 decoder 不支持 CFG；请使用 --cfg_guidance_scale 1.0。")
    return decoder.make_null_scene_conditions(scene_conditions)


def guided_velocity(
    decoder: torch.nn.Module,
    scene_conditions: Mapping[str, torch.Tensor],
    null_scene_conditions: Optional[Mapping[str, torch.Tensor]],
    x_t: torch.Tensor,
    t: torch.Tensor,
    prototype_latent: torch.Tensor,
    prototype_index: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    w = float(guidance_scale)
    if abs(w - 1.0) < 1e-8:
        return decoder._forward_core(
            scene_conditions, x_t, t,
            prototype_latent=prototype_latent,
            prototype_index=prototype_index,
        )
    if hasattr(decoder, "_prototype_cfg_velocity"):
        return decoder._prototype_cfg_velocity(
            scene_conditions=scene_conditions,
            null_scene_conditions=null_scene_conditions,
            x_t=x_t,
            t=t,
            prototype_latent=prototype_latent,
            prototype_index=prototype_index,
            guidance_scale=w,
        )
    if null_scene_conditions is None:
        raise RuntimeError("CFG 缺少 null_scene_conditions。")
    v_uncond = decoder._forward_core(
        null_scene_conditions, x_t, t,
        prototype_latent=prototype_latent,
        prototype_index=prototype_index,
    )
    v_cond = decoder._forward_core(
        scene_conditions, x_t, t,
        prototype_latent=prototype_latent,
        prototype_index=prototype_index,
    )
    return v_uncond + w * (v_cond - v_uncond)


@torch.no_grad()
def sample_fm_modes(
    model: QCNetFM,
    scene_conditions: Mapping[str, torch.Tensor],
    prototype_modes: torch.Tensor,
    std_modes: torch.Tensor,
    prototype_indices: torch.Tensor,
    center_modes: torch.Tensor,
    num_steps: int,
    solver: str,
    cfg_guidance_scale: float,
    source_std_multiplier: float,
) -> torch.Tensor:
    decoder = model.fm_decoder
    device = scene_conditions["c_hist"].device
    dtype = scene_conditions["c_hist"].dtype
    prototype_modes = prototype_modes.to(device=device, dtype=dtype)
    std_modes = std_modes.to(device=device, dtype=dtype).clamp_min(1e-4)
    prototype_indices = prototype_indices.to(device=device, dtype=torch.long)
    center_modes = center_modes.to(device=device, dtype=dtype)
    if prototype_modes.ndim != 4:
        raise ValueError(f"prototype_modes 应为 [N,M,1,D]，实际={tuple(prototype_modes.shape)}")
    if std_modes.shape != prototype_modes.shape or center_modes.shape != prototype_modes.shape:
        raise ValueError(
            f"prototype/std/center shape 不一致：{tuple(prototype_modes.shape)}, "
            f"{tuple(std_modes.shape)}, {tuple(center_modes.shape)}"
        )
    if prototype_indices.shape != prototype_modes.shape[:2]:
        raise ValueError(
            f"prototype_indices shape 错误：{tuple(prototype_indices.shape)} vs "
            f"{tuple(prototype_modes.shape[:2])}"
        )
    if num_steps <= 0:
        raise ValueError("num_steps 必须 > 0。")
    if source_std_multiplier <= 0.0:
        raise ValueError("source_std_multiplier 必须 > 0。")
    if cfg_guidance_scale < 0.0:
        raise ValueError("cfg_guidance_scale 必须 >= 0。")
    use_cfg = abs(float(cfg_guidance_scale) - 1.0) > 1e-8
    null_scene_conditions = (
        make_null_scene_conditions(decoder, scene_conditions) if use_cfg else None
    )
    residual_source_scale = float(model.residual_source_scale)
    dt = 1.0 / float(num_steps)
    t_grid = torch.linspace(0.0, 1.0 - dt, num_steps, device=device)
    n = prototype_modes.size(0)
    t_cur = torch.empty(n, device=device)
    t_next = torch.empty(n, device=device)
    decoded_modes = []
    for mode_idx in range(prototype_modes.size(1)):
        prototype = prototype_modes[:, mode_idx]
        center = center_modes[:, mode_idx]
        local_std = std_modes[:, mode_idx]
        prototype_index = prototype_indices[:, mode_idx]
        x_t = (
            torch.randn_like(prototype)
            * local_std
            * residual_source_scale
            * float(source_std_multiplier)
        )
        for t_value in t_grid:
            t_cur.fill_(t_value)
            v1 = guided_velocity(
                decoder=decoder,
                scene_conditions=scene_conditions,
                null_scene_conditions=null_scene_conditions,
                x_t=x_t,
                t=t_cur,
                prototype_latent=prototype,
                prototype_index=prototype_index,
                guidance_scale=cfg_guidance_scale,
            )
            if solver == "euler":
                x_t = x_t + dt * v1
                continue
            t_next.fill_(t_value + dt)
            x_euler = x_t + dt * v1
            v2 = guided_velocity(
                decoder=decoder,
                scene_conditions=scene_conditions,
                null_scene_conditions=null_scene_conditions,
                x_t=x_euler,
                t=t_next,
                prototype_latent=prototype,
                prototype_index=prototype_index,
                guidance_scale=cfg_guidance_scale,
            )
            x_t = x_t + 0.5 * dt * (v1 + v2)
        decoded_modes.append(
            squeeze_latent_decoder_output(
                model.latent_decoder(prototype + center + x_t)
            )
        )
    return torch.stack(decoded_modes, dim=1)


def interleave_center_and_fm(
    center_trajectories: torch.Tensor,
    fm_trajectories: torch.Tensor,
) -> torch.Tensor:
    if center_trajectories.shape != fm_trajectories.shape:
        raise ValueError(
            f"center/fm trajectory shape 不一致：{tuple(center_trajectories.shape)} vs "
            f"{tuple(fm_trajectories.shape)}"
        )
    n, modes, steps, dim = center_trajectories.shape
    return torch.stack((center_trajectories, fm_trajectories), dim=2).reshape(
        n, modes * 2, steps, dim
    )


def compute_minade_minfde(
    trajectories_m: torch.Tensor,
    target_m: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    trajectories_xy = trajectories_m[..., :2].float()
    target_xy = target_m[..., :2].float()
    valid_mask = valid_mask.bool()
    n, _, steps, _ = trajectories_xy.shape
    distance = torch.norm(trajectories_xy - target_xy.unsqueeze(1), dim=-1)
    mask = valid_mask.unsqueeze(1).to(distance.dtype)
    ade = (distance * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    min_ade = ade.min(dim=1).values
    last_valid_idx = (
        steps - 1 - valid_mask.flip(dims=[-1]).long().argmax(dim=-1)
    ).clamp(0, steps - 1)
    fde = distance[torch.arange(n, device=distance.device), :, last_valid_idx]
    min_fde = fde.min(dim=1).values
    return min_ade, min_fde


def compute_pairwise_diversity(
    trajectories_m: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    trajectories_xy = trajectories_m[..., :2].float()
    n, modes, steps, _ = trajectories_xy.shape
    if modes < 2:
        zero = torch.zeros(n, device=trajectories_xy.device)
        return zero, zero
    diff = trajectories_xy[:, :, None] - trajectories_xy[:, None, :]
    distance = torch.norm(diff, dim=-1)
    mask = valid_mask[:, None, None, :].to(distance.dtype)
    pair_ade = (distance * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    last_valid_idx = (
        steps - 1 - valid_mask.flip(dims=[-1]).long().argmax(dim=-1)
    ).clamp(0, steps - 1)
    pair_fde = distance[
        torch.arange(n, device=distance.device),
        :,
        :,
        last_valid_idx,
    ]
    upper = torch.triu(
        torch.ones(modes, modes, dtype=torch.bool, device=distance.device), diagonal=1
    )
    return pair_ade[:, upper].mean(dim=-1), pair_fde[:, upper].mean(dim=-1)


def new_accumulator() -> Dict[str, float]:
    return {
        "num_agents": 0.0,
        "minade_sum": 0.0,
        "minfde_sum": 0.0,
        "miss_sum": 0.0,
        "pairwise_ade_sum": 0.0,
        "pairwise_fde_sum": 0.0,
    }


def update_accumulator(
    accumulator: Dict[str, float],
    trajectories_m: torch.Tensor,
    target_m: torch.Tensor,
    valid_mask: torch.Tensor,
    miss_threshold: float,
) -> None:
    min_ade, min_fde = compute_minade_minfde(
        trajectories_m=trajectories_m,
        target_m=target_m,
        valid_mask=valid_mask,
    )
    pair_ade, pair_fde = compute_pairwise_diversity(
        trajectories_m=trajectories_m,
        valid_mask=valid_mask,
    )
    n = int(trajectories_m.size(0))
    accumulator["num_agents"] += n
    accumulator["minade_sum"] += float(min_ade.sum())
    accumulator["minfde_sum"] += float(min_fde.sum())
    accumulator["miss_sum"] += float((min_fde > miss_threshold).float().sum())
    accumulator["pairwise_ade_sum"] += float(pair_ade.sum())
    accumulator["pairwise_fde_sum"] += float(pair_fde.sum())


def finalize_accumulator(accumulator: Mapping[str, float]) -> Dict[str, float]:
    n = max(float(accumulator["num_agents"]), 1.0)
    return {
        "num_eval_agents": int(accumulator["num_agents"]),
        "minADE": float(accumulator["minade_sum"] / n),
        "minFDE": float(accumulator["minfde_sum"] / n),
        "MR": float(accumulator["miss_sum"] / n),
        "pairwise_ADE_diversity": float(accumulator["pairwise_ade_sum"] / n),
        "pairwise_FDE_diversity": float(accumulator["pairwise_fde_sum"] / n),
    }


@torch.no_grad()
def evaluate_three_strategies(
    model: QCNetFM,
    val_loader: DataLoader,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    device = resolve_device(args.device)
    trajectory_scale = float(model.trajectory_scale)
    cfg_guidance_scale = float(
        getattr(model, "cfg_guidance_scale", 1.0)
        if args.cfg_guidance_scale is None
        else args.cfg_guidance_scale
    )
    accumulators = {name: new_accumulator() for name in STRATEGY_NAMES}
    selector_entropy_sum = 0.0
    selector_top1_mass_sum = 0.0
    selector_top3_mass_sum = 0.0
    selector_top6_mass_sum = 0.0
    selector_agents = 0
    pbar = tqdm(val_loader, desc="Compare 3 center/FM allocation strategies")
    for batch_idx, data in enumerate(pbar):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        data = data.to(device)
        if isinstance(data, Batch):
            data["agent"]["av_index"] += data["agent"]["ptr"][:-1]
        target_norm_all = (
            data["agent"]["target"][..., :model.output_dim]
            / trajectory_scale
        )
        predict_mask_all = data["agent"]["predict_mask"][
            :, model.num_historical_steps:
        ].bool()
        category = data["agent"]["category"]
        eval_mask = category.eq(3) & predict_mask_all.any(dim=-1)
        if not eval_mask.any():
            continue
        with autocast_context(device, args.precision):
            scene_enc = model.encoder(data)
            _, scene_conditions_all = model.fm_decoder.build_scene_conditions(data, scene_enc)
            scene_conditions = slice_scene_conditions(scene_conditions_all, eval_mask)
            top_indices, prototype_modes, std_modes, top_probabilities = (
                select_top6_prototypes(model, scene_conditions)
            )
            center_modes = predict_center_modes(
                model=model,
                scene_conditions=scene_conditions,
                prototype_modes=prototype_modes,
                prototype_indices=top_indices,
            )
            center_trajectories = decode_latent_modes(
                model.latent_decoder,
                prototype_modes + center_modes,
            )
            # Strategy A: Top-6 prototypes, one deterministic center trajectory each.
            strategy_a = center_trajectories
            # Strategy B: Top-3 prototypes, each contributes center-only + one FM sample.
            fm_top3 = sample_fm_modes(
                model=model,
                scene_conditions=scene_conditions,
                prototype_modes=prototype_modes[:, :3],
                std_modes=std_modes[:, :3],
                prototype_indices=top_indices[:, :3],
                center_modes=center_modes[:, :3],
                num_steps=args.num_steps,
                solver=args.solver,
                cfg_guidance_scale=cfg_guidance_scale,
                source_std_multiplier=args.source_std_multiplier,
            )
            strategy_b = interleave_center_and_fm(
                center_trajectories[:, :3], fm_top3
            )
            # Strategy C: Top-1 prototype, one center-only + five independent FM samples.
            top1_prototype = prototype_modes[:, :1].expand(-1, 5, -1, -1).contiguous()
            top1_std = std_modes[:, :1].expand_as(top1_prototype).contiguous()
            top1_indices = top_indices[:, :1].expand(-1, 5).contiguous()
            top1_center = center_modes[:, :1].expand_as(top1_prototype).contiguous()
            fm_top1_five = sample_fm_modes(
                model=model,
                scene_conditions=scene_conditions,
                prototype_modes=top1_prototype,
                std_modes=top1_std,
                prototype_indices=top1_indices,
                center_modes=top1_center,
                num_steps=args.num_steps,
                solver=args.solver,
                cfg_guidance_scale=cfg_guidance_scale,
                source_std_multiplier=args.source_std_multiplier,
            )
            strategy_c = torch.cat(
                (center_trajectories[:, :1], fm_top1_five), dim=1
            )
            probabilities = top_probabilities
            entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=-1)
        target_m = target_norm_all[eval_mask].float() * trajectory_scale
        valid_mask = predict_mask_all[eval_mask]
        strategy_outputs = {
            "top6_center_only": strategy_a.float() * trajectory_scale,
            "top3_center_plus_one_fm_each": strategy_b.float() * trajectory_scale,
            "top1_center_plus_five_fm": strategy_c.float() * trajectory_scale,
        }
        for name, trajectories_m in strategy_outputs.items():
            if trajectories_m.size(1) != 6:
                raise RuntimeError(
                    f"{name} 输出 mode 数不是6：{trajectories_m.size(1)}"
                )
            update_accumulator(
                accumulators[name],
                trajectories_m=trajectories_m,
                target_m=target_m,
                valid_mask=valid_mask,
                miss_threshold=args.miss_threshold,
            )
        n = int(eval_mask.sum())
        selector_entropy_sum += float(entropy.sum())
        selector_top1_mass_sum += float(probabilities[:, :1].sum())
        selector_top3_mass_sum += float(probabilities[:, :3].sum())
        selector_top6_mass_sum += float(probabilities[:, :6].sum())
        selector_agents += n
        current = {
            name: accumulators[name]["minade_sum"]
            / max(accumulators[name]["num_agents"], 1.0)
            for name in STRATEGY_NAMES
        }
        pbar.set_postfix(
            A=f"{current['top6_center_only']:.3f}",
            B=f"{current['top3_center_plus_one_fm_each']:.3f}",
            C=f"{current['top1_center_plus_five_fm']:.3f}",
        )
    if selector_agents == 0:
        raise RuntimeError("没有找到可评估的 focal agents。")
    results = {
        name: finalize_accumulator(accumulators[name])
        for name in STRATEGY_NAMES
    }
    ranking = sorted(
        (
            {"strategy": name, "minADE": metrics["minADE"]}
            for name, metrics in results.items()
        ),
        key=lambda item: item["minADE"],
    )
    return {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "prototype_bank_path": str(
            Path(model.prototype_bank_path).expanduser().resolve()
            if isinstance(model.prototype_bank_path, str)
            else model.prototype_bank_path
        ),
        "prototype_num": int(model.prototype_latents_centered_raw.size(0)),
        "num_steps": args.num_steps,
        "solver": args.solver,
        "cfg_guidance_scale": cfg_guidance_scale,
        "residual_source_scale_from_checkpoint": float(model.residual_source_scale),
        "source_std_multiplier": args.source_std_multiplier,
        "effective_source_std_scale": float(model.residual_source_scale)
        * float(args.source_std_multiplier),
        "miss_threshold_m": args.miss_threshold,
        "selector_top6_truncated_entropy": selector_entropy_sum / selector_agents,
        "selector_top1_probability_mass": selector_top1_mass_sum / selector_agents,
        "selector_top3_probability_mass": selector_top3_mass_sum / selector_agents,
        "selector_top6_probability_mass": selector_top6_mass_sum / selector_agents,
        "strategies": results,
        "ranking_by_minADE": ranking,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare three six-mode inference allocations for prototype + residual center + FM: "
            "Top6 centers, Top3*(center+FM), and Top1 center+5 FM samples."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prototype_bank_path", type=str, default=None)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--train_raw_dir", type=str, default=None)
    parser.add_argument("--val_raw_dir", type=str, default=None)
    parser.add_argument("--test_raw_dir", type=str, default=None)
    parser.add_argument("--train_processed_dir", type=str, default=None)
    parser.add_argument("--val_processed_dir", type=str, required=True)
    parser.add_argument("--test_processed_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument(
        "--pin_memory", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--persistent_workers", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--precision", choices=["32", "bf16", "fp16"], default="bf16"
    )
    parser.add_argument("--num_steps", type=int, default=2)
    parser.add_argument("--solver", choices=["euler", "heun"], default="euler")
    parser.add_argument(
        "--cfg_guidance_scale",
        type=float,
        default=None,
        help="默认读取 checkpoint；例如传1.1进行CFG评估。",
    )
    parser.add_argument(
        "--source_std_multiplier",
        type=float,
        default=1.0,
        help="额外 source 尺度消融系数；正式比较保持1.0。",
    )
    parser.add_argument("--miss_threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument(
        "--allow_shape_mismatch",
        action="store_true",
        help="仅用于排查加载问题；正式评估不要开启。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_steps <= 0:
        raise ValueError("num_steps 必须 > 0。")
    set_seed(args.seed)
    model, hparams = build_model(args)
    val_loader = build_val_loader(args, hparams, model)
    result = evaluate_three_strategies(model, val_loader, args)
    print("\n========== Three Six-Mode Strategy Comparison ==========")
    for name in STRATEGY_NAMES:
        metrics = result["strategies"][name]
        print(
            f"{name}: minADE={metrics['minADE']:.6f}, "
            f"minFDE={metrics['minFDE']:.6f}, MR={metrics['MR']:.6f}, "
            f"pairADE={metrics['pairwise_ADE_diversity']:.6f}, "
            f"pairFDE={metrics['pairwise_FDE_diversity']:.6f}"
        )
    print("ranking_by_minADE:")
    for rank, item in enumerate(result["ranking_by_minADE"], start=1):
        print(f"  {rank}. {item['strategy']}: {item['minADE']:.6f}")
    print(
        "selector probability mass: "
        f"Top1={result['selector_top1_probability_mass']:.6f}, "
        f"Top3={result['selector_top3_probability_mass']:.6f}, "
        f"Top6={result['selector_top6_probability_mass']:.6f}"
    )
    if args.output_json is not None:
        output = Path(args.output_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("saved:", output)


if __name__ == "__main__":
    main()