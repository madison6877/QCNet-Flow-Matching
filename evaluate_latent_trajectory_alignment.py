#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check whether FM endpoint latent distance is aligned with decoded trajectory distance.

For every focal agent and every generated mode, this script computes:

    latent_std_RMSE:
        RMSE between generated endpoint latent and the standardized GT VAE latent.
        This is the geometry directly optimized by the current latent FM loss.

    latent_raw_RMSE:
        RMSE after undoing latent standardization.

    trajectory_ADE_m:
        Mean Euclidean trajectory error over valid future timesteps, in meters.

    trajectory_FDE_m:
        Euclidean error at the last valid future timestep, in meters.

It then reports:

1. Pooled Pearson/Spearman correlations across all agent-mode pairs.
2. Per-agent, within-mode Pearson/Spearman correlations.
3. Whether the latent-nearest mode is also the ADE/FDE-best mode.
4. ADE/FDE regret when selecting the latent-nearest mode:
       ADE(latent-nearest) - min_ADE
       FDE(latent-nearest) - min_FDE
5. Hit@K:
       whether the ADE-best mode lies in the K latent-nearest candidates.
6. Per-latent-dimension squared-error correlation with ADE/FDE.

The trajectories and endpoint latents are produced in the same Heun rollout, so
candidate indices are exactly aligned.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch

from datamodules import ArgoverseV2DataModule
from predictors import QCNetFM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate alignment between FM endpoint latent distance and "
            "decoded trajectory ADE/FDE on AV2 focal agents."
        )
    )
    parser.add_argument("--ckpt", required=True)

    # Data paths. CLI values override checkpoint hyperparameters.
    parser.add_argument("--root", default=None)
    parser.add_argument("--train_raw_dir", default=None)
    parser.add_argument("--train_processed_dir", default=None)
    parser.add_argument("--val_raw_dir", default=None)
    parser.add_argument("--val_processed_dir", default=None)
    parser.add_argument("--test_raw_dir", default=None)
    parser.add_argument("--test_processed_dir", default=None)

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
        default=64,
    )
    parser.add_argument(
        "--fm_num_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="0 means the complete validation set.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--trajectory_scale",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--focal_category",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--hit_ks",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20],
    )

    parser.add_argument(
        "--output_json",
        default=None,
    )
    parser.add_argument(
        "--output_agent_csv",
        default=None,
        help=(
            "Optional per-focal-agent diagnostic CSV. "
            "One row per evaluated focal agent."
        ),
    )
    parser.add_argument(
        "--allow_shape_mismatch",
        action="store_true",
        help=(
            "Diagnostic only. Skip checkpoint keys whose shape does not match. "
            "Do not use for formal evaluation."
        ),
    )
    return parser.parse_args()


def torch_load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        obj = torch.load(
            path,
            map_location="cpu",
        )
    if not isinstance(obj, dict):
        raise TypeError(
            f"checkpoint content must be dict, got {type(obj)}"
        )
    return obj


def as_plain_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "items"):
        return dict(obj.items())
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    raise TypeError(
        f"Cannot convert hyper_parameters to dict: {type(obj)}"
    )


def load_model(
    ckpt_path: Path,
    device: torch.device,
    allow_shape_mismatch: bool,
) -> Tuple[QCNetFM, Dict[str, Any]]:
    checkpoint = torch_load_checkpoint(
        ckpt_path
    )
    hparams = as_plain_dict(
        checkpoint.get(
            "hyper_parameters",
            {},
        )
    )
    if not hparams:
        raise RuntimeError(
            "checkpoint is missing hyper_parameters"
        )

    model = QCNetFM(**hparams)
    state_dict = checkpoint.get(
        "state_dict",
        checkpoint,
    )

    if not allow_shape_mismatch:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    else:
        current = model.state_dict()
        compatible: Dict[str, torch.Tensor] = {}
        mismatched: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []

        for key, value in state_dict.items():
            if key not in current:
                continue
            if tuple(value.shape) != tuple(current[key].shape):
                mismatched.append(
                    (
                        key,
                        tuple(value.shape),
                        tuple(current[key].shape),
                    )
                )
                continue
            compatible[key] = value

        missing, unexpected = model.load_state_dict(
            compatible,
            strict=False,
        )

        print(
            "\n[WARNING] --allow_shape_mismatch is enabled."
        )
        print(
            f"compatible keys={len(compatible):,}, "
            f"mismatched={len(mismatched):,}, "
            f"missing={len(missing):,}, "
            f"unexpected={len(unexpected):,}"
        )
        for item in mismatched[:20]:
            print(
                f"  skipped {item[0]}: "
                f"ckpt={item[1]} current={item[2]}"
            )
        print(
            "[WARNING] Formal metrics are invalid if a key FM/VAE head was skipped.\n"
        )

    model.to(device)
    model.eval()
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
        "train_raw_dir": args.train_raw_dir,
        "train_processed_dir": args.train_processed_dir,
        "val_raw_dir": args.val_raw_dir,
        "val_processed_dir": args.val_processed_dir,
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
    cfg.setdefault("num_workers", 4)
    cfg.setdefault("pin_memory", torch.cuda.is_available())
    cfg.setdefault("persistent_workers", False)
    cfg["shuffle"] = False

    if cfg.get("persistent_workers") and int(cfg["num_workers"]) <= 0:
        cfg["persistent_workers"] = False

    if not cfg.get("root"):
        raise ValueError(
            "Dataset root is missing. Pass --root or use a checkpoint "
            "whose hyper_parameters contain root."
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
            f"[warning] {precision} autocast is disabled on {device}; using fp32.",
            file=sys.stderr,
        )
        return contextlib.nullcontext()

    dtype = (
        torch.bfloat16
        if precision == "bf16"
        else torch.float16
    )
    return torch.autocast(
        device_type="cuda",
        dtype=dtype,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def sample_trajectories_and_latents(
    model: QCNetFM,
    data,
    scene_enc: Mapping[str, torch.Tensor],
    num_modes: int,
    num_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        trajectories_normalized:
            [N_agent, M, T_future, output_dim]

        latent_samples_std:
            [N_agent, M, num_intents, latent_dim]

    Latent and trajectory candidate index m are generated in the same rollout.
    """
    if num_modes <= 0:
        raise ValueError(
            f"num_modes must be positive, got {num_modes}"
        )
    if num_steps <= 0:
        raise ValueError(
            f"num_steps must be positive, got {num_steps}"
        )

    decoder = model.fm_decoder
    ctx = decoder._build_graph_context(
        data,
        scene_enc,
    )

    num_agents = ctx["pos_m"].size(0)
    device = ctx["pos_m"].device

    dt = 1.0 / float(num_steps)
    t_grid = torch.linspace(
        0.0,
        1.0 - dt,
        num_steps,
        device=device,
    )

    t_cur = torch.empty(
        num_agents,
        device=device,
    )
    t_next = torch.empty(
        num_agents,
        device=device,
    )

    latent_outputs: List[torch.Tensor] = []
    trajectory_outputs: List[torch.Tensor] = []

    for _ in range(num_modes):
        x_t = torch.randn(
            num_agents,
            decoder.num_intents,
            decoder.latent_dim,
            device=device,
        )

        for t_value in t_grid:
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
            x_t = x_t + 0.5 * (v1 + v2) * dt

        latent_outputs.append(
            x_t.float()
        )
        trajectory_outputs.append(
            model.latent_decoder(x_t).float()
        )

    latent_samples_std = torch.stack(
        latent_outputs,
        dim=1,
    )
    trajectories_normalized = torch.stack(
        trajectory_outputs,
        dim=1,
    )
    return trajectories_normalized, latent_samples_std


def trajectory_distance_per_mode(
    trajectories_m: torch.Tensor,
    target_m: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        trajectories_m: [N, M, T, D]
        target_m:       [N, T, D]
        valid_mask:     [N, T]

    Returns:
        ade_per_mode: [N, M]
        fde_per_mode: [N, M]
    """
    if trajectories_m.ndim != 4:
        raise ValueError(
            f"trajectories must be [N,M,T,D], got {tuple(trajectories_m.shape)}"
        )
    if target_m.ndim != 3:
        raise ValueError(
            f"target must be [N,T,D], got {tuple(target_m.shape)}"
        )
    if valid_mask.ndim != 2:
        raise ValueError(
            f"valid_mask must be [N,T], got {tuple(valid_mask.shape)}"
        )

    n, modes, steps, dims = trajectories_m.shape
    if target_m.shape != (n, steps, dims):
        raise ValueError(
            "trajectory/target shape mismatch: "
            f"pred={tuple(trajectories_m.shape)}, "
            f"target={tuple(target_m.shape)}"
        )
    if valid_mask.shape != (n, steps):
        raise ValueError(
            "mask shape mismatch: "
            f"mask={tuple(valid_mask.shape)}, expected={(n, steps)}"
        )

    valid_mask = valid_mask.bool()
    valid_counts = valid_mask.sum(
        dim=-1
    )
    if (valid_counts == 0).any():
        raise ValueError(
            "An evaluated focal agent has no valid future timestep."
        )

    displacement = torch.linalg.vector_norm(
        trajectories_m.float()
        - target_m.float().unsqueeze(1),
        dim=-1,
    )
    mask_f = valid_mask[:, None, :].to(
        displacement.dtype
    )

    ade_per_mode = (
        displacement * mask_f
    ).sum(dim=-1) / valid_counts[:, None].to(
        displacement.dtype
    )

    time_index = torch.arange(
        steps,
        device=valid_mask.device,
    ).view(1, steps)
    last_valid_index = time_index.masked_fill(
        ~valid_mask,
        -1,
    ).max(dim=-1).values

    pred_index = last_valid_index[
        :,
        None,
        None,
        None,
    ].expand(
        n,
        modes,
        1,
        dims,
    )
    pred_endpoint = trajectories_m.gather(
        dim=2,
        index=pred_index,
    ).squeeze(2)

    gt_index = last_valid_index[
        :,
        None,
        None,
    ].expand(
        n,
        1,
        dims,
    )
    gt_endpoint = target_m.gather(
        dim=1,
        index=gt_index,
    ).squeeze(1)

    fde_per_mode = torch.linalg.vector_norm(
        pred_endpoint
        - gt_endpoint.unsqueeze(1),
        dim=-1,
    )
    return ade_per_mode, fde_per_mode


def latent_distance_per_mode(
    latent_samples_std: torch.Tensor,
    latent_target_std: torch.Tensor,
    z_std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        latent_std_rmse: [N,M]
        latent_raw_rmse: [N,M]
        latent_std_sq_error_per_dim: [N,M,L]
    """
    if latent_samples_std.ndim != 4:
        raise ValueError(
            "latent_samples_std must be [N,M,I,D], got "
            f"{tuple(latent_samples_std.shape)}"
        )
    if latent_target_std.ndim != 3:
        raise ValueError(
            "latent_target_std must be [N,I,D], got "
            f"{tuple(latent_target_std.shape)}"
        )

    if (
        latent_samples_std.shape[0]
        != latent_target_std.shape[0]
        or latent_samples_std.shape[2:]
        != latent_target_std.shape[1:]
    ):
        raise ValueError(
            "latent sample/target shape mismatch: "
            f"samples={tuple(latent_samples_std.shape)}, "
            f"target={tuple(latent_target_std.shape)}"
        )

    delta_std = (
        latent_samples_std.float()
        - latent_target_std.float().unsqueeze(1)
    )
    sq_error_std = delta_std.pow(2)

    latent_std_rmse = sq_error_std.mean(
        dim=(-1, -2)
    ).sqrt()

    # z_mean cancels in pairwise differences; only scale is needed.
    scale = z_std.float().view(
        1,
        1,
        latent_target_std.size(1),
        latent_target_std.size(2),
    )
    delta_raw = delta_std * scale
    latent_raw_rmse = delta_raw.pow(2).mean(
        dim=(-1, -2)
    ).sqrt()

    per_dim_sq_error = sq_error_std.flatten(
        start_dim=2
    )
    return (
        latent_std_rmse,
        latent_raw_rmse,
        per_dim_sq_error,
    )


def rank_rows(values: torch.Tensor) -> torch.Tensor:
    """
    Ordinal row-wise ranks. Exact ties are extremely rare for continuous errors.
    Smallest value receives rank 0.
    """
    order = values.argsort(
        dim=1,
        stable=True,
    )
    ranks = torch.empty_like(
        values,
        dtype=torch.float32,
    )
    rank_values = torch.arange(
        values.size(1),
        device=values.device,
        dtype=torch.float32,
    ).view(1, -1).expand_as(values)
    ranks.scatter_(
        dim=1,
        index=order,
        src=rank_values,
    )
    return ranks


def rowwise_pearson(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.shape != y.shape:
        raise ValueError(
            f"rowwise correlation shape mismatch: {x.shape} vs {y.shape}"
        )

    x = x.float()
    y = y.float()
    x_centered = x - x.mean(
        dim=1,
        keepdim=True,
    )
    y_centered = y - y.mean(
        dim=1,
        keepdim=True,
    )

    numerator = (
        x_centered * y_centered
    ).sum(dim=1)
    denominator = (
        x_centered.pow(2).sum(dim=1)
        * y_centered.pow(2).sum(dim=1)
    ).clamp_min(eps).sqrt()

    valid = denominator > math.sqrt(eps)
    correlation = torch.full_like(
        numerator,
        float("nan"),
    )
    correlation[valid] = (
        numerator[valid]
        / denominator[valid]
    )
    return correlation, valid


def rowwise_spearman(
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return rowwise_pearson(
        rank_rows(x),
        rank_rows(y),
    )


def safe_quantiles(
    values: torch.Tensor,
    quantiles: Sequence[float] = (
        0.0,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.0,
    ),
) -> Dict[str, Optional[float]]:
    values = values.detach().cpu().float()
    values = values[
        torch.isfinite(values)
    ]
    if values.numel() == 0:
        return {
            f"{q:.2f}": None
            for q in quantiles
        }

    q_tensor = torch.tensor(
        list(quantiles),
        dtype=torch.float32,
    )
    result = torch.quantile(
        values,
        q_tensor,
    )
    return {
        f"{q:.2f}": float(value)
        for q, value in zip(
            quantiles,
            result.tolist(),
        )
    }


def scalar_summary(
    values: torch.Tensor,
) -> Dict[str, Any]:
    values = values.detach().cpu().float()
    finite = values[
        torch.isfinite(values)
    ]
    if finite.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "quantiles": safe_quantiles(
                finite
            ),
        }

    return {
        "count": int(
            finite.numel()
        ),
        "mean": float(
            finite.mean().item()
        ),
        "std": float(
            finite.std(
                unbiased=False
            ).item()
        ),
        "median": float(
            finite.median().item()
        ),
        "quantiles": safe_quantiles(
            finite
        ),
    }


def pearson_1d(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    x = x.detach().cpu().double()
    y = y.detach().cpu().double()

    finite = (
        torch.isfinite(x)
        & torch.isfinite(y)
    )
    x = x[finite]
    y = y[finite]

    if x.numel() < 2:
        return float("nan")

    x = x - x.mean()
    y = y - y.mean()
    denominator = (
        x.pow(2).sum()
        * y.pow(2).sum()
    ).sqrt()

    if denominator <= 0:
        return float("nan")

    return float(
        (x * y).sum().div(
            denominator
        ).item()
    )


def rank_1d(
    values: torch.Tensor,
) -> torch.Tensor:
    values = values.detach().cpu().float()
    order = values.argsort(
        stable=True,
    )
    ranks = torch.empty(
        values.numel(),
        dtype=torch.float64,
    )
    ranks[order] = torch.arange(
        values.numel(),
        dtype=torch.float64,
    )
    return ranks


def spearman_1d(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    finite = (
        torch.isfinite(x.detach().cpu())
        & torch.isfinite(y.detach().cpu())
    )
    x_finite = x.detach().cpu()[finite]
    y_finite = y.detach().cpu()[finite]

    if x_finite.numel() < 2:
        return float("nan")

    return pearson_1d(
        rank_1d(x_finite),
        rank_1d(y_finite),
    )


def gather_by_index(
    values: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    return values.gather(
        dim=1,
        index=index.unsqueeze(1),
    ).squeeze(1)


def index_rank(
    values: torch.Tensor,
    selected_index: torch.Tensor,
) -> torch.Tensor:
    ranks = rank_rows(
        values
    )
    return gather_by_index(
        ranks,
        selected_index,
    )


def try_scenario_ids(
    data,
    eval_mask: torch.Tensor,
) -> List[str]:
    num_eval = int(
        eval_mask.sum().item()
    )
    fallback = [""] * num_eval

    try:
        scenario_ids = data["scenario_id"]
        agent_batch = data["agent"].get(
            "batch",
            None,
        )
        if agent_batch is None:
            if num_eval == 1:
                return [
                    str(scenario_ids)
                ]
            return fallback

        batch_indices = agent_batch[
            eval_mask
        ].detach().cpu().tolist()

        if isinstance(
            scenario_ids,
            (list, tuple),
        ):
            return [
                str(scenario_ids[index])
                for index in batch_indices
            ]
    except Exception:
        return fallback

    return fallback


class AlignmentAccumulator:
    def __init__(
        self,
        latent_elements: int,
        hit_ks: Sequence[int],
        collect_agent_rows: bool,
    ) -> None:
        self.latent_elements = latent_elements
        self.hit_ks = list(hit_ks)
        self.collect_agent_rows = collect_agent_rows

        self.num_batches = 0
        self.num_agents = 0
        self.num_candidate_pairs = 0

        # Candidate-pair arrays for pooled rank correlations.
        self.std_latent_flat: List[torch.Tensor] = []
        self.raw_latent_flat: List[torch.Tensor] = []
        self.ade_flat: List[torch.Tensor] = []
        self.fde_flat: List[torch.Tensor] = []
        self.per_dim_sq_flat: List[List[torch.Tensor]] = [
            []
            for _ in range(latent_elements)
        ]

        # Per-agent correlations.
        self.agent_pearson_std_ade: List[torch.Tensor] = []
        self.agent_spearman_std_ade: List[torch.Tensor] = []
        self.agent_pearson_std_fde: List[torch.Tensor] = []
        self.agent_spearman_std_fde: List[torch.Tensor] = []
        self.agent_pearson_raw_ade: List[torch.Tensor] = []
        self.agent_spearman_raw_ade: List[torch.Tensor] = []

        # Oracle/selection statistics.
        self.min_latent: List[torch.Tensor] = []
        self.min_ade: List[torch.Tensor] = []
        self.min_fde: List[torch.Tensor] = []
        self.ade_at_latent_best: List[torch.Tensor] = []
        self.fde_at_latent_best: List[torch.Tensor] = []
        self.latent_at_ade_best: List[torch.Tensor] = []
        self.latent_at_fde_best: List[torch.Tensor] = []
        self.ade_regret_latent_best: List[torch.Tensor] = []
        self.fde_regret_latent_best: List[torch.Tensor] = []
        self.rank_ade_best_in_latent: List[torch.Tensor] = []
        self.rank_fde_best_in_latent: List[torch.Tensor] = []
        self.rank_latent_best_in_ade: List[torch.Tensor] = []

        self.exact_latent_ade_agreement = 0
        self.exact_latent_fde_agreement = 0
        self.hit_ade_best_in_latent_topk = {
            k: 0
            for k in self.hit_ks
        }
        self.hit_fde_best_in_latent_topk = {
            k: 0
            for k in self.hit_ks
        }
        self.hit_latent_best_in_ade_topk = {
            k: 0
            for k in self.hit_ks
        }

        self.agent_rows: List[Dict[str, Any]] = []

    def update(
        self,
        latent_std_rmse: torch.Tensor,
        latent_raw_rmse: torch.Tensor,
        per_dim_sq_error: torch.Tensor,
        ade_per_mode: torch.Tensor,
        fde_per_mode: torch.Tensor,
        valid_steps: torch.Tensor,
        scenario_ids: Sequence[str],
    ) -> None:
        if not (
            latent_std_rmse.shape
            == latent_raw_rmse.shape
            == ade_per_mode.shape
            == fde_per_mode.shape
        ):
            raise ValueError(
                "Candidate metric shapes do not match: "
                f"std={latent_std_rmse.shape}, "
                f"raw={latent_raw_rmse.shape}, "
                f"ade={ade_per_mode.shape}, "
                f"fde={fde_per_mode.shape}"
            )

        n, modes = latent_std_rmse.shape
        if per_dim_sq_error.shape != (
            n,
            modes,
            self.latent_elements,
        ):
            raise ValueError(
                "per_dim_sq_error shape mismatch: "
                f"got={per_dim_sq_error.shape}, "
                f"expected={(n, modes, self.latent_elements)}"
            )

        if len(scenario_ids) != n:
            scenario_ids = [""] * n

        pearson_std_ade, _ = rowwise_pearson(
            latent_std_rmse,
            ade_per_mode,
        )
        spearman_std_ade, _ = rowwise_spearman(
            latent_std_rmse,
            ade_per_mode,
        )
        pearson_std_fde, _ = rowwise_pearson(
            latent_std_rmse,
            fde_per_mode,
        )
        spearman_std_fde, _ = rowwise_spearman(
            latent_std_rmse,
            fde_per_mode,
        )
        pearson_raw_ade, _ = rowwise_pearson(
            latent_raw_rmse,
            ade_per_mode,
        )
        spearman_raw_ade, _ = rowwise_spearman(
            latent_raw_rmse,
            ade_per_mode,
        )

        latent_best_index = latent_std_rmse.argmin(
            dim=1
        )
        ade_best_index = ade_per_mode.argmin(
            dim=1
        )
        fde_best_index = fde_per_mode.argmin(
            dim=1
        )

        min_latent = gather_by_index(
            latent_std_rmse,
            latent_best_index,
        )
        min_ade = gather_by_index(
            ade_per_mode,
            ade_best_index,
        )
        min_fde = gather_by_index(
            fde_per_mode,
            fde_best_index,
        )

        ade_at_latent_best = gather_by_index(
            ade_per_mode,
            latent_best_index,
        )
        fde_at_latent_best = gather_by_index(
            fde_per_mode,
            latent_best_index,
        )
        latent_at_ade_best = gather_by_index(
            latent_std_rmse,
            ade_best_index,
        )
        latent_at_fde_best = gather_by_index(
            latent_std_rmse,
            fde_best_index,
        )

        ade_regret = (
            ade_at_latent_best
            - min_ade
        )
        fde_regret = (
            fde_at_latent_best
            - min_fde
        )

        rank_ade_best_in_latent = index_rank(
            latent_std_rmse,
            ade_best_index,
        )
        rank_fde_best_in_latent = index_rank(
            latent_std_rmse,
            fde_best_index,
        )
        rank_latent_best_in_ade = index_rank(
            ade_per_mode,
            latent_best_index,
        )

        self.num_agents += n
        self.num_candidate_pairs += (
            n * modes
        )

        self.std_latent_flat.append(
            latent_std_rmse.detach().cpu().flatten()
        )
        self.raw_latent_flat.append(
            latent_raw_rmse.detach().cpu().flatten()
        )
        self.ade_flat.append(
            ade_per_mode.detach().cpu().flatten()
        )
        self.fde_flat.append(
            fde_per_mode.detach().cpu().flatten()
        )

        per_dim_cpu = per_dim_sq_error.detach().cpu()
        for dim_index in range(
            self.latent_elements
        ):
            self.per_dim_sq_flat[
                dim_index
            ].append(
                per_dim_cpu[
                    :,
                    :,
                    dim_index,
                ].flatten()
            )

        self.agent_pearson_std_ade.append(
            pearson_std_ade.detach().cpu()
        )
        self.agent_spearman_std_ade.append(
            spearman_std_ade.detach().cpu()
        )
        self.agent_pearson_std_fde.append(
            pearson_std_fde.detach().cpu()
        )
        self.agent_spearman_std_fde.append(
            spearman_std_fde.detach().cpu()
        )
        self.agent_pearson_raw_ade.append(
            pearson_raw_ade.detach().cpu()
        )
        self.agent_spearman_raw_ade.append(
            spearman_raw_ade.detach().cpu()
        )

        for destination, value in (
            (self.min_latent, min_latent),
            (self.min_ade, min_ade),
            (self.min_fde, min_fde),
            (self.ade_at_latent_best, ade_at_latent_best),
            (self.fde_at_latent_best, fde_at_latent_best),
            (self.latent_at_ade_best, latent_at_ade_best),
            (self.latent_at_fde_best, latent_at_fde_best),
            (self.ade_regret_latent_best, ade_regret),
            (self.fde_regret_latent_best, fde_regret),
            (self.rank_ade_best_in_latent, rank_ade_best_in_latent),
            (self.rank_fde_best_in_latent, rank_fde_best_in_latent),
            (self.rank_latent_best_in_ade, rank_latent_best_in_ade),
        ):
            destination.append(
                value.detach().cpu()
            )

        self.exact_latent_ade_agreement += int(
            (
                latent_best_index
                == ade_best_index
            ).sum().item()
        )
        self.exact_latent_fde_agreement += int(
            (
                latent_best_index
                == fde_best_index
            ).sum().item()
        )

        for k in self.hit_ks:
            k_effective = min(
                k,
                modes,
            )
            self.hit_ade_best_in_latent_topk[
                k
            ] += int(
                (
                    rank_ade_best_in_latent
                    < k_effective
                ).sum().item()
            )
            self.hit_fde_best_in_latent_topk[
                k
            ] += int(
                (
                    rank_fde_best_in_latent
                    < k_effective
                ).sum().item()
            )
            self.hit_latent_best_in_ade_topk[
                k
            ] += int(
                (
                    rank_latent_best_in_ade
                    < k_effective
                ).sum().item()
            )

        if self.collect_agent_rows:
            valid_steps_cpu = (
                valid_steps.detach().cpu()
            )
            for i in range(n):
                self.agent_rows.append(
                    {
                        "agent_index": (
                            self.num_agents - n + i
                        ),
                        "scenario_id": (
                            scenario_ids[i]
                        ),
                        "valid_steps": int(
                            valid_steps_cpu[i].item()
                        ),
                        "pearson_std_latent_ADE": float(
                            pearson_std_ade[i].item()
                        ),
                        "spearman_std_latent_ADE": float(
                            spearman_std_ade[i].item()
                        ),
                        "pearson_std_latent_FDE": float(
                            pearson_std_fde[i].item()
                        ),
                        "spearman_std_latent_FDE": float(
                            spearman_std_fde[i].item()
                        ),
                        "pearson_raw_latent_ADE": float(
                            pearson_raw_ade[i].item()
                        ),
                        "spearman_raw_latent_ADE": float(
                            spearman_raw_ade[i].item()
                        ),
                        "latent_best_mode": int(
                            latent_best_index[i].item()
                        ),
                        "ADE_best_mode": int(
                            ade_best_index[i].item()
                        ),
                        "FDE_best_mode": int(
                            fde_best_index[i].item()
                        ),
                        "min_latent_std_RMSE": float(
                            min_latent[i].item()
                        ),
                        "min_ADE_m": float(
                            min_ade[i].item()
                        ),
                        "min_FDE_m": float(
                            min_fde[i].item()
                        ),
                        "ADE_at_latent_best_m": float(
                            ade_at_latent_best[i].item()
                        ),
                        "FDE_at_latent_best_m": float(
                            fde_at_latent_best[i].item()
                        ),
                        "latent_RMSE_at_ADE_best": float(
                            latent_at_ade_best[i].item()
                        ),
                        "latent_RMSE_at_FDE_best": float(
                            latent_at_fde_best[i].item()
                        ),
                        "ADE_regret_latent_best_m": float(
                            ade_regret[i].item()
                        ),
                        "FDE_regret_latent_best_m": float(
                            fde_regret[i].item()
                        ),
                        "rank_ADE_best_in_latent_0based": int(
                            rank_ade_best_in_latent[i].item()
                        ),
                        "rank_FDE_best_in_latent_0based": int(
                            rank_fde_best_in_latent[i].item()
                        ),
                        "rank_latent_best_in_ADE_0based": int(
                            rank_latent_best_in_ade[i].item()
                        ),
                    }
                )

    def finalize(self) -> Dict[str, Any]:
        if self.num_agents == 0:
            raise RuntimeError(
                "No focal agents were evaluated."
            )

        std_latent = torch.cat(
            self.std_latent_flat,
            dim=0,
        )
        raw_latent = torch.cat(
            self.raw_latent_flat,
            dim=0,
        )
        ade = torch.cat(
            self.ade_flat,
            dim=0,
        )
        fde = torch.cat(
            self.fde_flat,
            dim=0,
        )

        per_agent = {
            "pearson_std_latent_ADE": scalar_summary(
                torch.cat(
                    self.agent_pearson_std_ade,
                    dim=0,
                )
            ),
            "spearman_std_latent_ADE": scalar_summary(
                torch.cat(
                    self.agent_spearman_std_ade,
                    dim=0,
                )
            ),
            "pearson_std_latent_FDE": scalar_summary(
                torch.cat(
                    self.agent_pearson_std_fde,
                    dim=0,
                )
            ),
            "spearman_std_latent_FDE": scalar_summary(
                torch.cat(
                    self.agent_spearman_std_fde,
                    dim=0,
                )
            ),
            "pearson_raw_latent_ADE": scalar_summary(
                torch.cat(
                    self.agent_pearson_raw_ade,
                    dim=0,
                )
            ),
            "spearman_raw_latent_ADE": scalar_summary(
                torch.cat(
                    self.agent_spearman_raw_ade,
                    dim=0,
                )
            ),
        }

        # Reuse global ranks to avoid repeatedly sorting ADE/FDE arrays.
        rank_std_latent = rank_1d(std_latent)
        rank_raw_latent = rank_1d(raw_latent)
        rank_ade = rank_1d(ade)
        rank_fde = rank_1d(fde)

        per_dimension = []
        for dim_index in range(
            self.latent_elements
        ):
            dim_error = torch.cat(
                self.per_dim_sq_flat[
                    dim_index
                ],
                dim=0,
            )
            rank_dim_error = rank_1d(
                dim_error
            )
            per_dimension.append(
                {
                    "flat_latent_dimension": dim_index,
                    "pearson_sq_error_ADE": pearson_1d(
                        dim_error,
                        ade,
                    ),
                    "spearman_sq_error_ADE": pearson_1d(
                        rank_dim_error,
                        rank_ade,
                    ),
                    "pearson_sq_error_FDE": pearson_1d(
                        dim_error,
                        fde,
                    ),
                    "spearman_sq_error_FDE": pearson_1d(
                        rank_dim_error,
                        rank_fde,
                    ),
                    "mean_sq_error": float(
                        dim_error.mean().item()
                    ),
                }
            )

        min_latent = torch.cat(
            self.min_latent,
            dim=0,
        )
        min_ade = torch.cat(
            self.min_ade,
            dim=0,
        )
        min_fde = torch.cat(
            self.min_fde,
            dim=0,
        )
        ade_at_latent_best = torch.cat(
            self.ade_at_latent_best,
            dim=0,
        )
        fde_at_latent_best = torch.cat(
            self.fde_at_latent_best,
            dim=0,
        )
        latent_at_ade_best = torch.cat(
            self.latent_at_ade_best,
            dim=0,
        )
        latent_at_fde_best = torch.cat(
            self.latent_at_fde_best,
            dim=0,
        )
        ade_regret = torch.cat(
            self.ade_regret_latent_best,
            dim=0,
        )
        fde_regret = torch.cat(
            self.fde_regret_latent_best,
            dim=0,
        )
        rank_ade_best_in_latent = torch.cat(
            self.rank_ade_best_in_latent,
            dim=0,
        )
        rank_fde_best_in_latent = torch.cat(
            self.rank_fde_best_in_latent,
            dim=0,
        )
        rank_latent_best_in_ade = torch.cat(
            self.rank_latent_best_in_ade,
            dim=0,
        )

        return {
            "counts": {
                "num_batches": self.num_batches,
                "num_focal_agents": self.num_agents,
                "num_candidate_pairs": self.num_candidate_pairs,
                "latent_elements": self.latent_elements,
            },
            "pooled_candidate_pair_correlations": {
                "pearson_std_latent_ADE": pearson_1d(
                    std_latent,
                    ade,
                ),
                "spearman_std_latent_ADE": pearson_1d(
                    rank_std_latent,
                    rank_ade,
                ),
                "pearson_std_latent_FDE": pearson_1d(
                    std_latent,
                    fde,
                ),
                "spearman_std_latent_FDE": pearson_1d(
                    rank_std_latent,
                    rank_fde,
                ),
                "pearson_raw_latent_ADE": pearson_1d(
                    raw_latent,
                    ade,
                ),
                "spearman_raw_latent_ADE": pearson_1d(
                    rank_raw_latent,
                    rank_ade,
                ),
                "pearson_raw_latent_FDE": pearson_1d(
                    raw_latent,
                    fde,
                ),
                "spearman_raw_latent_FDE": pearson_1d(
                    rank_raw_latent,
                    rank_fde,
                ),
            },
            "per_agent_within_modes_correlations": per_agent,
            "oracle_and_latent_selection": {
                "min_latent_std_RMSE": scalar_summary(
                    min_latent
                ),
                "min_ADE_m": scalar_summary(
                    min_ade
                ),
                "min_FDE_m": scalar_summary(
                    min_fde
                ),
                "ADE_at_latent_nearest_m": scalar_summary(
                    ade_at_latent_best
                ),
                "FDE_at_latent_nearest_m": scalar_summary(
                    fde_at_latent_best
                ),
                "latent_RMSE_at_ADE_best": scalar_summary(
                    latent_at_ade_best
                ),
                "latent_RMSE_at_FDE_best": scalar_summary(
                    latent_at_fde_best
                ),
                "ADE_regret_using_latent_nearest_m": scalar_summary(
                    ade_regret
                ),
                "FDE_regret_using_latent_nearest_m": scalar_summary(
                    fde_regret
                ),
                "rank_ADE_best_in_latent_order_0based": scalar_summary(
                    rank_ade_best_in_latent
                ),
                "rank_FDE_best_in_latent_order_0based": scalar_summary(
                    rank_fde_best_in_latent
                ),
                "rank_latent_best_in_ADE_order_0based": scalar_summary(
                    rank_latent_best_in_ade
                ),
                "exact_latent_best_equals_ADE_best_rate": (
                    self.exact_latent_ade_agreement
                    / self.num_agents
                ),
                "exact_latent_best_equals_FDE_best_rate": (
                    self.exact_latent_fde_agreement
                    / self.num_agents
                ),
                "ADE_best_in_latent_topk_rate": {
                    str(k): (
                        self.hit_ade_best_in_latent_topk[k]
                        / self.num_agents
                    )
                    for k in self.hit_ks
                },
                "FDE_best_in_latent_topk_rate": {
                    str(k): (
                        self.hit_fde_best_in_latent_topk[k]
                        / self.num_agents
                    )
                    for k in self.hit_ks
                },
                "latent_best_in_ADE_topk_rate": {
                    str(k): (
                        self.hit_latent_best_in_ade_topk[k]
                        / self.num_agents
                    )
                    for k in self.hit_ks
                },
            },
            "per_latent_dimension": per_dimension,
        }


def write_agent_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise RuntimeError(
            "No per-agent rows were collected."
        )

    fieldnames = list(
        rows[0].keys()
    )
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate(
    model: QCNetFM,
    loader: Iterable,
    device: torch.device,
    precision: str,
    num_modes: int,
    fm_num_steps: int,
    seed: int,
    max_batches: int,
    log_interval: int,
    focal_category: int,
    trajectory_scale: float,
    hit_ks: Sequence[int],
    collect_agent_rows: bool,
) -> AlignmentAccumulator:
    set_seed(seed)
    model.eval()
    model.encoder.eval()
    model.latent_encoder.eval()
    model.latent_decoder.eval()
    model.fm_decoder.eval()

    latent_elements = int(
        model.vae_num_intents
        * model.latent_dim
    )
    accumulator = AlignmentAccumulator(
        latent_elements=latent_elements,
        hit_ks=hit_ks,
        collect_agent_rows=collect_agent_rows,
    )

    for batch_index, data in enumerate(loader):
        if (
            max_batches > 0
            and batch_index >= max_batches
        ):
            break

        if isinstance(data, Batch):
            data["agent"]["av_index"] += (
                data["agent"]["ptr"][:-1]
            )

        data = data.to(device)

        future_mask = data["agent"]["predict_mask"][
            :,
            model.num_historical_steps:,
        ].bool()
        eval_mask = (
            data["agent"]["category"]
            == int(focal_category)
        ) & future_mask.any(
            dim=-1
        )

        accumulator.num_batches += 1

        if not eval_mask.any():
            continue

        target_normalized = (
            data["agent"]["target"][
                ...,
                : model.output_dim,
            ].float()
            / float(trajectory_scale)
        )

        with autocast_context(
            device,
            precision,
        ):
            scene_enc = model.encoder(
                data
            )
            latent_target_raw = (
                model.latent_encoder.encode(
                    target_normalized,
                    predict_mask=future_mask,
                )
            )
            (
                trajectories_normalized,
                latent_samples_std,
            ) = sample_trajectories_and_latents(
                model=model,
                data=data,
                scene_enc=scene_enc,
                num_modes=num_modes,
                num_steps=fm_num_steps,
            )

        latent_target_std = (
            latent_target_raw.float()
            - model.z_mean.float()
        ) / (
            model.z_std.float()
            + 1e-6
        )

        target_eval_m = (
            target_normalized[
                eval_mask
            ].float()
            * float(trajectory_scale)
        )
        trajectories_eval_m = (
            trajectories_normalized[
                eval_mask
            ].float()
            * float(trajectory_scale)
        )
        future_mask_eval = future_mask[
            eval_mask
        ]
        latent_samples_eval = latent_samples_std[
            eval_mask
        ].float()
        latent_target_eval = latent_target_std[
            eval_mask
        ].float()

        ade_per_mode, fde_per_mode = (
            trajectory_distance_per_mode(
                trajectories_m=(
                    trajectories_eval_m
                ),
                target_m=target_eval_m,
                valid_mask=future_mask_eval,
            )
        )

        (
            latent_std_rmse,
            latent_raw_rmse,
            per_dim_sq_error,
        ) = latent_distance_per_mode(
            latent_samples_std=(
                latent_samples_eval
            ),
            latent_target_std=(
                latent_target_eval
            ),
            z_std=model.z_std,
        )

        scenario_ids = try_scenario_ids(
            data,
            eval_mask,
        )

        accumulator.update(
            latent_std_rmse=(
                latent_std_rmse
            ),
            latent_raw_rmse=(
                latent_raw_rmse
            ),
            per_dim_sq_error=(
                per_dim_sq_error
            ),
            ade_per_mode=ade_per_mode,
            fde_per_mode=fde_per_mode,
            valid_steps=(
                future_mask_eval.sum(
                    dim=-1
                )
            ),
            scenario_ids=scenario_ids,
        )

        if (
            log_interval > 0
            and (
                batch_index == 0
                or (batch_index + 1)
                % log_interval
                == 0
            )
        ):
            running_min_latent = torch.cat(
                accumulator.min_latent,
                dim=0,
            ).mean().item()
            running_min_ade = torch.cat(
                accumulator.min_ade,
                dim=0,
            ).mean().item()
            running_ade_at_latent = torch.cat(
                accumulator.ade_at_latent_best,
                dim=0,
            ).mean().item()
            running_regret = torch.cat(
                accumulator.ade_regret_latent_best,
                dim=0,
            ).mean().item()
            running_spearman = torch.cat(
                accumulator.agent_spearman_std_ade,
                dim=0,
            )
            running_spearman = running_spearman[
                torch.isfinite(
                    running_spearman
                )
            ].mean().item()

            print(
                f"[seed={seed}] "
                f"batch={batch_index + 1:>5d} "
                f"agents={accumulator.num_agents:>7,d} "
                f"minLatentRMSE={running_min_latent:.6f} "
                f"minADE={running_min_ade:.6f} "
                f"ADE@latentBest={running_ade_at_latent:.6f} "
                f"ADEregret={running_regret:.6f} "
                f"meanAgentSpearman={running_spearman:.6f}"
            )

    return accumulator


def print_summary(
    result: Dict[str, Any],
) -> None:
    counts = result["counts"]
    pooled = result[
        "pooled_candidate_pair_correlations"
    ]
    per_agent = result[
        "per_agent_within_modes_correlations"
    ]
    selection = result[
        "oracle_and_latent_selection"
    ]

    print("\n" + "=" * 100)
    print("Latent-distance vs trajectory-distance alignment")
    print("=" * 100)

    print("\nCounts")
    for key, value in counts.items():
        print(
            f"  {key:<32s}: {value:,}"
            if isinstance(value, int)
            else f"  {key:<32s}: {value}"
        )

    print("\nPooled candidate-pair correlations")
    for key, value in pooled.items():
        print(
            f"  {key:<34s}: {value:.6f}"
        )

    print("\nPer-agent within-mode correlations")
    for key, summary in per_agent.items():
        print(
            f"  {key:<34s}: "
            f"mean={summary['mean']:.6f}, "
            f"median={summary['median']:.6f}, "
            f"q10={summary['quantiles']['0.10']:.6f}, "
            f"q90={summary['quantiles']['0.90']:.6f}"
        )

    print("\nOracle and latent-nearest selection")
    important_keys = (
        "min_latent_std_RMSE",
        "min_ADE_m",
        "min_FDE_m",
        "ADE_at_latent_nearest_m",
        "FDE_at_latent_nearest_m",
        "ADE_regret_using_latent_nearest_m",
        "FDE_regret_using_latent_nearest_m",
        "rank_ADE_best_in_latent_order_0based",
    )
    for key in important_keys:
        summary = selection[key]
        print(
            f"  {key:<42s}: "
            f"mean={summary['mean']:.6f}, "
            f"median={summary['median']:.6f}, "
            f"q90={summary['quantiles']['0.90']:.6f}"
        )

    print(
        "\nExact agreement rates"
    )
    print(
        "  latent-best == ADE-best"
        f" : {selection['exact_latent_best_equals_ADE_best_rate']:.2%}"
    )
    print(
        "  latent-best == FDE-best"
        f" : {selection['exact_latent_best_equals_FDE_best_rate']:.2%}"
    )

    print(
        "\nADE-best contained in latent top-K"
    )
    for key, value in selection[
        "ADE_best_in_latent_topk_rate"
    ].items():
        print(
            f"  K={int(key):>3d}: {value:.2%}"
        )

    print(
        "\nPer-latent-dimension squared-error correlation with ADE"
    )
    for row in result[
        "per_latent_dimension"
    ]:
        print(
            f"  dim={row['flat_latent_dimension']:>2d} "
            f"Pearson={row['pearson_sq_error_ADE']:+.6f} "
            f"Spearman={row['spearman_sq_error_ADE']:+.6f} "
            f"meanSqErr={row['mean_sq_error']:.6f}"
        )


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(
        args.ckpt
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if args.num_modes < 2:
        raise ValueError(
            "At least 2 modes are required for correlation analysis."
        )

    hit_ks = sorted(
        {
            max(
                1,
                min(
                    int(k),
                    args.num_modes,
                ),
            )
            for k in args.hit_ks
        }
    )

    device = torch.device(
        args.device
    )
    model, hparams = load_model(
        ckpt_path=checkpoint_path,
        device=device,
        allow_shape_mismatch=(
            args.allow_shape_mismatch
        ),
    )

    fm_num_steps = (
        int(args.fm_num_steps)
        if args.fm_num_steps is not None
        else int(model.fm_num_steps)
    )

    datamodule = build_datamodule(
        hparams=hparams,
        args=args,
    )
    val_loader = (
        datamodule.val_dataloader()
    )

    print(f"checkpoint       : {checkpoint_path}")
    print(f"device           : {device}")
    print(f"precision        : {args.precision}")
    print(f"seed             : {args.seed}")
    print(f"num_modes        : {args.num_modes}")
    print(f"fm_num_steps     : {fm_num_steps}")
    print(f"hit_ks           : {hit_ks}")
    print(
        f"trajectory_scale : "
        f"{args.trajectory_scale}"
    )

    accumulator = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        precision=args.precision,
        num_modes=args.num_modes,
        fm_num_steps=fm_num_steps,
        seed=args.seed,
        max_batches=args.max_batches,
        log_interval=args.log_interval,
        focal_category=args.focal_category,
        trajectory_scale=(
            args.trajectory_scale
        ),
        hit_ks=hit_ks,
        collect_agent_rows=(
            args.output_agent_csv
            is not None
        ),
    )

    result = accumulator.finalize()
    output = {
        "config": {
            "checkpoint": str(
                checkpoint_path.resolve()
            ),
            "device": str(device),
            "precision": args.precision,
            "seed": args.seed,
            "num_modes": args.num_modes,
            "fm_num_steps": fm_num_steps,
            "focal_category": (
                args.focal_category
            ),
            "trajectory_scale": (
                args.trajectory_scale
            ),
            "hit_ks": hit_ks,
            "max_batches": (
                args.max_batches
            ),
            "latent_distance": (
                "RMSE in standardized VAE latent space"
            ),
            "trajectory_distance": (
                "ADE/FDE in meters"
            ),
        },
        **result,
    }

    print_summary(
        output
    )

    if args.output_json:
        output_json_path = Path(
            args.output_json
        )
        output_json_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_json_path.write_text(
            json.dumps(
                output,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "\nJSON saved to: "
            f"{output_json_path.resolve()}"
        )

    if args.output_agent_csv:
        output_csv_path = Path(
            args.output_agent_csv
        )
        write_agent_csv(
            output_csv_path,
            accumulator.agent_rows,
        )
        print(
            "Per-agent CSV saved to: "
            f"{output_csv_path.resolve()}"
        )


if __name__ == "__main__":
    main()