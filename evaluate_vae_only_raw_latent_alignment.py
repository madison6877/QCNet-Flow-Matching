#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate VAE geometry entirely in the raw latent space.

For each focal agent, this script:
1. Encodes the ground-truth future trajectory to raw latent z_raw.
2. Samples isotropic raw-space perturbations:
       z_candidate_raw = z_raw + radius * direction.
3. Decodes candidates directly with the raw VAE decoder.
4. Uses raw-latent RMSE for correlations, nearest-neighbour selection,
   regret, rank, Exact@1 and Hit@K.
5. Directly evaluates ADE-FDE consistency, including ADE@minFDE,
   FDE@minADE, cross-metric regret, Exact@1, rank and Hit@K.
6. Reports alignment to both ground truth and the center reconstruction D(z_raw).

The checkpoint z_mean/z_std are printed only as reference. They are not used for
candidate sampling, latent distance, nearest selection, or decoding transforms.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
import sys
import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch

from datamodules import ArgoverseV2DataModule
from datasets import ArgoverseV2Dataset
from predictors import QCNetFM


class ExistingCacheOnlyArgoverseV2Dataset(ArgoverseV2Dataset):
    """Never permit PyG to rebuild processed data during evaluation."""
    def process(self) -> None:
        raise RuntimeError(
            "ArgoverseV2Dataset attempted to preprocess validation data even though this evaluator "
            "is cache-only. The effective processed directory contains no recognized direct "
            ".pkl/.pickle files, or a different dataset implementation was imported."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VAE alignment using isotropic sampling and distance in raw latent space.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--train_raw_dir", default=None)
    parser.add_argument("--train_processed_dir", default=None)
    parser.add_argument("--val_raw_dir", default=None)
    parser.add_argument("--val_processed_dir", default=None)
    parser.add_argument("--test_raw_dir", default=None)
    parser.add_argument("--test_processed_dir", default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["32", "bf16", "fp16"], default="32")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max_batches", type=int, default=0, help="0 means the complete validation set.")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--radius_min", type=float, default=0.02, help="Minimum Euclidean perturbation radius in flattened raw latent space.")
    parser.add_argument("--radius_max", type=float, default=1.0, help="Maximum Euclidean perturbation radius in flattened raw latent space.")
    parser.add_argument("--radius_distribution", choices=["uniform", "log_uniform"], default="uniform")
    parser.add_argument("--decode_chunk_size", type=int, default=4096)
    parser.add_argument("--trajectory_scale", type=float, default=10.0)
    parser.add_argument("--focal_category", type=int, default=3)
    parser.add_argument("--hit_ks", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_agent_csv_prefix", default=None, help="Writes <prefix>_gt.csv and <prefix>_reconstruction.csv.")
    parser.add_argument("--allow_shape_mismatch", action="store_true")
    return parser.parse_args()


def torch_load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint content must be dict, got {type(obj)}")
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
    raise TypeError(f"Cannot convert hyper_parameters to dict: {type(obj)}")


def strip_geometry_training_state(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Drop geometry-training-only buffers before loading the inference QCNetFM.

    These root-level geometry_* values are required only for resuming the
    geometry-aware trainer. They do not participate in VAE encoding, raw-latent
    decoding, z_mean/z_std, or the alignment metrics evaluated by this script.
    """
    ignored = sorted(key for key in state_dict if key.startswith("geometry_"))
    filtered = {key: value for key, value in state_dict.items() if key not in ignored}
    return filtered, ignored


def load_model(ckpt_path: Path, device: torch.device, allow_shape_mismatch: bool) -> Tuple[QCNetFM, Dict[str, Any]]:
    checkpoint = torch_load_checkpoint(ckpt_path)
    hparams = as_plain_dict(checkpoint.get("hyper_parameters", {}))
    if not hparams:
        raise RuntimeError("checkpoint is missing hyper_parameters")
    model = QCNetFM(**hparams)
    raw_state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict, ignored_geometry_keys = strip_geometry_training_state(raw_state_dict)
    if ignored_geometry_keys:
        print(f"[Checkpoint] ignored {len(ignored_geometry_keys)} geometry-training-only keys:")
        for key in ignored_geometry_keys:
            print(f"  - {key}")
    if not allow_shape_mismatch:
        model.load_state_dict(state_dict, strict=True)
    else:
        current = model.state_dict()
        compatible: Dict[str, torch.Tensor] = {}
        mismatched: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
        for key, value in state_dict.items():
            if key not in current:
                continue
            if tuple(value.shape) != tuple(current[key].shape):
                mismatched.append((key, tuple(value.shape), tuple(current[key].shape)))
                continue
            compatible[key] = value
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print("\n[WARNING] --allow_shape_mismatch is enabled.")
        print(f"compatible={len(compatible):,}, mismatched={len(mismatched):,}, missing={len(missing):,}, unexpected={len(unexpected):,}")
        for item in mismatched[:20]:
            print(f"  skipped {item[0]}: ckpt={item[1]} current={item[2]}")
        missing_vae = [key for key in missing if key.startswith("latent_encoder.")]
        if missing_vae:
            raise RuntimeError(f"VAE keys are missing after checkpoint loading: {missing_vae[:20]}")
        print("[WARNING] Formal metrics are invalid if a relevant VAE key was skipped.\n")
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, hparams


def build_datamodule(hparams: Dict[str, Any], args: argparse.Namespace) -> ArgoverseV2DataModule:
    """Build only the validation dataset.

    ArgoverseV2DataModule.setup(stage="fit") in this project ignores ``stage``
    and instantiates train, val and test datasets. Instantiating the unused train
    or test split can trigger preprocessing even when --val_processed_dir is
    valid. This evaluator needs only validation data, so it creates val_dataset
    directly and never touches train/test caches.
    """
    cfg = dict(hparams)
    overrides = {"root": args.root, "train_raw_dir": args.train_raw_dir, "train_processed_dir": args.train_processed_dir,
                 "val_raw_dir": args.val_raw_dir, "val_processed_dir": args.val_processed_dir,
                 "test_raw_dir": args.test_raw_dir, "test_processed_dir": args.test_processed_dir,
                 "val_batch_size": args.val_batch_size, "num_workers": args.num_workers,
                 "pin_memory": args.pin_memory, "persistent_workers": args.persistent_workers}
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
    cfg["prototype_assignment_dir"] = None
    cfg["prototype_assignment_strict"] = False
    if cfg.get("persistent_workers") and int(cfg["num_workers"]) <= 0:
        cfg["persistent_workers"] = False
    if not cfg.get("root"):
        raise ValueError("Dataset root is missing. Pass --root or use a checkpoint containing root.")
    if not cfg.get("val_processed_dir"):
        raise ValueError("val_processed_dir is missing. Pass --val_processed_dir explicitly.")

    requested_dir = Path(cfg["val_processed_dir"]).expanduser().resolve()
    if not requested_dir.is_dir():
        raise FileNotFoundError(f"val_processed_dir does not exist or is not a directory: {requested_dir}")

    def direct_processed_files(path: Path) -> List[Path]:
        return sorted(p for p in path.iterdir()
                      if p.is_file() and p.name.lower().endswith((".pkl", ".pickle")))

    val_processed_dir = requested_dir
    processed_files = direct_processed_files(val_processed_dir)

    # Common path-level mistakes: passing .../val or the dataset root instead
    # of the directory that directly contains scenario .pkl files.
    if not processed_files:
        candidates = [requested_dir / "processed",
                      requested_dir / "val" / "processed"]
        valid_candidates = []
        for candidate in candidates:
            if candidate.is_dir():
                files = direct_processed_files(candidate)
                if files:
                    valid_candidates.append((candidate.resolve(), files))
        if len(valid_candidates) == 1:
            val_processed_dir, processed_files = valid_candidates[0]
            print(f"[Data] auto-corrected val_processed_dir: {requested_dir} -> {val_processed_dir}")
        elif len(valid_candidates) > 1:
            details = "\n".join(f"  - {path} ({len(files):,} files)"
                                for path, files in valid_candidates)
            raise RuntimeError("Multiple candidate validation processed directories were found; "
                               "pass the exact one explicitly:\n" + details)

    # ArgoverseV2Dataset decides whether preprocessing is needed by listing
    # direct *.pkl/*.pickle children. Directory existence alone is insufficient.
    if not processed_files:
        entries = sorted(p.name for p in requested_dir.iterdir())[:30]
        raise RuntimeError(
            "The supplied val_processed_dir exists, but it contains no direct .pkl/.pickle files.\n"
            f"requested: {requested_dir}\n"
            f"first entries: {entries}\n"
            "ArgoverseV2Dataset would therefore call process() and rebuild the cache. "
            "Pass the directory that directly contains scenario_*.pkl files, normally "
            "<root>/val/processed."
        )

    cfg["val_processed_dir"] = str(val_processed_dir)
    datamodule = ArgoverseV2DataModule(**cfg)
    print("[Data] imported dataset implementation:", inspect.getfile(ArgoverseV2Dataset))
    datamodule.val_dataset = ExistingCacheOnlyArgoverseV2Dataset(
        datamodule.root, "val", datamodule.val_raw_dir,
        str(val_processed_dir), datamodule.val_transform)

    actual_dir = Path(getattr(datamodule.val_dataset, "processed_dir", "")).expanduser().resolve()
    if actual_dir != val_processed_dir:
        raise RuntimeError(f"Dataset changed processed_dir unexpectedly: requested={val_processed_dir}, actual={actual_dir}")

    print("[Data] validation-only mode: train/test datasets were not instantiated.")
    print("[Data] requested val_processed_dir:", requested_dir)
    print("[Data] effective val_processed_dir:", val_processed_dir)
    print("[Data] direct .pkl/.pickle files:", f"{len(processed_files):,}")
    print("[Data] validation samples:", len(datamodule.val_dataset))
    return datamodule


def autocast_context(device: torch.device, precision: str):
    if precision == "32":
        return contextlib.nullcontext()
    if device.type != "cuda":
        print(f"[warning] {precision} autocast is disabled on {device}; using fp32.", file=sys.stderr)
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_raw_decoder(model: QCNetFM):
    """Return the decoder that directly accepts raw latent values.

    Current centered/raw and standardized wrappers expose the actual raw decoder
    through `.decoder`. A plain LatentSpaceDecoder is returned unchanged.
    """
    decoder = model.latent_decoder
    visited = set()
    while hasattr(decoder, "decoder") and id(decoder) not in visited:
        visited.add(id(decoder))
        child = getattr(decoder, "decoder")
        if not isinstance(child, torch.nn.Module):
            break
        decoder = child
    return decoder


def decode_raw(model: QCNetFM, z_raw: torch.Tensor) -> torch.Tensor:
    return get_raw_decoder(model)(z_raw)


def sample_candidate_latents_raw(latent_target_raw: torch.Tensor, num_candidates: int, radius_min: float,
                                 radius_max: float, radius_distribution: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if latent_target_raw.ndim != 3:
        raise ValueError(f"latent_target_raw must be [N,I,D], got {tuple(latent_target_raw.shape)}")
    if num_candidates < 2:
        raise ValueError("num_candidates must be at least 2.")
    if radius_min <= 0 or radius_max <= 0 or radius_max <= radius_min:
        raise ValueError("Require 0 < radius_min < radius_max.")
    n, num_intents, latent_dim = latent_target_raw.shape
    flat_dim = num_intents * latent_dim
    directions = torch.randn(n, num_candidates, flat_dim, device=latent_target_raw.device, dtype=latent_target_raw.dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    uniform = torch.rand(n, num_candidates, 1, device=latent_target_raw.device, dtype=latent_target_raw.dtype)
    if radius_distribution == "uniform":
        radii = radius_min + (radius_max - radius_min) * uniform
    else:
        radii = torch.exp(math.log(radius_min) + (math.log(radius_max) - math.log(radius_min)) * uniform)
    delta_raw = (radii * directions).reshape(n, num_candidates, num_intents, latent_dim)
    return latent_target_raw[:, None] + delta_raw, radii.squeeze(-1)


@torch.no_grad()
def decode_candidate_latents_raw(model: QCNetFM, candidates_raw: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if candidates_raw.ndim != 4:
        raise ValueError(f"candidates_raw must be [N,M,I,D], got {tuple(candidates_raw.shape)}")
    n, modes, num_intents, latent_dim = candidates_raw.shape
    flat = candidates_raw.reshape(n * modes, num_intents, latent_dim)
    outputs: List[torch.Tensor] = []
    for start in range(0, flat.size(0), chunk_size):
        outputs.append(decode_raw(model, flat[start:min(start + chunk_size, flat.size(0))]).float())
    trajectory = torch.cat(outputs, dim=0)
    return trajectory.reshape(n, modes, trajectory.size(-2), trajectory.size(-1))


def trajectory_distance_per_mode(trajectories_m: torch.Tensor, target_m: torch.Tensor,
                                 valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if trajectories_m.ndim != 4 or target_m.ndim != 3 or valid_mask.ndim != 2:
        raise ValueError("Expected trajectories [N,M,T,D], target [N,T,D], mask [N,T].")
    n, modes, steps, dims = trajectories_m.shape
    if target_m.shape != (n, steps, dims) or valid_mask.shape != (n, steps):
        raise ValueError(f"Shape mismatch: pred={trajectories_m.shape}, target={target_m.shape}, mask={valid_mask.shape}")
    valid_mask = valid_mask.bool()
    valid_counts = valid_mask.sum(dim=-1)
    if (valid_counts == 0).any():
        raise ValueError("An evaluated focal agent has no valid future timestep.")
    displacement = torch.linalg.vector_norm(trajectories_m.float() - target_m.float().unsqueeze(1), dim=-1)
    ade = (displacement * valid_mask[:, None].to(displacement.dtype)).sum(dim=-1) / valid_counts[:, None].to(displacement.dtype)
    time_index = torch.arange(steps, device=valid_mask.device).view(1, steps)
    last_valid = time_index.masked_fill(~valid_mask, -1).max(dim=-1).values
    pred_endpoint = trajectories_m.gather(2, last_valid[:, None, None, None].expand(n, modes, 1, dims)).squeeze(2)
    gt_endpoint = target_m.gather(1, last_valid[:, None, None].expand(n, 1, dims)).squeeze(1)
    fde = torch.linalg.vector_norm(pred_endpoint - gt_endpoint.unsqueeze(1), dim=-1)
    return ade, fde


def latent_distance_raw(latent_samples_raw: torch.Tensor, latent_target_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    delta_raw = latent_samples_raw.float() - latent_target_raw.float().unsqueeze(1)
    sq_error_raw = delta_raw.pow(2)
    raw_rmse = sq_error_raw.mean(dim=(-1, -2)).sqrt()
    return raw_rmse, sq_error_raw.flatten(start_dim=2)


def rank_rows(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort(dim=1, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks.scatter_(1, order, torch.arange(values.size(1), device=values.device, dtype=torch.float32).view(1, -1).expand_as(values))
    return ranks


def rowwise_pearson(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.shape != y.shape:
        raise ValueError(f"rowwise correlation shape mismatch: {x.shape} vs {y.shape}")
    x, y = x.float(), y.float()
    xc, yc = x - x.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
    numerator = (xc * yc).sum(1)
    denominator = (xc.pow(2).sum(1) * yc.pow(2).sum(1)).clamp_min(eps).sqrt()
    valid = denominator > math.sqrt(eps)
    correlation = torch.full_like(numerator, float("nan"))
    correlation[valid] = numerator[valid] / denominator[valid]
    return correlation, valid


def rowwise_spearman(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return rowwise_pearson(rank_rows(x), rank_rows(y))


def safe_quantiles(values: torch.Tensor, quantiles: Sequence[float] = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)) -> Dict[str, Any]:
    values = values.detach().cpu().float()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {f"{q:.2f}": None for q in quantiles}
    result = torch.quantile(values, torch.tensor(list(quantiles), dtype=torch.float32))
    return {f"{q:.2f}": float(v) for q, v in zip(quantiles, result.tolist())}


def scalar_summary(values: torch.Tensor) -> Dict[str, Any]:
    values = values.detach().cpu().float()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {"count": 0, "mean": None, "std": None, "median": None, "quantiles": safe_quantiles(finite)}
    return {"count": int(finite.numel()), "mean": float(finite.mean()), "std": float(finite.std(unbiased=False)),
            "median": float(finite.median()), "quantiles": safe_quantiles(finite)}


def pearson_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.detach().cpu().double(), y.detach().cpu().double()
    finite = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[finite], y[finite]
    if x.numel() < 2:
        return float("nan")
    x, y = x - x.mean(), y - y.mean()
    denominator = (x.pow(2).sum() * y.pow(2).sum()).sqrt()
    return float("nan") if denominator <= 0 else float((x * y).sum() / denominator)


def rank_1d(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().cpu().float()
    order = values.argsort(stable=True)
    ranks = torch.empty(values.numel(), dtype=torch.float64)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float64)
    return ranks


def gather_by_index(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return values.gather(1, index.unsqueeze(1)).squeeze(1)


def index_rank(values: torch.Tensor, selected_index: torch.Tensor) -> torch.Tensor:
    return gather_by_index(rank_rows(values), selected_index)


def try_scenario_ids(data, eval_mask: torch.Tensor) -> List[str]:
    num_eval = int(eval_mask.sum().item())
    fallback = [""] * num_eval
    try:
        scenario_ids = data["scenario_id"]
        agent_batch = data["agent"].get("batch", None)
        if agent_batch is None:
            return [str(scenario_ids)] if num_eval == 1 else fallback
        indices = agent_batch[eval_mask].detach().cpu().tolist()
        if isinstance(scenario_ids, (list, tuple)):
            return [str(scenario_ids[index]) for index in indices]
    except Exception:
        pass
    return fallback


class RawAlignmentAccumulator:
    def __init__(self, latent_elements: int, hit_ks: Sequence[int], collect_agent_rows: bool, label: str) -> None:
        self.latent_elements, self.hit_ks = latent_elements, list(hit_ks)
        self.collect_agent_rows, self.label = collect_agent_rows, label
        self.num_batches = self.num_agents = self.num_candidate_pairs = 0
        self.raw_latent_flat: List[torch.Tensor] = []
        self.ade_flat: List[torch.Tensor] = []
        self.fde_flat: List[torch.Tensor] = []
        self.per_dim_sq_flat: List[List[torch.Tensor]] = [[] for _ in range(latent_elements)]
        self.agent_pearson_raw_ade: List[torch.Tensor] = []
        self.agent_spearman_raw_ade: List[torch.Tensor] = []
        self.agent_pearson_raw_fde: List[torch.Tensor] = []
        self.agent_spearman_raw_fde: List[torch.Tensor] = []
        self.agent_pearson_ade_fde: List[torch.Tensor] = []
        self.agent_spearman_ade_fde: List[torch.Tensor] = []
        self.min_latent: List[torch.Tensor] = []
        self.min_ade: List[torch.Tensor] = []
        self.min_fde: List[torch.Tensor] = []
        self.ade_at_latent_best: List[torch.Tensor] = []
        self.fde_at_latent_best: List[torch.Tensor] = []
        self.latent_at_ade_best: List[torch.Tensor] = []
        self.latent_at_fde_best: List[torch.Tensor] = []
        self.ade_regret: List[torch.Tensor] = []
        self.fde_regret: List[torch.Tensor] = []
        self.rank_ade_best_in_latent: List[torch.Tensor] = []
        self.rank_fde_best_in_latent: List[torch.Tensor] = []
        self.rank_latent_best_in_ade: List[torch.Tensor] = []

        # Direct ADE-FDE consistency diagnostics.
        self.ade_at_fde_best: List[torch.Tensor] = []
        self.fde_at_ade_best: List[torch.Tensor] = []
        self.ade_regret_using_fde_best: List[torch.Tensor] = []
        self.fde_regret_using_ade_best: List[torch.Tensor] = []
        self.rank_ade_best_in_fde: List[torch.Tensor] = []
        self.rank_fde_best_in_ade: List[torch.Tensor] = []

        self.exact_ade = self.exact_fde = 0
        self.exact_ade_fde = 0
        self.hit_ade = {k: 0 for k in self.hit_ks}
        self.hit_fde = {k: 0 for k in self.hit_ks}
        self.hit_latent_in_ade = {k: 0 for k in self.hit_ks}
        self.hit_ade_in_fde = {k: 0 for k in self.hit_ks}
        self.hit_fde_in_ade = {k: 0 for k in self.hit_ks}
        self.agent_rows: List[Dict[str, Any]] = []

    def update(self, latent_raw_rmse: torch.Tensor, per_dim_sq_error_raw: torch.Tensor, ade_per_mode: torch.Tensor,
               fde_per_mode: torch.Tensor, valid_steps: torch.Tensor, scenario_ids: Sequence[str]) -> None:
        if not (latent_raw_rmse.shape == ade_per_mode.shape == fde_per_mode.shape):
            raise ValueError("Candidate metric shapes do not match.")
        n, modes = latent_raw_rmse.shape
        if per_dim_sq_error_raw.shape != (n, modes, self.latent_elements):
            raise ValueError(f"per_dim shape mismatch: {per_dim_sq_error_raw.shape}")
        if len(scenario_ids) != n:
            scenario_ids = [""] * n
        p_ade, _ = rowwise_pearson(latent_raw_rmse, ade_per_mode)
        s_ade, _ = rowwise_spearman(latent_raw_rmse, ade_per_mode)
        p_fde, _ = rowwise_pearson(latent_raw_rmse, fde_per_mode)
        s_fde, _ = rowwise_spearman(latent_raw_rmse, fde_per_mode)
        p_ade_fde, _ = rowwise_pearson(ade_per_mode, fde_per_mode)
        s_ade_fde, _ = rowwise_spearman(ade_per_mode, fde_per_mode)
        latent_best = latent_raw_rmse.argmin(1)
        ade_best, fde_best = ade_per_mode.argmin(1), fde_per_mode.argmin(1)
        min_latent = gather_by_index(latent_raw_rmse, latent_best)
        min_ade, min_fde = gather_by_index(ade_per_mode, ade_best), gather_by_index(fde_per_mode, fde_best)
        ade_at, fde_at = gather_by_index(ade_per_mode, latent_best), gather_by_index(fde_per_mode, latent_best)
        latent_at_ade, latent_at_fde = gather_by_index(latent_raw_rmse, ade_best), gather_by_index(latent_raw_rmse, fde_best)
        ade_regret, fde_regret = ade_at - min_ade, fde_at - min_fde
        rank_ade = index_rank(latent_raw_rmse, ade_best)
        rank_fde = index_rank(latent_raw_rmse, fde_best)
        rank_latent_in_ade = index_rank(ade_per_mode, latent_best)

        # Direct ADE-FDE selection consistency.
        ade_at_fde_best = gather_by_index(ade_per_mode, fde_best)
        fde_at_ade_best = gather_by_index(fde_per_mode, ade_best)
        ade_regret_using_fde_best = ade_at_fde_best - min_ade
        fde_regret_using_ade_best = fde_at_ade_best - min_fde
        rank_ade_best_in_fde = index_rank(fde_per_mode, ade_best)
        rank_fde_best_in_ade = index_rank(ade_per_mode, fde_best)

        self.num_agents += n
        self.num_candidate_pairs += n * modes
        self.raw_latent_flat.append(latent_raw_rmse.detach().cpu().flatten())
        self.ade_flat.append(ade_per_mode.detach().cpu().flatten())
        self.fde_flat.append(fde_per_mode.detach().cpu().flatten())
        per_dim_cpu = per_dim_sq_error_raw.detach().cpu()
        for d in range(self.latent_elements):
            self.per_dim_sq_flat[d].append(per_dim_cpu[:, :, d].flatten())
        for destination, value in ((self.agent_pearson_raw_ade, p_ade), (self.agent_spearman_raw_ade, s_ade),
                                   (self.agent_pearson_raw_fde, p_fde), (self.agent_spearman_raw_fde, s_fde),
                                   (self.agent_pearson_ade_fde, p_ade_fde), (self.agent_spearman_ade_fde, s_ade_fde),
                                   (self.min_latent, min_latent), (self.min_ade, min_ade), (self.min_fde, min_fde),
                                   (self.ade_at_latent_best, ade_at), (self.fde_at_latent_best, fde_at),
                                   (self.latent_at_ade_best, latent_at_ade), (self.latent_at_fde_best, latent_at_fde),
                                   (self.ade_regret, ade_regret), (self.fde_regret, fde_regret),
                                   (self.rank_ade_best_in_latent, rank_ade), (self.rank_fde_best_in_latent, rank_fde),
                                   (self.rank_latent_best_in_ade, rank_latent_in_ade),
                                   (self.ade_at_fde_best, ade_at_fde_best), (self.fde_at_ade_best, fde_at_ade_best),
                                   (self.ade_regret_using_fde_best, ade_regret_using_fde_best),
                                   (self.fde_regret_using_ade_best, fde_regret_using_ade_best),
                                   (self.rank_ade_best_in_fde, rank_ade_best_in_fde),
                                   (self.rank_fde_best_in_ade, rank_fde_best_in_ade)):
            destination.append(value.detach().cpu())
        self.exact_ade += int((latent_best == ade_best).sum())
        self.exact_fde += int((latent_best == fde_best).sum())
        self.exact_ade_fde += int((ade_best == fde_best).sum())
        for k in self.hit_ks:
            ke = min(k, modes)
            self.hit_ade[k] += int((rank_ade < ke).sum())
            self.hit_fde[k] += int((rank_fde < ke).sum())
            self.hit_latent_in_ade[k] += int((rank_latent_in_ade < ke).sum())
            self.hit_ade_in_fde[k] += int((rank_ade_best_in_fde < ke).sum())
            self.hit_fde_in_ade[k] += int((rank_fde_best_in_ade < ke).sum())
        if self.collect_agent_rows:
            valid_steps_cpu = valid_steps.detach().cpu()
            for i in range(n):
                self.agent_rows.append({"alignment_target": self.label, "agent_index": self.num_agents - n + i,
                    "scenario_id": scenario_ids[i], "valid_steps": int(valid_steps_cpu[i]),
                    "pearson_raw_latent_ADE": float(p_ade[i]), "spearman_raw_latent_ADE": float(s_ade[i]),
                    "pearson_raw_latent_FDE": float(p_fde[i]), "spearman_raw_latent_FDE": float(s_fde[i]),
                    "pearson_ADE_FDE": float(p_ade_fde[i]), "spearman_ADE_FDE": float(s_ade_fde[i]),
                    "raw_latent_best_candidate": int(latent_best[i]), "ADE_best_candidate": int(ade_best[i]),
                    "FDE_best_candidate": int(fde_best[i]), "min_raw_latent_RMSE": float(min_latent[i]),
                    "min_ADE_m": float(min_ade[i]), "min_FDE_m": float(min_fde[i]),
                    "ADE_at_raw_latent_nearest_m": float(ade_at[i]), "FDE_at_raw_latent_nearest_m": float(fde_at[i]),
                    "ADE_regret_raw_latent_nearest_m": float(ade_regret[i]), "FDE_regret_raw_latent_nearest_m": float(fde_regret[i]),
                    "ADE_at_FDE_best_m": float(ade_at_fde_best[i]), "FDE_at_ADE_best_m": float(fde_at_ade_best[i]),
                    "ADE_regret_using_FDE_best_m": float(ade_regret_using_fde_best[i]),
                    "FDE_regret_using_ADE_best_m": float(fde_regret_using_ade_best[i]),
                    "ADE_best_equals_FDE_best": int(ade_best[i] == fde_best[i]),
                    "rank_ADE_best_in_raw_latent_0based": int(rank_ade[i]), "rank_FDE_best_in_raw_latent_0based": int(rank_fde[i]),
                    "rank_raw_latent_best_in_ADE_0based": int(rank_latent_in_ade[i]),
                    "rank_ADE_best_in_FDE_order_0based": int(rank_ade_best_in_fde[i]),
                    "rank_FDE_best_in_ADE_order_0based": int(rank_fde_best_in_ade[i])})

    def finalize(self) -> Dict[str, Any]:
        if self.num_agents == 0:
            raise RuntimeError(f"No focal agents were evaluated for {self.label}.")
        raw = torch.cat(self.raw_latent_flat)
        ade, fde = torch.cat(self.ade_flat), torch.cat(self.fde_flat)
        rank_raw, rank_ade, rank_fde = rank_1d(raw), rank_1d(ade), rank_1d(fde)
        per_dim = []
        for d in range(self.latent_elements):
            error = torch.cat(self.per_dim_sq_flat[d])
            rank_error = rank_1d(error)
            per_dim.append({"flat_latent_dimension": d, "pearson_raw_sq_error_ADE": pearson_1d(error, ade),
                            "spearman_raw_sq_error_ADE": pearson_1d(rank_error, rank_ade),
                            "pearson_raw_sq_error_FDE": pearson_1d(error, fde),
                            "spearman_raw_sq_error_FDE": pearson_1d(rank_error, rank_fde),
                            "mean_raw_sq_error": float(error.mean())})
        cat = lambda xs: torch.cat(xs)
        return {"label": self.label,
            "counts": {"num_batches": self.num_batches, "num_focal_agents": self.num_agents,
                       "num_candidate_pairs": self.num_candidate_pairs, "latent_elements": self.latent_elements},
            "pooled_candidate_pair_correlations": {
                "pearson_raw_latent_ADE": pearson_1d(raw, ade), "spearman_raw_latent_ADE": pearson_1d(rank_raw, rank_ade),
                "pearson_raw_latent_FDE": pearson_1d(raw, fde), "spearman_raw_latent_FDE": pearson_1d(rank_raw, rank_fde)},
            "per_agent_within_candidates_correlations": {
                "pearson_raw_latent_ADE": scalar_summary(cat(self.agent_pearson_raw_ade)),
                "spearman_raw_latent_ADE": scalar_summary(cat(self.agent_spearman_raw_ade)),
                "pearson_raw_latent_FDE": scalar_summary(cat(self.agent_pearson_raw_fde)),
                "spearman_raw_latent_FDE": scalar_summary(cat(self.agent_spearman_raw_fde)),
                "pearson_ADE_FDE": scalar_summary(cat(self.agent_pearson_ade_fde)),
                "spearman_ADE_FDE": scalar_summary(cat(self.agent_spearman_ade_fde))},
            "oracle_and_raw_latent_selection": {
                "min_raw_latent_RMSE": scalar_summary(cat(self.min_latent)), "min_ADE_m": scalar_summary(cat(self.min_ade)),
                "min_FDE_m": scalar_summary(cat(self.min_fde)), "ADE_at_raw_latent_nearest_m": scalar_summary(cat(self.ade_at_latent_best)),
                "FDE_at_raw_latent_nearest_m": scalar_summary(cat(self.fde_at_latent_best)),
                "raw_latent_RMSE_at_ADE_best": scalar_summary(cat(self.latent_at_ade_best)),
                "raw_latent_RMSE_at_FDE_best": scalar_summary(cat(self.latent_at_fde_best)),
                "ADE_regret_using_raw_latent_nearest_m": scalar_summary(cat(self.ade_regret)),
                "FDE_regret_using_raw_latent_nearest_m": scalar_summary(cat(self.fde_regret)),
                "rank_ADE_best_in_raw_latent_order_0based": scalar_summary(cat(self.rank_ade_best_in_latent)),
                "rank_FDE_best_in_raw_latent_order_0based": scalar_summary(cat(self.rank_fde_best_in_latent)),
                "rank_raw_latent_best_in_ADE_order_0based": scalar_summary(cat(self.rank_latent_best_in_ade)),
                "exact_raw_latent_best_equals_ADE_best_rate": self.exact_ade / self.num_agents,
                "exact_raw_latent_best_equals_FDE_best_rate": self.exact_fde / self.num_agents,
                "ADE_best_in_raw_latent_topk_rate": {str(k): self.hit_ade[k] / self.num_agents for k in self.hit_ks},
                "FDE_best_in_raw_latent_topk_rate": {str(k): self.hit_fde[k] / self.num_agents for k in self.hit_ks},
                "raw_latent_best_in_ADE_topk_rate": {str(k): self.hit_latent_in_ade[k] / self.num_agents for k in self.hit_ks}},
            "direct_ADE_FDE_consistency": {
                "pooled_pearson_ADE_FDE": pearson_1d(ade, fde),
                "pooled_spearman_ADE_FDE": pearson_1d(rank_ade, rank_fde),
                "per_agent_pearson_ADE_FDE": scalar_summary(cat(self.agent_pearson_ade_fde)),
                "per_agent_spearman_ADE_FDE": scalar_summary(cat(self.agent_spearman_ade_fde)),
                "ADE_at_FDE_best_m": scalar_summary(cat(self.ade_at_fde_best)),
                "FDE_at_ADE_best_m": scalar_summary(cat(self.fde_at_ade_best)),
                "ADE_regret_using_FDE_best_m": scalar_summary(cat(self.ade_regret_using_fde_best)),
                "FDE_regret_using_ADE_best_m": scalar_summary(cat(self.fde_regret_using_ade_best)),
                "rank_ADE_best_in_FDE_order_0based": scalar_summary(cat(self.rank_ade_best_in_fde)),
                "rank_FDE_best_in_ADE_order_0based": scalar_summary(cat(self.rank_fde_best_in_ade)),
                "ADE_best_equals_FDE_best_rate": self.exact_ade_fde / self.num_agents,
                "ADE_best_in_FDE_topk_rate": {str(k): self.hit_ade_in_fde[k] / self.num_agents for k in self.hit_ks},
                "FDE_best_in_ADE_topk_rate": {str(k): self.hit_fde_in_ade[k] / self.num_agents for k in self.hit_ks}},
            "per_raw_latent_dimension": per_dim}


def write_agent_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No per-agent rows were collected.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate(model: QCNetFM, loader: Iterable, device: torch.device, precision: str, num_candidates: int,
             radius_min: float, radius_max: float, radius_distribution: str, decode_chunk_size: int,
             seed: int, max_batches: int, log_interval: int, focal_category: int, trajectory_scale: float,
             hit_ks: Sequence[int], collect_agent_rows: bool) -> Tuple[RawAlignmentAccumulator, RawAlignmentAccumulator, Dict[str, Any]]:
    set_seed(seed)
    model.eval(); model.latent_encoder.eval(); model.latent_decoder.eval(); get_raw_decoder(model).eval()
    latent_elements = int(model.vae_num_intents * model.latent_dim)
    gt_acc = RawAlignmentAccumulator(latent_elements, hit_ks, collect_agent_rows, "ground_truth")
    recon_acc = RawAlignmentAccumulator(latent_elements, hit_ks, collect_agent_rows, "center_reconstruction")
    center_ade_all: List[torch.Tensor] = []; center_fde_all: List[torch.Tensor] = []; radii_all: List[torch.Tensor] = []
    for batch_index, data in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        if isinstance(data, Batch):
            data["agent"]["av_index"] += data["agent"]["ptr"][:-1]
        data = data.to(device)
        future_mask = data["agent"]["predict_mask"][:, model.num_historical_steps:].bool()
        eval_mask = (data["agent"]["category"] == int(focal_category)) & future_mask.any(-1)
        gt_acc.num_batches += 1; recon_acc.num_batches += 1
        if not eval_mask.any():
            continue
        target_norm = data["agent"]["target"][..., :model.output_dim].float() / float(trajectory_scale)
        with autocast_context(device, precision):
            latent_target_raw = model.latent_encoder.encode(target_norm, predict_mask=future_mask).float()
        latent_eval_raw = latent_target_raw[eval_mask]
        mask_eval = future_mask[eval_mask]
        target_eval_m = target_norm[eval_mask].float() * float(trajectory_scale)
        center_m = decode_raw(model, latent_eval_raw).float() * float(trajectory_scale)
        candidates_raw, radii = sample_candidate_latents_raw(latent_eval_raw, num_candidates, radius_min, radius_max, radius_distribution)
        candidate_m = decode_candidate_latents_raw(model, candidates_raw, decode_chunk_size).float() * float(trajectory_scale)
        raw_rmse, per_dim_sq_raw = latent_distance_raw(candidates_raw, latent_eval_raw)
        ade_gt, fde_gt = trajectory_distance_per_mode(candidate_m, target_eval_m, mask_eval)
        ade_recon, fde_recon = trajectory_distance_per_mode(candidate_m, center_m, mask_eval)
        center_ade, center_fde = trajectory_distance_per_mode(center_m.unsqueeze(1), target_eval_m, mask_eval)
        scenario_ids = try_scenario_ids(data, eval_mask)
        common = {"latent_raw_rmse": raw_rmse, "per_dim_sq_error_raw": per_dim_sq_raw,
                  "valid_steps": mask_eval.sum(-1), "scenario_ids": scenario_ids}
        gt_acc.update(ade_per_mode=ade_gt, fde_per_mode=fde_gt, **common)
        recon_acc.update(ade_per_mode=ade_recon, fde_per_mode=fde_recon, **common)
        center_ade_all.append(center_ade.squeeze(1).cpu()); center_fde_all.append(center_fde.squeeze(1).cpu()); radii_all.append(radii.cpu().flatten())
        if log_interval > 0 and (batch_index == 0 or (batch_index + 1) % log_interval == 0):
            gt_s = torch.cat(gt_acc.agent_spearman_raw_ade); gt_s = gt_s[torch.isfinite(gt_s)].mean().item()
            rc_s = torch.cat(recon_acc.agent_spearman_raw_ade); rc_s = rc_s[torch.isfinite(rc_s)].mean().item()
            gt_r = torch.cat(gt_acc.ade_regret).mean().item(); rc_r = torch.cat(recon_acc.ade_regret).mean().item()
            gt_fde_to_ade = torch.cat(gt_acc.ade_regret_using_fde_best).mean().item()
            rc_fde_to_ade = torch.cat(recon_acc.ade_regret_using_fde_best).mean().item()
            ca = torch.cat(center_ade_all).mean().item()
            print(f"[seed={seed}] batch={batch_index + 1:>5d} agents={gt_acc.num_agents:>7,d} centerADE={ca:.6f} "
                  f"GT_raw_Spearman={gt_s:.6f} GT_raw_regret={gt_r:.6f} GT_FDE_to_ADE_regret={gt_fde_to_ade:.6f} "
                  f"Recon_raw_Spearman={rc_s:.6f} Recon_raw_regret={rc_r:.6f} "
                  f"Recon_FDE_to_ADE_regret={rc_fde_to_ade:.6f}")
    if not center_ade_all:
        raise RuntimeError("No focal agents were evaluated.")
    diagnostics = {"center_reconstruction_ADE_m": scalar_summary(torch.cat(center_ade_all)),
                   "center_reconstruction_FDE_m": scalar_summary(torch.cat(center_fde_all)),
                   "sampled_raw_radius": scalar_summary(torch.cat(radii_all))}
    return gt_acc, recon_acc, diagnostics


def print_alignment_summary(title: str, result: Dict[str, Any]) -> None:
    counts = result["counts"]; pooled = result["pooled_candidate_pair_correlations"]
    per_agent = result["per_agent_within_candidates_correlations"]
    selection = result["oracle_and_raw_latent_selection"]
    ade_fde = result["direct_ADE_FDE_consistency"]
    print("\n" + "=" * 100); print(title); print("=" * 100)
    print("\nCounts")
    for key, value in counts.items():
        print(f"  {key:<32s}: {value:,}" if isinstance(value, int) else f"  {key:<32s}: {value}")
    print("\nRaw-space pooled candidate-pair correlations")
    for key, value in pooled.items():
        print(f"  {key:<38s}: {value:.6f}")
    print("\nRaw-space per-agent within-candidate correlations")
    for key, summary in per_agent.items():
        print(f"  {key:<38s}: mean={summary['mean']:.6f}, median={summary['median']:.6f}, q10={summary['quantiles']['0.10']:.6f}, q90={summary['quantiles']['0.90']:.6f}")
    print("\nOracle and raw-latent-nearest selection")
    keys = ("min_raw_latent_RMSE", "min_ADE_m", "min_FDE_m", "ADE_at_raw_latent_nearest_m",
            "FDE_at_raw_latent_nearest_m", "ADE_regret_using_raw_latent_nearest_m",
            "FDE_regret_using_raw_latent_nearest_m", "rank_ADE_best_in_raw_latent_order_0based")
    for key in keys:
        s = selection[key]
        print(f"  {key:<48s}: mean={s['mean']:.6f}, median={s['median']:.6f}, q90={s['quantiles']['0.90']:.6f}")
    print("\nExact agreement rates")
    print(f"  raw-latent-best == ADE-best : {selection['exact_raw_latent_best_equals_ADE_best_rate']:.2%}")
    print(f"  raw-latent-best == FDE-best : {selection['exact_raw_latent_best_equals_FDE_best_rate']:.2%}")
    print("\nADE-best contained in raw-latent top-K")
    for key, value in selection["ADE_best_in_raw_latent_topk_rate"].items():
        print(f"  K={int(key):>3d}: {value:.2%}")

    print("\nDirect ADE-FDE consistency")
    print(f"  pooled_pearson_ADE_FDE                         : {ade_fde['pooled_pearson_ADE_FDE']:.6f}")
    print(f"  pooled_spearman_ADE_FDE                        : {ade_fde['pooled_spearman_ADE_FDE']:.6f}")
    for key in ("per_agent_pearson_ADE_FDE", "per_agent_spearman_ADE_FDE", "ADE_at_FDE_best_m",
                "FDE_at_ADE_best_m", "ADE_regret_using_FDE_best_m", "FDE_regret_using_ADE_best_m",
                "rank_ADE_best_in_FDE_order_0based", "rank_FDE_best_in_ADE_order_0based"):
        summary = ade_fde[key]
        print(f"  {key:<48s}: mean={summary['mean']:.6f}, median={summary['median']:.6f}, "
              f"q90={summary['quantiles']['0.90']:.6f}")
    print(f"  ADE-best == FDE-best                            : {ade_fde['ADE_best_equals_FDE_best_rate']:.2%}")
    print("\nADE-best contained in FDE top-K")
    for key, value in ade_fde["ADE_best_in_FDE_topk_rate"].items():
        print(f"  K={int(key):>3d}: {value:.2%}")
    print("\nFDE-best contained in ADE top-K")
    for key, value in ade_fde["FDE_best_in_ADE_topk_rate"].items():
        print(f"  K={int(key):>3d}: {value:.2%}")

    print("\nPer-raw-latent-dimension squared-error correlation with ADE")
    for row in result["per_raw_latent_dimension"]:
        print(f"  dim={row['flat_latent_dimension']:>2d} Pearson={row['pearson_raw_sq_error_ADE']:+.6f} "
              f"Spearman={row['spearman_raw_sq_error_ADE']:+.6f} meanSqErr={row['mean_raw_sq_error']:.6f}")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.ckpt)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if args.num_candidates < 2:
        raise ValueError("At least 2 candidates are required.")
    hit_ks = sorted({max(1, min(int(k), args.num_candidates)) for k in args.hit_ks})
    device = torch.device(args.device)
    model, hparams = load_model(checkpoint_path, device, args.allow_shape_mismatch)
    val_loader = build_datamodule(hparams, args).val_dataloader()
    print(f"checkpoint          : {checkpoint_path}")
    print(f"device              : {device}")
    print(f"precision           : {args.precision}")
    print(f"seed                : {args.seed}")
    print(f"sampling_space      : raw latent")
    print(f"selection_space     : raw latent")
    print(f"num_candidates      : {args.num_candidates}")
    print(f"raw_radius_range    : [{args.radius_min}, {args.radius_max}]")
    print(f"radius_distribution : {args.radius_distribution}")
    print(f"decode_chunk_size   : {args.decode_chunk_size}")
    print(f"hit_ks              : {hit_ks}")
    print(f"trajectory_scale    : {args.trajectory_scale}")
    print(f"z_mean(reference)   : {model.z_mean.detach().cpu().flatten().tolist()}")
    print(f"z_std(reference)    : {model.z_std.detach().cpu().flatten().tolist()}")
    gt_acc, recon_acc, diagnostics = evaluate(model, val_loader, device, args.precision, args.num_candidates,
        args.radius_min, args.radius_max, args.radius_distribution, args.decode_chunk_size, args.seed,
        args.max_batches, args.log_interval, args.focal_category, args.trajectory_scale, hit_ks,
        args.output_agent_csv_prefix is not None)
    gt_result, recon_result = gt_acc.finalize(), recon_acc.finalize()
    print("\n" + "=" * 100); print("Center VAE reconstruction diagnostics"); print("=" * 100)
    for key, summary in diagnostics.items():
        print(f"  {key:<38s}: mean={summary['mean']:.6f}, median={summary['median']:.6f}, q90={summary['quantiles']['0.90']:.6f}")
    print_alignment_summary("RAW VAE-only alignment: candidate trajectory vs ground truth", gt_result)
    print_alignment_summary("RAW VAE-only alignment: candidate trajectory vs center reconstruction", recon_result)
    output = {"config": {"checkpoint": str(checkpoint_path.resolve()), "device": str(device), "precision": args.precision,
        "seed": args.seed, "num_candidates": args.num_candidates, "radius_min_raw": args.radius_min,
        "radius_max_raw": args.radius_max, "radius_distribution": args.radius_distribution,
        "focal_category": args.focal_category, "trajectory_scale": args.trajectory_scale, "hit_ks": hit_ks,
        "max_batches": args.max_batches, "candidate_sampling": "isotropic Euclidean perturbation in raw VAE latent space",
        "latent_distance": "RMSE in raw VAE latent space", "nearest_selection": "raw-latent RMSE",
        "trajectory_distance_to_gt": "ADE/FDE to ground truth in meters",
        "trajectory_distance_to_reconstruction": "ADE/FDE to D(z_raw_gt) in meters"},
        "latent_statistics_reference_only": {"z_mean": model.z_mean.detach().cpu().tolist(), "z_std": model.z_std.detach().cpu().tolist()},
        "diagnostics": diagnostics, "alignment_to_ground_truth": gt_result,
        "alignment_to_center_reconstruction": recon_result}
    if args.output_json:
        path = Path(args.output_json); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON saved to: {path.resolve()}")
    if args.output_agent_csv_prefix:
        prefix = Path(args.output_agent_csv_prefix)
        gt_path = prefix.parent / f"{prefix.name}_gt.csv"
        recon_path = prefix.parent / f"{prefix.name}_reconstruction.csv"
        write_agent_csv(gt_path, gt_acc.agent_rows); write_agent_csv(recon_path, recon_acc.agent_rows)
        print(f"Per-agent GT CSV saved to: {gt_path.resolve()}")
        print(f"Per-agent reconstruction CSV saved to: {recon_path.resolve()}")


if __name__ == "__main__":
    main()