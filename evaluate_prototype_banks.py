#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import torch


def resolve_bank_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_dir(): path = path / "prototype_bank.pt"
    if not path.is_file(): raise FileNotFoundError(f"找不到 prototype bank：{path}")
    return path


def to_float_tensor(value: Any, name: str) -> torch.Tensor:
    if value is None: raise KeyError(f"prototype bank 缺少 {name}")
    return torch.as_tensor(value).detach().cpu().float()


def flatten_prototypes(value: Any) -> torch.Tensor:
    x = to_float_tensor(value, "prototype_latents_centered_raw")
    if x.ndim == 3 and x.size(1) == 1: x = x[:, 0]
    if x.ndim != 2: raise ValueError(f"prototype latent 应为 [M,D] 或 [M,1,D]，实际 {tuple(x.shape)}")
    return x


def flatten_stat(value: Any, M: int, D: int, name: str, default: float = 0.0) -> torch.Tensor:
    if value is None: return torch.full((M, D), float(default), dtype=torch.float32)
    x = torch.as_tensor(value).detach().cpu().float()
    if x.ndim == 3 and x.size(1) == 1: x = x[:, 0]
    if x.ndim == 1 and x.numel() == D: x = x.view(1, D).expand(M, D).clone()
    if x.shape != (M, D): raise ValueError(f"{name} 应为 [M,D] 或 [M,1,D]，实际 {tuple(x.shape)}")
    return x


def finite_np(x: torch.Tensor) -> np.ndarray:
    a = x.detach().cpu().double().numpy().reshape(-1)
    return a[np.isfinite(a)]


def qstats(x: torch.Tensor, prefix: str) -> Dict[str, float]:
    a = finite_np(x)
    if a.size == 0: return {f"{prefix}_{k}": float("nan") for k in ("min", "p10", "p25", "median", "p75", "p90", "max", "mean")}
    qs = np.quantile(a, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {f"{prefix}_min": float(a.min()), f"{prefix}_p10": float(qs[0]), f"{prefix}_p25": float(qs[1]), f"{prefix}_median": float(qs[2]), f"{prefix}_p75": float(qs[3]), f"{prefix}_p90": float(qs[4]), f"{prefix}_max": float(a.max()), f"{prefix}_mean": float(a.mean())}


def weighted_mean(x: torch.Tensor, w: torch.Tensor) -> float:
    m = torch.isfinite(x) & torch.isfinite(w) & w.gt(0)
    if not m.any(): return float("nan")
    return float((x[m].double() * w[m].double()).sum() / w[m].double().sum())


def gini_coefficient(counts: torch.Tensor) -> float:
    x = counts.detach().cpu().double().clamp_min(0).sort().values
    n = x.numel(); total = x.sum()
    if n == 0 or total <= 0: return float("nan")
    index = torch.arange(1, n + 1, dtype=torch.float64)
    return float((2.0 * (index * x).sum() / (n * total)) - (n + 1.0) / n)


def count_metrics(counts: torch.Tensor, prefix: str = "count") -> Dict[str, float]:
    counts = counts.detach().cpu().double().clamp_min(0)
    total = counts.sum(); M = counts.numel()
    if total <= 0: return {f"{prefix}_total": 0.0, f"{prefix}_effective_k": 0.0, f"{prefix}_effective_ratio": 0.0}
    p = counts / total; nz = p.gt(0); entropy = -(p[nz] * p[nz].log()).sum(); effective_k = entropy.exp(); sorted_p = p.sort(descending=True).values
    bottom_n = max(1, int(math.ceil(M * 0.25))); top10_n = max(1, int(math.ceil(M * 0.10)))
    out = qstats(counts.float(), prefix)
    out.update({f"{prefix}_total": float(total), f"{prefix}_nonempty": float(nz.sum()), f"{prefix}_empty_ratio": float((~nz).float().mean()), f"{prefix}_max_min_ratio": float(counts.max() / counts[nz].min()) if nz.any() else float("inf"), f"{prefix}_entropy": float(entropy), f"{prefix}_effective_k": float(effective_k), f"{prefix}_effective_ratio": float(effective_k / max(M, 1)), f"{prefix}_gini": gini_coefficient(counts), f"{prefix}_top1_share": float(sorted_p[:1].sum()), f"{prefix}_top6_share": float(sorted_p[:min(6, M)].sum()), f"{prefix}_top10pct_share": float(sorted_p[:top10_n].sum()), f"{prefix}_bottom25pct_share": float(sorted_p[-bottom_n:].sum())})
    return out


def nearest_latent_neighbors(proto: torch.Tensor, chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    M = proto.size(0); best_d = torch.full((M,), float("inf")); best_j = torch.full((M,), -1, dtype=torch.long)
    for i0 in range(0, M, chunk_size):
        i1 = min(i0 + chunk_size, M); dist = torch.cdist(proto[i0:i1].float(), proto.float(), p=2)
        row = torch.arange(i1 - i0); col = torch.arange(i0, i1); dist[row, col] = float("inf")
        d, j = dist.min(dim=1); best_d[i0:i1] = d; best_j[i0:i1] = j
    return best_d, best_j


def nearest_trajectory_neighbors(traj_m: torch.Tensor, chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    traj = traj_m[..., :2].float(); M = traj.size(0); best_ade = torch.full((M,), float("inf")); best_fde = torch.full((M,), float("inf")); best_j = torch.full((M,), -1, dtype=torch.long)
    for i0 in range(0, M, chunk_size):
        i1 = min(i0 + chunk_size, M); diff = traj[i0:i1, None] - traj[None]
        dist = diff.square().sum(dim=-1).clamp_min(1e-12).sqrt(); ade = dist.mean(dim=-1); fde = dist[..., -1]
        row = torch.arange(i1 - i0); col = torch.arange(i0, i1); ade[row, col] = float("inf")
        a, j = ade.min(dim=1); best_ade[i0:i1] = a; best_j[i0:i1] = j; best_fde[i0:i1] = fde[row, j]
    return best_ade, best_fde, best_j


def pair_duplicate_rates_latent(proto: torch.Tensor, thresholds: Sequence[float], chunk_size: int) -> Dict[str, float]:
    M = proto.size(0); counts = {float(t): 0 for t in thresholds}; total = M * (M - 1) // 2
    for i0 in range(0, M, chunk_size):
        i1 = min(i0 + chunk_size, M); dist = torch.cdist(proto[i0:i1].float(), proto.float(), p=2)
        rows = torch.arange(i0, i1).view(-1, 1); cols = torch.arange(M).view(1, -1); upper = cols > rows
        vals = dist[upper]
        for t in thresholds: counts[float(t)] += int(vals.lt(float(t)).sum())
    return {f"latent_pair_rate_lt_{t:g}": counts[float(t)] / max(total, 1) for t in thresholds}


def pair_duplicate_rates_traj(traj_m: torch.Tensor, thresholds: Sequence[float], chunk_size: int) -> Dict[str, float]:
    traj = traj_m[..., :2].float(); M = traj.size(0); counts = {float(t): 0 for t in thresholds}; total = M * (M - 1) // 2
    for i0 in range(0, M, chunk_size):
        i1 = min(i0 + chunk_size, M); dist = (traj[i0:i1, None] - traj[None]).square().sum(dim=-1).clamp_min(1e-12).sqrt().mean(dim=-1)
        rows = torch.arange(i0, i1).view(-1, 1); cols = torch.arange(M).view(1, -1); vals = dist[cols > rows]
        for t in thresholds: counts[float(t)] += int(vals.lt(float(t)).sum())
    return {f"traj_pair_rate_lt_{t:g}m": counts[float(t)] / max(total, 1) for t in thresholds}


def load_manifest(bank_path: Path) -> Optional[Dict[str, Any]]:
    json_path = bank_path.parent / "manifest.json"; pt_path = bank_path.parent / "manifest.pt"
    if json_path.is_file(): return json.loads(json_path.read_text(encoding="utf-8"))
    if pt_path.is_file(): return torch.load(pt_path, map_location="cpu", weights_only=False)
    return None


def manifest_metrics(manifest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not manifest: return out
    for split, stats in manifest.get("splits", {}).items():
        for key in ("num_scenes", "num_valid_agents", "num_assigned_agents", "mean_match_latent_raw_l2", "mean_match_ade_m", "mean_match_fde_m", "mean_match_traj_score_m"):
            if key in stats: out[f"{split}_{key}"] = stats[key]
    return out


class Reservoir:
    def __init__(self, capacity: int, seed: int):
        self.capacity = max(1, int(capacity)); self.rng = np.random.default_rng(seed); self.values = np.empty((0,), dtype=np.float64); self.keys = np.empty((0,), dtype=np.float64); self.total = 0; self.sum = 0.0
    def update(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64).reshape(-1); x = x[np.isfinite(x)]
        if x.size == 0: return
        self.total += int(x.size); self.sum += float(x.sum()); keys = self.rng.random(x.size)
        all_v = np.concatenate([self.values, x]); all_k = np.concatenate([self.keys, keys])
        if all_v.size > self.capacity:
            keep = np.argpartition(all_k, -self.capacity)[-self.capacity:]; all_v = all_v[keep]; all_k = all_k[keep]
        self.values, self.keys = all_v, all_k
    def summary(self, prefix: str) -> Dict[str, float]:
        if self.total == 0: return {f"{prefix}_count": 0, f"{prefix}_mean": float("nan")}
        q = np.quantile(self.values, [0.5, 0.9, 0.95, 0.99])
        return {f"{prefix}_count": self.total, f"{prefix}_mean": self.sum / self.total, f"{prefix}_median_approx": float(q[0]), f"{prefix}_p90_approx": float(q[1]), f"{prefix}_p95_approx": float(q[2]), f"{prefix}_p99_approx": float(q[3]), f"{prefix}_reservoir_size": int(self.values.size)}


def scan_assignment_split(split_dir: Path, split: str, M: int, reservoir_size: int, seed: int) -> Tuple[Dict[str, Any], torch.Tensor]:
    split_dir = split_dir.expanduser().resolve()
    paths = sorted(split_dir.glob("shard_*.pt"))
    if not paths:
        paths = sorted(split_dir.rglob("shard_*.pt"))
    if not paths:
        return {
            f"{split}_assignment_dir": str(split_dir),
            f"{split}_assignment_shards": 0,
        }, torch.zeros(M, dtype=torch.long)
    latent = Reservoir(reservoir_size, seed + 1); ade = Reservoir(reservoir_size, seed + 2); fde = Reservoir(reservoir_size, seed + 3); score = Reservoir(reservoir_size, seed + 4); counts = torch.zeros(M, dtype=torch.long); scenes = 0
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records = payload.get("scenes", []) if isinstance(payload, dict) else []
        scenes += len(records)
        for record in records:
            idx = torch.as_tensor(record["prototype_index"]).long().view(-1); valid = torch.as_tensor(record.get("valid_agent_mask", idx.ge(0))).bool().view(-1); valid = valid & idx.ge(0) & idx.lt(M)
            if not valid.any(): continue
            counts += torch.bincount(idx[valid], minlength=M)
            for key, stat in (("match_latent_raw_l2", latent), ("match_ade_m", ade), ("match_fde_m", fde), ("match_traj_score_m", score)):
                if key in record: stat.update(torch.as_tensor(record[key]).view(-1)[valid].double().numpy())
    out: Dict[str, Any] = {
        f"{split}_assignment_dir": str(split_dir),
        f"{split}_assignment_shards": len(paths),
        f"{split}_assignment_scenes": scenes,
    }
    out.update(latent.summary(f"{split}_match_latent_raw_l2")); out.update(ade.summary(f"{split}_match_ade_m")); out.update(fde.summary(f"{split}_match_fde_m")); out.update(score.summary(f"{split}_match_traj_score_m")); out.update(count_metrics(counts, f"{split}_assignment_count"))
    return out, counts


def analyze_bank(bank_path: Path, name: str, args: argparse.Namespace, out_dir: Path, assignment_root: Optional[Path] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    bank = torch.load(bank_path, map_location="cpu", weights_only=False); proto = flatten_prototypes(bank.get("prototype_latents_centered_raw")); M, D = proto.shape
    count_value = bank.get("prototype_residual_count", bank.get("kmeans_labels_count"))
    counts = None if count_value is None else torch.as_tensor(count_value).detach().cpu().long().view(-1)
    if counts is not None and counts.numel() != M: raise ValueError(f"count 长度 {counts.numel()} 与 prototype 数量 {M} 不一致")
    mean = flatten_stat(bank.get("prototype_residual_mean"), M, D, "prototype_residual_mean", 0.0); std = flatten_stat(bank.get("prototype_residual_std_population", bank.get("residual_std")), M, D, "prototype_residual_std_population", float("nan"))
    std_radius = std.square().sum(dim=-1).sqrt(); residual_rms_radius = (std.square() + mean.square()).sum(dim=-1).sqrt(); residual_rms_per_dim = residual_rms_radius / math.sqrt(D); residual_mean_norm = mean.square().sum(dim=-1).sqrt()
    latent_nn, latent_nn_idx = nearest_latent_neighbors(proto, args.chunk_size)
    self_sep = latent_nn / residual_rms_radius.clamp_min(1e-8); pair_sep = latent_nn / (residual_rms_radius + residual_rms_radius[latent_nn_idx]).clamp_min(1e-8)
    traj_value = bank.get("prototype_trajectories_m")
    if traj_value is None and bank.get("prototype_trajectories_normalized") is not None:
        traj_value = torch.as_tensor(bank["prototype_trajectories_normalized"]).float() * float(bank.get("trajectory_scale", 10.0))
    traj_m = torch.as_tensor(traj_value).detach().cpu().float() if traj_value is not None else None
    if traj_m is not None and traj_m.ndim == 4 and traj_m.size(1) == 1: traj_m = traj_m[:, 0]
    if traj_m is not None and (traj_m.ndim != 3 or traj_m.size(0) != M): raise ValueError(f"prototype trajectory 形状错误：{tuple(traj_m.shape)}")
    count_source = "prototype_residual_count" if "prototype_residual_count" in bank else ("kmeans_labels_count" if "kmeans_labels_count" in bank else "missing")
    summary: Dict[str, Any] = {"name": name, "bank_path": str(bank_path), "num_prototypes": M, "latent_dim": D, "agent_scope": bank.get("agent_scope"), "assignment_rule": bank.get("prototype_assignment_rule"), "kmeans_space": bank.get("kmeans_space"), "count_source": count_source}
    weight_counts = counts.float() if counts is not None else torch.ones(M, dtype=torch.float32)
    if counts is not None:
        summary.update(count_metrics(counts))
    else:
        for key in ("count_effective_k", "count_effective_ratio", "count_gini", "count_empty_ratio", "count_nonempty"):
            summary[key] = float("nan")
    summary.update(qstats(std_radius, "within_std_radius_l2")); summary.update(qstats(residual_rms_radius, "within_residual_rms_radius_l2")); summary.update(qstats(residual_rms_per_dim, "within_residual_rms_per_dim")); summary.update(qstats(residual_mean_norm, "residual_mean_norm_l2")); summary["within_residual_rms_radius_l2_weighted_mean"] = weighted_mean(residual_rms_radius, weight_counts); summary["within_residual_rms_per_dim_weighted_mean"] = weighted_mean(residual_rms_per_dim, weight_counts); summary["residual_mean_norm_l2_weighted_mean"] = weighted_mean(residual_mean_norm, weight_counts)
    summary.update(qstats(latent_nn, "latent_nn_l2")); summary.update(qstats(self_sep, "separation_self_ratio")); summary.update(qstats(pair_sep, "separation_pair_ratio")); summary["separation_self_ratio_lt_1"] = float(self_sep.lt(1).float().mean()); summary["separation_self_ratio_lt_2"] = float(self_sep.lt(2).float().mean()); summary["separation_pair_ratio_lt_1"] = float(pair_sep.lt(1).float().mean()); summary["separation_pair_ratio_lt_2"] = float(pair_sep.lt(2).float().mean())
    for t in args.latent_duplicate_thresholds: summary[f"latent_nn_rate_lt_{t:g}"] = float(latent_nn.lt(float(t)).float().mean())
    summary.update(pair_duplicate_rates_latent(proto, args.latent_duplicate_thresholds, args.chunk_size))
    traj_nn_ade = traj_nn_fde = traj_nn_idx = None
    if traj_m is not None:
        traj_nn_ade, traj_nn_fde, traj_nn_idx = nearest_trajectory_neighbors(traj_m, args.chunk_size); summary.update(qstats(traj_nn_ade, "traj_nn_ADE_m")); summary.update(qstats(traj_nn_fde, "traj_nn_FDE_m"))
        for t in args.traj_duplicate_thresholds: summary[f"traj_nn_ADE_rate_lt_{t:g}m"] = float(traj_nn_ade.lt(float(t)).float().mean())
        summary.update(pair_duplicate_rates_traj(traj_m, args.traj_duplicate_thresholds, args.chunk_size))
    medoid_gap = bank.get("kmeans_medoid_to_center_raw_l2")
    if medoid_gap is not None: summary.update(qstats(torch.as_tensor(medoid_gap).float(), "medoid_to_kmeans_center_l2"))
    summary.update(manifest_metrics(load_manifest(bank_path)))
    assignment_counts: Dict[str, torch.Tensor] = {}
    if args.scan_assignments:
        root = assignment_root if assignment_root is not None else bank_path.parent
        summary["assignment_root"] = str(root)
        for split in args.splits:
            split_metrics, split_counts = scan_assignment_split(root / split, split, M, args.reservoir_size, args.seed)
            summary.update(split_metrics)
            assignment_counts[split] = split_counts
        primary_split = "train" if assignment_counts.get("train", torch.empty(0)).sum().item() > 0 else ("val" if assignment_counts.get("val", torch.empty(0)).sum().item() > 0 else None)
        if primary_split is not None:
            counts = assignment_counts[primary_split]
            summary.update(count_metrics(counts))
            summary["count_source"] = f"{primary_split}_assignment_shards"
    per_proto: List[Dict[str, Any]] = []; neighbor_rows: List[Dict[str, Any]] = []
    for i in range(M):
        j = int(latent_nn_idx[i]); row = {"prototype": i, "count": int(counts[i]) if counts is not None else None, "residual_mean_norm_l2": float(residual_mean_norm[i]), "within_std_radius_l2": float(std_radius[i]), "within_residual_rms_radius_l2": float(residual_rms_radius[i]), "within_residual_rms_per_dim": float(residual_rms_per_dim[i]), "latent_nn_index": j, "latent_nn_l2": float(latent_nn[i]), "separation_self_ratio": float(self_sep[i]), "separation_pair_ratio": float(pair_sep[i])}
        if traj_nn_ade is not None: row.update({"traj_nn_index": int(traj_nn_idx[i]), "traj_nn_ADE_m": float(traj_nn_ade[i]), "traj_nn_FDE_m": float(traj_nn_fde[i])})
        per_proto.append(row); neighbor_rows.append({"prototype": i, "latent_nn_index": j, "latent_nn_l2": float(latent_nn[i]), "traj_nn_index": int(traj_nn_idx[i]) if traj_nn_idx is not None else None, "traj_nn_ADE_m": float(traj_nn_ade[i]) if traj_nn_ade is not None else None, "traj_nn_FDE_m": float(traj_nn_fde[i]) if traj_nn_fde is not None else None})
    bank_out = out_dir / name; bank_out.mkdir(parents=True, exist_ok=True); write_csv(bank_out / "per_prototype_metrics.csv", per_proto); write_csv(bank_out / "nearest_neighbors.csv", neighbor_rows); (bank_out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    return summary, per_proto, neighbor_rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows: return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def print_core_summary(rows: List[Dict[str, Any]]) -> None:
    keys = ["name", "num_prototypes", "val_mean_match_ade_m", "val_mean_match_fde_m", "within_residual_rms_per_dim_weighted_mean", "count_effective_k", "count_effective_ratio", "count_gini", "latent_nn_l2_median", "traj_nn_ADE_m_median", "separation_pair_ratio_median", "separation_pair_ratio_lt_1"]
    print("\n========== Prototype Bank Comparison ==========")
    print(" | ".join(keys))
    for row in sorted(rows, key=lambda x: int(x["num_prototypes"])):
        values=[]
        for key in keys:
            value=row.get(key, "NA"); values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        print(" | ".join(values))
    if len(rows) > 1:
        print("\n========== Incremental Changes (sorted by K) ==========")
        ordered = sorted(rows, key=lambda x: int(x["num_prototypes"]))
        for prev, cur in zip(ordered[:-1], ordered[1:]):
            print(f"K {prev['num_prototypes']} -> {cur['num_prototypes']}")
            for key in ("val_mean_match_ade_m", "val_mean_match_fde_m", "within_residual_rms_per_dim_weighted_mean", "count_effective_ratio", "traj_nn_ADE_m_median", "separation_pair_ratio_median"):
                if isinstance(prev.get(key), (int, float)) and isinstance(cur.get(key), (int, float)) and math.isfinite(float(prev[key])) and math.isfinite(float(cur[key])):
                    delta=float(cur[key])-float(prev[key]); rel=delta/max(abs(float(prev[key])),1e-12)*100.0; print(f"  {key}: {prev[key]:.6g} -> {cur[key]:.6g}  delta={delta:+.6g} ({rel:+.2f}%)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="直接读取一个或多个 prototype_bank.pt，评估量化误差、局部残差、类别不平衡、重复 prototype、cluster separation，并可扫描 assignment shards。")
    p.add_argument("--prototype_bank_paths", nargs="+", required=True, help="prototype_bank.pt 路径或其所在目录，可同时传多个 K。")
    p.add_argument("--names", nargs="*", default=None, help="可选显示名称，数量需与 bank 路径一致。")
    p.add_argument("--output_dir", type=str, default="prototype_bank_evaluation")
    p.add_argument("--chunk_size", type=int, default=128)
    p.add_argument("--latent_duplicate_thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    p.add_argument("--traj_duplicate_thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.50])
    p.add_argument("--scan_assignments", action=argparse.BooleanOptionalAction, default=False, help="扫描 assignment shard，补充真实类别计数、精确均值与近似分位数。")
    p.add_argument("--assignment_roots", nargs="*", default=None, help="每个 prototype bank 对应的 assignment 根目录；根目录下应包含 train/val/shard_*.pt。数量必须与 --prototype_bank_paths 相同。未提供时仍使用 bank 所在目录。")
    p.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    p.add_argument("--reservoir_size", type=int, default=500000, help="assignment 分位数的均匀 reservoir 大小。")
    p.add_argument("--seed", type=int, default=2030)
    return p.parse_args()


def main() -> None:
    args = parse_args(); bank_paths = [resolve_bank_path(x) for x in args.prototype_bank_paths]
    if args.names is not None and len(args.names) not in (0, len(bank_paths)): raise ValueError("--names 数量必须与 --prototype_bank_paths 一致。")
    if args.assignment_roots is not None and len(args.assignment_roots) not in (0, len(bank_paths)):
        raise ValueError("--assignment_roots 数量必须与 --prototype_bank_paths 一致。")
    assignment_roots = [Path(x).expanduser().resolve() for x in args.assignment_roots] if args.assignment_roots else [None] * len(bank_paths)
    names = args.names if args.names else [p.parent.name or p.stem for p in bank_paths]; out_dir = Path(args.output_dir).expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True); rows=[]
    for path, name, assignment_root in zip(bank_paths, names, assignment_roots):
        print(f"\n分析 {name}: {path}")
        if args.scan_assignments:
            print(f"assignment root: {assignment_root if assignment_root is not None else path.parent}")
        summary, _, _ = analyze_bank(path, name, args, out_dir, assignment_root=assignment_root); rows.append(summary)
    write_csv(out_dir / "prototype_bank_comparison.csv", rows); (out_dir / "prototype_bank_comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"); print_core_summary(rows); print(f"\n结果已保存到：{out_dir}")


if __name__ == "__main__": main()