#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为纯 Transformer 确定性 latent 回归头生成“预合并 scene batch”缓存。

缓存直接复用：
    model.fm_decoder._build_graph_context(data, scene_enc)

生成阶段先按 scene 提取上下文，再在写 shard 时自动完成：
    1. 拼接 x_m / x_t_hist / x_pl；
    2. 修正 t2a、pl2a、a2a 边索引偏移；
    3. 拼接 latent target、轨迹和掩码。

最终每个 shard_*.pt 本身就是一个可直接训练的 batch，顶层不再保存 scenes 列表。
训练 DataLoader 使用 batch_size=None 即可，无需自定义 collate_fn，也无需在模型中再次合并图。

重要：
1. 默认 compact_sources=True，只保留实际出现在 t2a/pl2a 边中的历史和地图源节点，
   可显著减小回归头缓存体积。
2. compact_sources=True 适用于确定性回归头，但不能直接传给
   QCNetFMDecoder._forward_core()，因为后者要求完整 [N_a*T_h,H] 的 x_t_hist。
3. 如需同时为冻结 encoder 的 FM decoder 缓存完整上下文，请使用 --full_sources。
4. latent 坐标系为 centered-raw：z_target = z_raw - z_mean，不除以 z_std。
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch_geometric.data import Batch

from datamodules import ArgoverseV2DataModule
from predictors import QCNetFM


CACHE_FORMAT_VERSION = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成预合并 scene-batch 确定性 latent 回归缓存。"
    )

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)

    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--train_raw_dir", type=str, default=None)
    parser.add_argument("--train_processed_dir", type=str, default=None)
    parser.add_argument("--val_raw_dir", type=str, default=None)
    parser.add_argument("--val_processed_dir", type=str, default=None)
    parser.add_argument("--test_raw_dir", type=str, default=None)
    parser.add_argument("--test_processed_dir", type=str, default=None)

    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument(
        "--scenes_per_shard",
        type=int,
        default=64,
        help="每个预合并训练 batch（shard）包含的场景数量。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--precision",
        choices=["32", "bf16", "fp16"],
        default="bf16",
        help="encoder 与图上下文提取时的推理精度。",
    )
    parser.add_argument(
        "--feature_storage_dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help=(
            "x_m/x_t_hist/x_pl/r_* 等冻结场景特征的保存精度。"
            "latent target、轨迹 target 始终保存为 float32。"
        ),
    )

    parser.add_argument(
        "--full_sources",
        action="store_true",
        default=False,
        help=(
            "保存每个场景完整的 x_t_hist 和 x_pl，并保持原始局部源索引。"
            "开启后可供冻结 encoder 的 FM decoder 复用，但磁盘占用显著增大。"
            "默认仅保存 t2a/pl2a 边实际使用的源节点，适合确定性回归头。"
        ),
    )

    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=25)

    parser.add_argument(
        "--keep_existing_shards",
        action="store_true",
        default=False,
        help="保留旧 shard 并从下一个编号继续写入；通常不建议使用。",
    )

    return parser.parse_args()


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"checkpoint 必须为 dict，实际为 {type(checkpoint)}"
        )

    return checkpoint


def autocast_ctx(
    device: torch.device,
    precision: str,
):
    if device.type != "cuda" or precision == "32":
        return contextlib.nullcontext()

    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "当前 CUDA 设备不支持 bfloat16，"
                "请使用 --precision 32 或 fp16。"
            )
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        raise ValueError(f"未知 precision：{precision}")

    return torch.autocast(
        device_type="cuda",
        dtype=dtype,
    )


def resolve_storage_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if name not in table:
        raise ValueError(f"未知 feature_storage_dtype：{name}")
    return table[name]


def tensor_to_cpu(
    value: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if dtype is not None and value.is_floating_point():
        value = value.to(dtype=dtype)
    return value.detach().contiguous().cpu()


def get_store_ptr(
    store,
    num_graphs: int,
) -> torch.Tensor:
    """
    获取 PyG Batch 某个 node store 的 ptr。
    优先使用 store['ptr']；不存在时由 batch vector 计算。
    """
    ptr = store.get("ptr", None)
    if ptr is not None:
        return ptr.long()

    batch = store.get("batch", None)
    if batch is None:
        if num_graphs != 1:
            raise RuntimeError(
                "缺少 ptr 和 batch，无法恢复多场景节点边界。"
            )
        num_nodes = int(store.num_nodes)
        return torch.tensor(
            [0, num_nodes],
            device=next(iter(store.values())).device,
            dtype=torch.long,
        )

    counts = torch.bincount(
        batch.long(),
        minlength=num_graphs,
    )
    return torch.cat(
        [
            torch.zeros(
                1,
                device=batch.device,
                dtype=torch.long,
            ),
            counts.cumsum(dim=0),
        ],
        dim=0,
    )


def get_scenario_ids(
    data: Batch,
    num_graphs: int,
    split: str,
    first_global_scene_index: int,
) -> List[str]:
    value = None

    try:
        value = data["scenario_id"]
    except Exception:
        value = getattr(data, "scenario_id", None)

    if value is None:
        return [
            f"{split}_{first_global_scene_index + index:08d}"
            for index in range(num_graphs)
        ]

    if isinstance(value, str):
        values = [value]
    elif isinstance(value, torch.Tensor):
        values = value.detach().cpu().view(-1).tolist()
    else:
        values = list(value)

    if len(values) != num_graphs:
        return [
            f"{split}_{first_global_scene_index + index:08d}"
            for index in range(num_graphs)
        ]

    return [str(item) for item in values]


def merge_scene_records(
    scenes: List[Dict[str, Any]],
    split: str,
) -> Dict[str, Any]:
    """将若干完整场景预合并为一个可直接训练的稀疏图 batch。"""
    if not scenes:
        raise ValueError("不能合并空场景列表。")

    num_intents = int(scenes[0]["num_intents"])
    hidden_dim = int(scenes[0]["hidden_dim"])
    compact_sources = bool(scenes[0]["compact_sources"])

    x_m_parts: List[torch.Tensor] = []
    x_t_hist_parts: List[torch.Tensor] = []
    x_pl_parts: List[torch.Tensor] = []

    edge_t2a_parts: List[torch.Tensor] = []
    edge_pl2a_parts: List[torch.Tensor] = []
    edge_a2a_parts: List[torch.Tensor] = []
    r_t2a_parts: List[torch.Tensor] = []
    r_pl2a_parts: List[torch.Tensor] = []
    r_a2a_parts: List[torch.Tensor] = []

    map_gate_parts: List[torch.Tensor] = []
    threat_gate_parts: List[torch.Tensor] = []
    has_map_gate = False
    has_threat_gate = False

    z_target_parts: List[torch.Tensor] = []
    target_parts: List[torch.Tensor] = []
    predict_mask_parts: List[torch.Tensor] = []
    valid_mask_parts: List[torch.Tensor] = []
    category_parts: List[torch.Tensor] = []
    eval_mask_parts: List[torch.Tensor] = []
    agent_batch_parts: List[torch.Tensor] = []

    scenario_ids: List[str] = []
    agent_ptr = [0]
    history_source_ptr = [0]
    map_source_ptr = [0]

    agent_offset = 0
    history_source_offset = 0
    map_source_offset = 0
    total_valid_agents = 0

    for scene_index, scene in enumerate(scenes):
        if int(scene["num_intents"]) != num_intents:
            raise ValueError("同一 shard 内 num_intents 不一致。")
        if int(scene["hidden_dim"]) != hidden_dim:
            raise ValueError("同一 shard 内 hidden_dim 不一致。")
        if bool(scene["compact_sources"]) != compact_sources:
            raise ValueError("同一 shard 内 compact_sources 配置不一致。")

        x_m = scene["x_m"]
        x_t_hist = scene["x_t_hist"]
        x_pl = scene["x_pl"]
        num_agents = int(x_m.size(0))
        expanded_agent_offset = agent_offset * num_intents

        edge_t2a = scene["edge_index_t2a_exp"].clone()
        edge_t2a[0] += history_source_offset
        edge_t2a[1] += expanded_agent_offset

        edge_pl2a = scene["edge_index_pl2a_exp"].clone()
        edge_pl2a[0] += map_source_offset
        edge_pl2a[1] += expanded_agent_offset

        edge_a2a = scene["edge_index_a2a_exp"].clone()
        edge_a2a += expanded_agent_offset

        x_m_parts.append(x_m)
        x_t_hist_parts.append(x_t_hist)
        x_pl_parts.append(x_pl)
        edge_t2a_parts.append(edge_t2a)
        edge_pl2a_parts.append(edge_pl2a)
        edge_a2a_parts.append(edge_a2a)
        r_t2a_parts.append(scene["r_t2a_exp"])
        r_pl2a_parts.append(scene["r_pl2a_exp"])
        r_a2a_parts.append(scene["r_a2a_exp"])

        map_gate = scene.get("edge_map_exp")
        if map_gate is None:
            map_gate_parts.append(
                torch.ones(
                    edge_pl2a.size(1),
                    dtype=scene["r_pl2a_exp"].dtype,
                )
            )
        else:
            has_map_gate = True
            map_gate_parts.append(map_gate)

        threat_gate = scene.get("edge_threat_exp")
        if threat_gate is None:
            threat_gate_parts.append(
                torch.ones(
                    edge_a2a.size(1),
                    dtype=scene["r_a2a_exp"].dtype,
                )
            )
        else:
            has_threat_gate = True
            threat_gate_parts.append(threat_gate)

        z_target_parts.append(scene["z_target_centered_raw"])
        target_parts.append(scene["target"])
        predict_mask_parts.append(scene["predict_mask"])
        valid_mask_parts.append(scene["valid_agent_mask"])
        category_parts.append(scene["category"])
        eval_mask_parts.append(scene["eval_mask"])
        agent_batch_parts.append(
            torch.full(
                (num_agents,),
                scene_index,
                dtype=torch.long,
            )
        )

        scenario_ids.append(str(scene["scenario_id"]))
        total_valid_agents += int(scene["num_valid_agents"])
        agent_offset += num_agents
        history_source_offset += int(x_t_hist.size(0))
        map_source_offset += int(x_pl.size(0))
        agent_ptr.append(agent_offset)
        history_source_ptr.append(history_source_offset)
        map_source_ptr.append(map_source_offset)

    payload: Dict[str, Any] = {
        "format_version": CACHE_FORMAT_VERSION,
        "cache_type": "merged_scene_context_batch",
        "split": split,
        "num_scenes": len(scenes),
        "num_agents": agent_offset,
        "num_valid_agents": total_valid_agents,
        "num_intents": num_intents,
        "hidden_dim": hidden_dim,
        "compact_sources": compact_sources,
        "scenario_ids": scenario_ids,
        "agent_ptr": torch.tensor(agent_ptr, dtype=torch.long),
        "history_source_ptr": torch.tensor(history_source_ptr, dtype=torch.long),
        "map_source_ptr": torch.tensor(map_source_ptr, dtype=torch.long),
        "agent_batch": torch.cat(agent_batch_parts, dim=0),
        "x_m": torch.cat(x_m_parts, dim=0).contiguous(),
        "x_t_hist": torch.cat(x_t_hist_parts, dim=0).contiguous(),
        "x_pl": torch.cat(x_pl_parts, dim=0).contiguous(),
        "edge_index_t2a_exp": torch.cat(edge_t2a_parts, dim=1).contiguous(),
        "r_t2a_exp": torch.cat(r_t2a_parts, dim=0).contiguous(),
        "edge_index_pl2a_exp": torch.cat(edge_pl2a_parts, dim=1).contiguous(),
        "r_pl2a_exp": torch.cat(r_pl2a_parts, dim=0).contiguous(),
        "edge_index_a2a_exp": torch.cat(edge_a2a_parts, dim=1).contiguous(),
        "r_a2a_exp": torch.cat(r_a2a_parts, dim=0).contiguous(),
        "edge_map_exp": (
            torch.cat(map_gate_parts, dim=0).contiguous()
            if has_map_gate
            else None
        ),
        "edge_threat_exp": (
            torch.cat(threat_gate_parts, dim=0).contiguous()
            if has_threat_gate
            else None
        ),
        "z_target_centered_raw": torch.cat(z_target_parts, dim=0).contiguous(),
        "target": torch.cat(target_parts, dim=0).contiguous(),
        "predict_mask": torch.cat(predict_mask_parts, dim=0).contiguous(),
        "valid_agent_mask": torch.cat(valid_mask_parts, dim=0).contiguous(),
        "category": torch.cat(category_parts, dim=0).contiguous(),
        "eval_mask": torch.cat(eval_mask_parts, dim=0).contiguous(),
    }

    validate_merged_batch(payload)
    return payload


def validate_merged_batch(batch: Dict[str, Any]) -> None:
    num_agents = int(batch["num_agents"])
    num_intents = int(batch["num_intents"])
    hidden_dim = int(batch["hidden_dim"])
    expanded_agents = num_agents * num_intents

    if tuple(batch["x_m"].shape) != (num_agents, hidden_dim):
        raise RuntimeError(
            f"合并后 x_m 形状错误：{tuple(batch['x_m'].shape)}"
        )
    if batch["z_target_centered_raw"].size(0) != num_agents:
        raise RuntimeError("合并后 latent target agent 数量错误。")

    for prefix, source_key in (("t2a", "x_t_hist"), ("pl2a", "x_pl")):
        edge = batch[f"edge_index_{prefix}_exp"]
        relation = batch[f"r_{prefix}_exp"]
        if relation.size(0) != edge.size(1):
            raise RuntimeError(f"合并后 {prefix} relation/edge 数量不一致。")
        if edge.numel() > 0:
            if int(edge[0].max()) >= batch[source_key].size(0):
                raise RuntimeError(f"合并后 {prefix} source index 越界。")
            if int(edge[1].max()) >= expanded_agents:
                raise RuntimeError(f"合并后 {prefix} destination index 越界。")

    edge_a2a = batch["edge_index_a2a_exp"]
    if batch["r_a2a_exp"].size(0) != edge_a2a.size(1):
        raise RuntimeError("合并后 a2a relation/edge 数量不一致。")
    if edge_a2a.numel() > 0 and int(edge_a2a.max()) >= expanded_agents:
        raise RuntimeError("合并后 a2a edge index 越界。")


class SceneShardWriter:
    """缓存生成阶段自动合并场景；每个 shard 本身就是训练 batch。"""

    def __init__(
        self,
        out_dir: Path,
        split: str,
        scenes_per_shard: int,
        keep_existing_shards: bool,
    ) -> None:
        if scenes_per_shard <= 0:
            raise ValueError("scenes_per_shard 必须大于 0。")

        self.split = split
        self.out_dir = out_dir / split
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.scenes_per_shard = scenes_per_shard

        old_shards = sorted(self.out_dir.glob("shard_*.pt"))
        if keep_existing_shards:
            self.shard_index = (
                max(int(path.stem.split("_")[-1]) for path in old_shards) + 1
                if old_shards
                else 0
            )
        else:
            for path in old_shards:
                path.unlink()
            if old_shards:
                print(f"[{split}] 已删除 {len(old_shards)} 个旧缓存 shard。")
            self.shard_index = 0

        self.start_shard_index = self.shard_index
        self.written_shards = 0
        self.buffer: List[Dict[str, Any]] = []
        self.total_scenes = 0
        self.total_agents = 0
        self.total_valid_agents = 0

    def add(self, scene: Dict[str, Any]) -> None:
        self.buffer.append(scene)
        self.total_scenes += 1
        self.total_agents += int(scene["num_agents"])
        self.total_valid_agents += int(scene["num_valid_agents"])
        if len(self.buffer) >= self.scenes_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return

        payload = merge_scene_records(
            scenes=self.buffer,
            split=self.split,
        )
        path = self.out_dir / f"shard_{self.shard_index:06d}.pt"
        torch.save(payload, path)

        self.shard_index += 1
        self.written_shards += 1
        self.buffer = []

    def close(self) -> Dict[str, int]:
        self._flush()
        return {
            "num_shards": self.written_shards,
            "first_shard_index": self.start_shard_index,
            "next_shard_index": self.shard_index,
            "num_scenes": self.total_scenes,
            "num_agents": self.total_agents,
            "num_valid_agents": self.total_valid_agents,
        }


class LatentStatistics:
    def __init__(self, latent_dim: int) -> None:
        self.latent_dim = latent_dim
        self.count = 0
        self.sum = torch.zeros(latent_dim, dtype=torch.float64)
        self.square_sum = torch.zeros(latent_dim, dtype=torch.float64)

    def update(
        self,
        latent: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        selected = latent[valid_mask]
        if selected.numel() == 0:
            return

        flat = selected.detach().double().reshape(
            -1,
            self.latent_dim,
        )
        self.count += flat.size(0)
        self.sum += flat.sum(dim=0).cpu()
        self.square_sum += flat.pow(2).sum(dim=0).cpu()

    def compute(self) -> Dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": [],
                "std": [],
                "global_rms": 0.0,
            }

        mean = self.sum / self.count
        second_moment = self.square_sum / self.count
        variance = (
            second_moment - mean.pow(2)
        ).clamp_min(0.0)

        return {
            "count": int(self.count),
            "mean": mean.tolist(),
            "std": variance.sqrt().tolist(),
            "global_rms": float(
                second_moment.mean().sqrt().item()
            ),
        }


def select_scene_edges(
    edge_index: torch.Tensor,
    relation: torch.Tensor,
    dst_start: int,
    dst_end: int,
    src_start: int,
    source_features: torch.Tensor,
    full_sources: bool,
    src_end: Optional[int] = None,
    edge_gate: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    """
    从 batch 全局 expanded edge 中提取单场景边。

    full_sources=False：
        仅保留实际使用的 source feature，并把 source index 压缩到 [0,N_used)。

    full_sources=True：
        保存场景完整 source feature，并减去场景 source offset。
    """
    if edge_index is None:
        raise ValueError("edge_index 不能为 None。")
    if relation is None:
        raise ValueError("relation 不能为 None。")

    edge_mask = (
        (edge_index[1] >= dst_start)
        & (edge_index[1] < dst_end)
    )

    scene_edge = edge_index[:, edge_mask].clone()
    scene_relation = relation[edge_mask]

    if edge_gate is None:
        scene_gate = None
    else:
        scene_gate = edge_gate[edge_mask]

    scene_edge[1] -= dst_start

    if full_sources:
        if src_end is None:
            raise ValueError(
                "full_sources=True 时必须提供 src_end。"
            )
        scene_source_features = source_features[
            src_start:src_end
        ]
        scene_edge[0] -= src_start
    else:
        if scene_edge.size(1) == 0:
            scene_source_features = source_features[:0]
            scene_edge = torch.empty(
                (2, 0),
                device=edge_index.device,
                dtype=edge_index.dtype,
            )
        else:
            unique_source, inverse = torch.unique(
                scene_edge[0],
                sorted=True,
                return_inverse=True,
            )
            scene_source_features = source_features[
                unique_source
            ]
            scene_edge[0] = inverse

    return (
        scene_source_features,
        scene_edge,
        scene_relation,
        scene_gate,
    )


def slice_a2a_edges(
    edge_index: torch.Tensor,
    relation: torch.Tensor,
    expanded_agent_start: int,
    expanded_agent_end: int,
    edge_gate: Optional[torch.Tensor],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    edge_mask = (
        (edge_index[1] >= expanded_agent_start)
        & (edge_index[1] < expanded_agent_end)
    )

    scene_edge = edge_index[:, edge_mask].clone()
    scene_edge[0] -= expanded_agent_start
    scene_edge[1] -= expanded_agent_start

    scene_relation = relation[edge_mask]

    if edge_gate is None:
        scene_gate = None
    else:
        scene_gate = edge_gate[edge_mask]

    return scene_edge, scene_relation, scene_gate


def validate_scene_record(
    scene: Dict[str, Any],
    hidden_dim: int,
    latent_dim: int,
    num_intents: int,
) -> None:
    num_agents = int(scene["num_agents"])

    if scene["x_m"].shape != (num_agents, hidden_dim):
        raise RuntimeError(
            f"x_m 形状错误：{tuple(scene['x_m'].shape)}"
        )

    expected_latent = (
        num_agents,
        num_intents,
        latent_dim,
    )
    if tuple(scene["z_target_centered_raw"].shape) != expected_latent:
        raise RuntimeError(
            "z_target_centered_raw 形状错误："
            f"{tuple(scene['z_target_centered_raw'].shape)}，"
            f"预期 {expected_latent}"
        )

    edge_specs = [
        ("t2a", "x_t_hist"),
        ("pl2a", "x_pl"),
    ]

    for prefix, source_key in edge_specs:
        edge = scene[f"edge_index_{prefix}_exp"]
        relation = scene[f"r_{prefix}_exp"]
        source = scene[source_key]

        if edge.ndim != 2 or edge.size(0) != 2:
            raise RuntimeError(
                f"edge_index_{prefix}_exp 形状错误。"
            )
        if relation.size(0) != edge.size(1):
            raise RuntimeError(
                f"{prefix} relation 与 edge 数量不一致。"
            )
        if edge.numel() > 0:
            if int(edge[0].max()) >= source.size(0):
                raise RuntimeError(
                    f"{prefix} source index 越界。"
                )
            if int(edge[1].max()) >= num_agents * num_intents:
                raise RuntimeError(
                    f"{prefix} destination index 越界。"
                )

    a2a_edge = scene["edge_index_a2a_exp"]
    a2a_relation = scene["r_a2a_exp"]

    if a2a_relation.size(0) != a2a_edge.size(1):
        raise RuntimeError("a2a relation 与 edge 数量不一致。")
    if a2a_edge.numel() > 0:
        upper = num_agents * num_intents
        if int(a2a_edge.max()) >= upper:
            raise RuntimeError("a2a edge index 越界。")


def extract_scene_records_from_batch(
    model: QCNetFM,
    data: Batch,
    scene_enc: Dict[str, torch.Tensor],
    context: Dict[str, torch.Tensor],
    z_target_centered_raw: torch.Tensor,
    target: torch.Tensor,
    predict_mask: torch.Tensor,
    scenario_ids: List[str],
    storage_dtype: torch.dtype,
    full_sources: bool,
) -> List[Dict[str, Any]]:
    num_graphs = int(data.num_graphs)
    num_intents = int(model.vae_num_intents)
    hidden_dim = int(model.hidden_dim)
    num_historical_steps = int(model.num_historical_steps)

    agent_ptr = get_store_ptr(
        data["agent"],
        num_graphs=num_graphs,
    )
    polygon_ptr = get_store_ptr(
        data["map_polygon"],
        num_graphs=num_graphs,
    )

    x_m_global = scene_enc["x_a"][:, -1, :]
    x_t_hist_global = context["x_t_hist"]
    x_pl_global = context["x_pl"]

    expected_hist_rows = (
        x_m_global.size(0)
        * num_historical_steps
    )
    if x_t_hist_global.size(0) != expected_hist_rows:
        raise RuntimeError(
            "x_t_hist 与 agent 数量不一致："
            f"actual={x_t_hist_global.size(0)}, "
            f"expected={expected_hist_rows}"
        )

    category_global = data["agent"]["category"].long()
    valid_global = predict_mask.any(dim=-1)

    records: List[Dict[str, Any]] = []

    for scene_index in range(num_graphs):
        a0 = int(agent_ptr[scene_index].item())
        a1 = int(agent_ptr[scene_index + 1].item())
        p0 = int(polygon_ptr[scene_index].item())
        p1 = int(polygon_ptr[scene_index + 1].item())

        num_agents = a1 - a0
        num_polygons = p1 - p0

        if num_agents <= 0:
            continue

        scene_valid = valid_global[a0:a1]
        if not bool(scene_valid.any()):
            continue

        expanded_a0 = a0 * num_intents
        expanded_a1 = a1 * num_intents

        hist_start = a0 * num_historical_steps
        hist_end = a1 * num_historical_steps

        (
            scene_x_t_hist,
            scene_edge_t2a,
            scene_r_t2a,
            _,
        ) = select_scene_edges(
            edge_index=context["edge_index_t2a_exp"],
            relation=context["r_t2a_exp"],
            dst_start=expanded_a0,
            dst_end=expanded_a1,
            src_start=hist_start,
            src_end=hist_end,
            source_features=x_t_hist_global,
            full_sources=full_sources,
            edge_gate=None,
        )

        (
            scene_x_pl,
            scene_edge_pl2a,
            scene_r_pl2a,
            scene_edge_map,
        ) = select_scene_edges(
            edge_index=context["edge_index_pl2a_exp"],
            relation=context["r_pl2a_exp"],
            dst_start=expanded_a0,
            dst_end=expanded_a1,
            src_start=p0,
            src_end=p1,
            source_features=x_pl_global,
            full_sources=full_sources,
            edge_gate=context.get("edge_map_exp"),
        )

        (
            scene_edge_a2a,
            scene_r_a2a,
            scene_edge_threat,
        ) = slice_a2a_edges(
            edge_index=context["edge_index_a2a_exp"],
            relation=context["r_a2a_exp"],
            expanded_agent_start=expanded_a0,
            expanded_agent_end=expanded_a1,
            edge_gate=context.get("edge_threat_exp"),
        )

        scene_category = category_global[a0:a1]
        scene_eval = (
            (scene_category == 3)
            & scene_valid
        )

        record: Dict[str, Any] = {
            "scenario_id": scenario_ids[scene_index],
            "num_agents": num_agents,
            "num_polygons_original": num_polygons,
            "num_valid_agents": int(
                scene_valid.sum().item()
            ),
            "num_intents": num_intents,
            "hidden_dim": hidden_dim,
            "compact_sources": not full_sources,

            # Transformer query 和 a2a source 使用。
            "x_m": tensor_to_cpu(
                x_m_global[a0:a1],
                dtype=storage_dtype,
            ),

            # t2a / pl2a source。
            "x_t_hist": tensor_to_cpu(
                scene_x_t_hist,
                dtype=storage_dtype,
            ),
            "x_pl": tensor_to_cpu(
                scene_x_pl,
                dtype=storage_dtype,
            ),

            "edge_index_t2a_exp": tensor_to_cpu(
                scene_edge_t2a,
            ),
            "r_t2a_exp": tensor_to_cpu(
                scene_r_t2a,
                dtype=storage_dtype,
            ),

            "edge_index_pl2a_exp": tensor_to_cpu(
                scene_edge_pl2a,
            ),
            "r_pl2a_exp": tensor_to_cpu(
                scene_r_pl2a,
                dtype=storage_dtype,
            ),

            "edge_index_a2a_exp": tensor_to_cpu(
                scene_edge_a2a,
            ),
            "r_a2a_exp": tensor_to_cpu(
                scene_r_a2a,
                dtype=storage_dtype,
            ),

            "edge_map_exp": (
                None
                if scene_edge_map is None
                else tensor_to_cpu(
                    scene_edge_map,
                    dtype=storage_dtype,
                )
            ),
            "edge_threat_exp": (
                None
                if scene_edge_threat is None
                else tensor_to_cpu(
                    scene_edge_threat,
                    dtype=storage_dtype,
                )
            ),

            "z_target_centered_raw": tensor_to_cpu(
                z_target_centered_raw[a0:a1],
                dtype=torch.float32,
            ),
            "target": tensor_to_cpu(
                target[a0:a1],
                dtype=torch.float32,
            ),
            "predict_mask": tensor_to_cpu(
                predict_mask[a0:a1].bool(),
            ),
            "valid_agent_mask": tensor_to_cpu(
                scene_valid.bool(),
            ),
            "category": tensor_to_cpu(
                scene_category,
            ),
            "eval_mask": tensor_to_cpu(
                scene_eval.bool(),
            ),
        }

        validate_scene_record(
            record,
            hidden_dim=hidden_dim,
            latent_dim=int(model.latent_dim),
            num_intents=num_intents,
        )
        records.append(record)

    return records


@torch.inference_mode()
def build_split(
    model: QCNetFM,
    loader: Iterable,
    split: str,
    out_dir: Path,
    scenes_per_shard: int,
    device: torch.device,
    precision: str,
    storage_dtype: torch.dtype,
    full_sources: bool,
    max_batches: int,
    log_interval: int,
    keep_existing_shards: bool,
) -> Dict[str, Any]:
    writer = SceneShardWriter(
        out_dir=out_dir,
        split=split,
        scenes_per_shard=scenes_per_shard,
        keep_existing_shards=keep_existing_shards,
    )

    latent_statistics = LatentStatistics(
        latent_dim=model.latent_dim,
    )

    model.encoder.eval()
    model.fm_decoder.eval()
    model.latent_encoder.eval()

    processed_batches = 0
    global_scene_index = 0

    for batch_idx, data in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        if not isinstance(data, Batch):
            raise TypeError(
                "scene cache 生成要求 DataLoader 返回 PyG Batch，"
                f"实际为 {type(data)}"
            )

        data["agent"]["av_index"] += (
            data["agent"]["ptr"][:-1]
        )

        data = data.to(device)

        target = (
            data["agent"]["target"][
                ...,
                : model.output_dim,
            ]
            / model.trajectory_scale
        )

        predict_mask = data["agent"]["predict_mask"][
            :,
            model.num_historical_steps :,
        ].bool()

        with autocast_ctx(
            device=device,
            precision=precision,
        ):
            scene_enc = model.encoder(data)
            context = model.fm_decoder._build_graph_context(
                data=data,
                scene_enc=scene_enc,
            )
            z_raw = model.latent_encoder.encode(
                target,
                predict_mask=predict_mask,
            )

        z_target_centered_raw = (
            z_raw.float()
            - model.z_mean.float()
        )

        valid_agent_mask = predict_mask.any(dim=-1)
        latent_statistics.update(
            z_target_centered_raw,
            valid_mask=valid_agent_mask,
        )

        num_graphs = int(data.num_graphs)
        scenario_ids = get_scenario_ids(
            data=data,
            num_graphs=num_graphs,
            split=split,
            first_global_scene_index=global_scene_index,
        )

        records = extract_scene_records_from_batch(
            model=model,
            data=data,
            scene_enc=scene_enc,
            context=context,
            z_target_centered_raw=z_target_centered_raw,
            target=target,
            predict_mask=predict_mask,
            scenario_ids=scenario_ids,
            storage_dtype=storage_dtype,
            full_sources=full_sources,
        )

        for record in records:
            writer.add(record)

        global_scene_index += num_graphs
        processed_batches += 1

        if (
            log_interval > 0
            and (
                processed_batches == 1
                or processed_batches % log_interval == 0
            )
        ):
            print(
                f"[{split}] "
                f"batches={processed_batches}, "
                f"scenes={writer.total_scenes:,}, "
                f"agents={writer.total_agents:,}, "
                f"valid_agents={writer.total_valid_agents:,}, "
                f"written_shards={writer.written_shards}"
            )

    writer_stats = writer.close()
    latent_stats = latent_statistics.compute()

    print(f"\n[{split}] centered-raw latent statistics")
    print(
        "mean:",
        [
            round(value, 6)
            for value in latent_stats["mean"]
        ],
    )
    print(
        "std:",
        [
            round(value, 6)
            for value in latent_stats["std"]
        ],
    )
    print(
        "global_rms:",
        round(
            float(latent_stats["global_rms"]),
            6,
        ),
    )
    print(
        "checkpoint z_std:",
        [
            round(float(value), 6)
            for value in model.z_std.detach()
            .cpu()
            .view(-1)
            .tolist()
        ],
    )

    return {
        **writer_stats,
        "num_processed_batches": processed_batches,
        "latent_statistics": latent_stats,
    }


def main() -> None:
    args = parse_args()

    torch.set_float32_matmul_precision("high")

    ckpt_path = Path(args.ckpt)
    cache_dir = Path(args.cache_dir)

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"checkpoint 不存在：{ckpt_path}"
        )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.keep_existing_shards:
        manifest_path = cache_dir / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()

    checkpoint = load_checkpoint(ckpt_path)

    hparams = dict(
        checkpoint.get(
            "hyper_parameters",
            {},
        )
    )
    if not hparams:
        raise RuntimeError(
            "checkpoint 中缺少 hyper_parameters。"
        )

    print("正在构建 checkpoint 对应的 QCNetFM")
    print(
        f"hidden_dim={hparams.get('hidden_dim')}, "
        f"latent_dim={hparams.get('latent_dim')}, "
        f"vae_num_intents={hparams.get('vae_num_intents')}"
    )

    model = QCNetFM(**hparams)
    state_dict = checkpoint.get(
        "state_dict",
        checkpoint,
    )

    incompatible = model.load_state_dict(
        state_dict,
        strict=True,
    )
    if incompatible.missing_keys:
        raise RuntimeError(
            f"missing keys：{incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected keys：{incompatible.unexpected_keys}"
        )

    device = torch.device(args.device)
    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "指定了 CUDA，但当前 CUDA 不可用。"
        )

    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    storage_dtype = resolve_storage_dtype(
        args.feature_storage_dtype
    )

    print("\nCheckpoint latent configuration")
    print(
        "z_mean:",
        model.z_mean.detach().cpu().view(-1).tolist(),
    )
    print(
        "z_std:",
        model.z_std.detach().cpu().view(-1).tolist(),
    )
    print(
        "latent coordinate system: "
        "centered_raw = z_raw - z_mean"
    )
    print(
        "cache source mode:",
        "full_sources" if args.full_sources else "compact_sources",
    )
    print(
        "feature storage dtype:",
        args.feature_storage_dtype,
    )

    cfg = dict(hparams)

    overrides = {
        "root": args.root,
        "train_raw_dir": args.train_raw_dir,
        "train_processed_dir": args.train_processed_dir,
        "val_raw_dir": args.val_raw_dir,
        "val_processed_dir": args.val_processed_dir,
        "test_raw_dir": args.test_raw_dir,
        "test_processed_dir": args.test_processed_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    cfg["train_batch_size"] = args.train_batch_size
    cfg["val_batch_size"] = args.val_batch_size
    cfg.setdefault(
        "test_batch_size",
        args.val_batch_size,
    )
    cfg["num_workers"] = args.num_workers
    cfg["shuffle"] = False
    cfg["pin_memory"] = device.type == "cuda"
    cfg["persistent_workers"] = args.num_workers > 0

    if not cfg.get("root"):
        raise ValueError(
            "未找到数据集 root，请通过 --root 指定。"
        )

    datamodule = ArgoverseV2DataModule(**cfg)
    datamodule.setup(stage="fit")

    train_stats = build_split(
        model=model,
        loader=datamodule.train_dataloader(),
        split="train",
        out_dir=cache_dir,
        scenes_per_shard=args.scenes_per_shard,
        device=device,
        precision=args.precision,
        storage_dtype=storage_dtype,
        full_sources=args.full_sources,
        max_batches=args.max_train_batches,
        log_interval=args.log_interval,
        keep_existing_shards=args.keep_existing_shards,
    )

    val_stats = build_split(
        model=model,
        loader=datamodule.val_dataloader(),
        split="val",
        out_dir=cache_dir,
        scenes_per_shard=args.scenes_per_shard,
        device=device,
        precision=args.precision,
        storage_dtype=storage_dtype,
        full_sources=args.full_sources,
        max_batches=args.max_val_batches,
        log_interval=args.log_interval,
        keep_existing_shards=args.keep_existing_shards,
    )

    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "cache_type": "merged_scene_context_latent_regression",
        "checkpoint": str(ckpt_path.resolve()),

        "latent_coordinate_system": "centered_raw",
        "latent_target_formula": "z_raw - z_mean",
        "latent_target_divided_by_std": False,
        "latent_target_cache_key": "z_target_centered_raw",

        "context_builder": (
            "QCNetFM.fm_decoder._build_graph_context"
        ),
        "shard_is_ready_training_batch": True,
        "training_dataloader_batch_size": None,
        "requires_custom_collate_fn": False,
        "context_is_fourier_embedded": True,
        "context_edges_are_intent_expanded": True,
        "compact_sources": not args.full_sources,
        "direct_fm_forward_core_compatible": bool(
            args.full_sources
        ),
        "fm_compatibility_note": (
            "full_sources=True 时可在冻结 encoder 和冻结 relation embedding "
            "的前提下复用缓存上下文调用 fm_decoder._forward_core；"
            "该模式主要用于冻结 encoder 后复用上下文。"
            if args.full_sources
            else
            "默认 compact_sources 缓存只保留 t2a/pl2a 实际使用源节点，"
            "适用于确定性回归头，不可直接传入 fm_decoder._forward_core。"
        ),

        "hidden_dim": int(model.hidden_dim),
        "latent_dim": int(model.latent_dim),
        "num_intents": int(model.vae_num_intents),
        "num_historical_steps": int(
            model.num_historical_steps
        ),
        "num_future_steps": int(model.num_future_steps),
        "output_dim": int(model.output_dim),
        "trajectory_scale": float(
            model.trajectory_scale
        ),

        "z_mean": (
            model.z_mean.detach()
            .cpu()
            .view(-1)
            .tolist()
        ),
        "z_std": (
            model.z_std.detach()
            .cpu()
            .view(-1)
            .tolist()
        ),

        "precision_used_for_extraction": args.precision,
        "feature_storage_dtype": (
            args.feature_storage_dtype
        ),
        "scenes_per_shard": int(
            args.scenes_per_shard
        ),

        "train": train_stats,
        "val": val_stats,
    }

    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n预合并 scene-batch 缓存生成完成")
    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        f"\nmanifest: {manifest_path.resolve()}"
    )


if __name__ == "__main__":
    main()