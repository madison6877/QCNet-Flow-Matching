#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge prototype-assignment sidecar shards into a new QCNet processed dataset.

The original processed dataset is never modified. For each scene, this script:
1. loads the original processed .pkl/.pt file;
2. finds the matching sidecar record by scenario_id;
3. writes the prototype fields into data["agent"];
4. saves the merged scene under a new processed directory.

Default embedded fields are the fields needed by residual-FM / Top-2 selector
training. Diagnostic matching metrics can be added with --include_diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import torch
from tqdm import tqdm


TRAINING_FIELDS: Tuple[str, ...] = (
    "coarse_index",
    "prototype_index",
    "secondary_prototype_index",
    "support_prototype_ids",
    "support_weights",
    "support_size",
    "boundary_margin",
    "is_boundary",
    "valid_agent_mask",
    "z_gt_centered_raw",
    "z_residual",
)

DIAGNOSTIC_FIELDS: Tuple[str, ...] = (
    "match_latent_raw_l2",
    "second_match_latent_raw_l2",
    "match_ade_m",
    "match_fde_m",
    "match_traj_score_m",
)

SUPPORTED_SUFFIXES: Tuple[str, ...] = (".pkl", ".pickle", ".pt", ".pth")


def normalize_scenario_id(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if torch.is_tensor(value):
        if value.numel() == 1:
            return str(value.detach().cpu().item())
        return str(value.detach().cpu().tolist())
    return str(value)


def load_object(path: Path) -> Tuple[Any, str]:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        try:
            with path.open("rb") as handle:
                return pickle.load(handle), "pickle"
        except Exception as pickle_error:
            try:
                return torch.load(path, map_location="cpu", weights_only=False), "torch"
            except TypeError:
                return torch.load(path, map_location="cpu"), "torch"
            except Exception:
                raise RuntimeError(f"无法读取 processed 文件：{path}") from pickle_error
    try:
        return torch.load(path, map_location="cpu", weights_only=False), "torch"
    except TypeError:
        return torch.load(path, map_location="cpu"), "torch"


def atomic_save_object(obj: Any, path: Path, serializer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        if serializer == "pickle":
            with temp_path.open("wb") as handle:
                pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif serializer == "torch":
            torch.save(obj, temp_path)
        else:
            raise ValueError(f"未知 serializer={serializer}")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def get_agent_store(data: Any) -> Any:
    try:
        return data["agent"]
    except Exception as error:
        raise TypeError(
            "processed 场景中找不到 data['agent']。"
            f"对象类型={type(data)}"
        ) from error


def get_store_value(store: Any, key: str) -> Any:
    try:
        return store[key]
    except Exception:
        return getattr(store, key, None)


def get_num_agents(data: Any) -> int:
    store = get_agent_store(data)
    num_nodes = getattr(store, "num_nodes", None)
    if num_nodes is not None:
        return int(num_nodes)
    for key in ("target", "position", "valid_mask", "predict_mask", "category"):
        value = get_store_value(store, key)
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.size(0))
    raise RuntimeError("无法从 processed 场景中推断 agent 数量。")


def get_embedded_scenario_id(data: Any) -> Optional[str]:
    keys = ("scenario_id", "scenario_ids")
    for key in keys:
        try:
            if isinstance(data, Mapping) and key in data:
                return normalize_scenario_id(data[key])
            value = data[key]
            return normalize_scenario_id(value)
        except Exception:
            continue
    return None


def set_agent_field(data: Any, key: str, value: Any) -> None:
    store = get_agent_store(data)
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().clone()
    store[key] = value


def collect_source_files(source_dir: Path) -> Dict[str, Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"processed 目录不存在：{source_dir}")
    files = sorted(
        path for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise RuntimeError(f"processed 目录中没有找到场景文件：{source_dir}")
    index: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}
    for path in files:
        sid = path.stem
        if sid in index:
            duplicates.setdefault(sid, [index[sid]]).append(path)
        else:
            index[sid] = path
    if duplicates:
        examples = {key: [str(p) for p in value] for key, value in list(duplicates.items())[:5]}
        raise RuntimeError(f"processed 目录存在重复文件名 stem，无法唯一匹配 scenario_id：{examples}")
    return index


def iter_sidecar_records(sidecar_dir: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    shard_paths = sorted(sidecar_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"没有找到 sidecar shard：{sidecar_dir}/shard_*.pt")
    for shard_path in shard_paths:
        try:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(shard_path, map_location="cpu")
        scenes = payload.get("scenes") if isinstance(payload, Mapping) else None
        if not isinstance(scenes, list):
            raise RuntimeError(f"sidecar shard 格式错误，缺少 scenes 列表：{shard_path}")
        for record in scenes:
            if not isinstance(record, Mapping):
                raise RuntimeError(f"sidecar scene record 不是字典：{shard_path}")
            yield shard_path, dict(record)


def validate_record(record: Mapping[str, Any], fields: Sequence[str], num_agents: int, sid: str) -> None:
    record_num_agents = int(record.get("num_agents", num_agents))
    if record_num_agents != num_agents:
        raise RuntimeError(
            f"agent 数量不一致：scenario_id={sid}, "
            f"processed={num_agents}, sidecar={record_num_agents}"
        )
    missing = [field for field in fields if field not in record]
    if missing:
        raise KeyError(f"sidecar 缺少训练字段：scenario_id={sid}, missing={missing}")
    for field in fields:
        value = record[field]
        if torch.is_tensor(value) and value.ndim >= 1 and int(value.size(0)) != num_agents:
            raise RuntimeError(
                f"字段首维与 agent 数量不一致：scenario_id={sid}, "
                f"field={field}, shape={tuple(value.shape)}, num_agents={num_agents}"
            )


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"输出目录已存在且非空：{path}；如需覆盖请加 --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def merge_split(
    split: str,
    source_dir: Path,
    sidecar_dir: Path,
    output_dir: Path,
    fields: Sequence[str],
    overwrite: bool,
    allow_partial: bool,
) -> Dict[str, Any]:
    prepare_output_dir(output_dir, overwrite)
    source_index = collect_source_files(source_dir)
    processed_sids: Set[str] = set()
    missing_sources: List[str] = []
    serializer_counts = {"pickle": 0, "torch": 0}

    records = iter_sidecar_records(sidecar_dir)
    progress = tqdm(records, desc=f"Merge {split} sidecar -> processed", unit="scene")
    for shard_path, record in progress:
        sid = normalize_scenario_id(record.get("scenario_id"))
        if not sid or sid == "None":
            raise RuntimeError(f"sidecar record 缺少 scenario_id：{shard_path}")
        if sid in processed_sids:
            raise RuntimeError(f"sidecar 中 scenario_id 重复：{sid}")
        source_path = source_index.get(sid)
        if source_path is None:
            missing_sources.append(sid)
            if allow_partial:
                continue
            raise FileNotFoundError(
                f"找不到 scenario_id={sid} 对应的 processed 文件。"
                f"预期文件 stem 与 scenario_id 相同，目录={source_dir}"
            )

        data, serializer = load_object(source_path)
        embedded_sid = get_embedded_scenario_id(data)
        if embedded_sid is not None and embedded_sid != sid:
            raise RuntimeError(
                f"processed 内部 scenario_id 不匹配：文件={source_path}, "
                f"sidecar={sid}, processed={embedded_sid}"
            )
        num_agents = get_num_agents(data)
        validate_record(record, fields, num_agents, sid)
        for field in fields:
            set_agent_field(data, field, record[field])

        relative_path = source_path.relative_to(source_dir)
        output_path = output_dir / relative_path
        atomic_save_object(data, output_path, serializer)
        serializer_counts[serializer] += 1
        processed_sids.add(sid)

    source_sids = set(source_index)
    unmerged_sources = sorted(source_sids - processed_sids)
    if unmerged_sources and not allow_partial:
        raise RuntimeError(
            f"{split} 有 {len(unmerged_sources)} 个 processed 场景没有对应 sidecar。"
            f"示例：{unmerged_sources[:10]}"
        )

    success_path = output_dir / "_SUCCESS"
    success_path.touch()
    return {
        "split": split,
        "source_processed_dir": str(source_dir.resolve()),
        "sidecar_dir": str(sidecar_dir.resolve()),
        "output_processed_dir": str(output_dir.resolve()),
        "num_source_files": len(source_index),
        "num_merged_scenes": len(processed_sids),
        "num_missing_source_for_sidecar": len(missing_sources),
        "num_source_without_sidecar": len(unmerged_sources),
        "serializer_counts": serializer_counts,
        "embedded_fields": list(fields),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 prototype sidecar assignment 写入新的 QCNet processed 数据集。"
    )
    parser.add_argument("--assignment_root", required=True,
                        help="包含 train/shard_*.pt、val/shard_*.pt 的 sidecar 根目录。")
    parser.add_argument("--out_root", required=True,
                        help="新 processed 数据集根目录；将生成 out_root/train 和 out_root/val。")
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    parser.add_argument("--train_processed_dir", default=None)
    parser.add_argument("--val_processed_dir", default=None)
    parser.add_argument("--include_diagnostics", action="store_true",
                        help="额外写入匹配距离、ADE/FDE 等诊断字段。")
    parser.add_argument("--allow_partial", action="store_true",
                        help="允许 processed 与 sidecar 场景集合不完全一致；默认严格要求完整对应。")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignment_root = Path(args.assignment_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    fields = list(TRAINING_FIELDS)
    if args.include_diagnostics:
        fields.extend(DIAGNOSTIC_FIELDS)

    split_source_dirs = {
        "train": Path(args.train_processed_dir).expanduser().resolve()
        if args.train_processed_dir else None,
        "val": Path(args.val_processed_dir).expanduser().resolve()
        if args.val_processed_dir else None,
    }
    manifests: Dict[str, Any] = {
        "format_version": 1,
        "mode": "prototype_fields_embedded_in_processed",
        "assignment_root": str(assignment_root),
        "out_root": str(out_root),
        "embedded_fields": fields,
        "splits": {},
    }

    out_root.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        source_dir = split_source_dirs[split]
        if source_dir is None:
            raise ValueError(f"处理 {split} 时必须提供 --{split}_processed_dir")
        result = merge_split(
            split=split,
            source_dir=source_dir,
            sidecar_dir=assignment_root / split,
            output_dir=out_root / split,
            fields=fields,
            overwrite=args.overwrite,
            allow_partial=args.allow_partial,
        )
        manifests["splits"][split] = result
        print(
            f"[{split}] merged={result['num_merged_scenes']:,}/"
            f"{result['num_source_files']:,}, output={result['output_processed_dir']}"
        )

    manifest_path = out_root / "embedded_prototype_manifest.json"
    manifest_path.write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n完成。原 processed 数据未修改。")
    print(f"新 processed 根目录：{out_root}")
    print(f"manifest：{manifest_path}")
    print("FM 训练时将 --train_processed_dir/--val_processed_dir 指向新目录，")
    print("并移除 --prototype_assignment_dir，避免训练阶段重复读取 sidecar shards。")


if __name__ == "__main__":
    main()