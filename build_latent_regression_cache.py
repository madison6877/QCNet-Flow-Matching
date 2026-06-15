#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from torch_geometric.data import Batch

from datamodules import ArgoverseV2DataModule
from predictors import QCNetFM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--train_raw_dir", default=None)
    p.add_argument("--train_processed_dir", default=None)
    p.add_argument("--val_raw_dir", default=None)
    p.add_argument("--val_processed_dir", default=None)
    p.add_argument("--test_raw_dir", default=None)
    p.add_argument("--test_processed_dir", default=None)
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--val_batch_size", type=int, default=48)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--chunk_size", type=int, default=16384)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", choices=["32", "bf16", "fp16"], default="bf16")
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=50)
    return p.parse_args()


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def autocast_ctx(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


class ShardWriter:
    def __init__(self, out_dir: Path, split: str, chunk_size: int) -> None:
        self.out_dir = out_dir / split
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.buffer: Dict[str, list[torch.Tensor]] = {}
        self.buffer_size = 0
        self.shard_index = 0
        self.total_agents = 0

    def add(self, item: Dict[str, torch.Tensor]) -> None:
        if not item:
            return
        n = next(iter(item.values())).size(0)
        if n == 0:
            return
        for key, value in item.items():
            if value.size(0) != n:
                raise ValueError(f"{key} 第一维 {value.size(0)} != {n}")
            self.buffer.setdefault(key, []).append(value.cpu())
        self.buffer_size += n
        self.total_agents += n
        self._flush_full_chunks()

    def _merged(self) -> Dict[str, torch.Tensor]:
        return {k: torch.cat(v, dim=0) for k, v in self.buffer.items()}

    def _flush_full_chunks(self) -> None:
        if self.buffer_size < self.chunk_size:
            return
        merged = self._merged()
        first_key = next(iter(merged))
        while merged[first_key].size(0) >= self.chunk_size:
            shard = {k: v[: self.chunk_size].contiguous() for k, v in merged.items()}
            self._save(shard)
            merged = {k: v[self.chunk_size :] for k, v in merged.items()}
        self.buffer = {k: ([v] if v.size(0) else []) for k, v in merged.items()}
        self.buffer_size = merged[first_key].size(0)

    def _save(self, shard: Dict[str, torch.Tensor]) -> None:
        path = self.out_dir / f"shard_{self.shard_index:06d}.pt"
        torch.save(shard, path)
        self.shard_index += 1

    def close(self) -> Dict[str, int]:
        if self.buffer_size > 0:
            self._save(self._merged())
        self.buffer.clear()
        self.buffer_size = 0
        return {"num_shards": self.shard_index, "num_agents": self.total_agents}


@torch.inference_mode()
def build_split(
    model: QCNetFM,
    loader: Iterable,
    split: str,
    out_dir: Path,
    chunk_size: int,
    device: torch.device,
    precision: str,
    max_batches: int,
    log_interval: int,
) -> Dict[str, int]:
    writer = ShardWriter(out_dir, split, chunk_size)
    model.encoder.eval()
    model.latent_encoder.eval()

    for batch_idx, data in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        if isinstance(data, Batch):
            data["agent"]["av_index"] += data["agent"]["ptr"][:-1]

        data = data.to(device)
        target = data["agent"]["target"][..., : model.output_dim] / 10.0
        predict_mask = data["agent"]["predict_mask"][
            :, model.num_historical_steps:
        ].bool()
        valid = predict_mask.any(dim=-1)
        if not valid.any():
            continue

        with autocast_ctx(device, precision):
            scene_enc = model.encoder(data)
            x_m = scene_enc["x_a"][:, -1, :]
            z_raw = model.latent_encoder.encode(
                target,
                predict_mask=predict_mask,
            )

        z_target_std = (
            z_raw.float() - model.z_mean.float()
        ) / (model.z_std.float() + 1e-6)

        category = data["agent"]["category"].long()
        eval_mask = category == 3

        writer.add(
            {
                "x_m": x_m[valid].float().cpu(),
                "z_target_std": z_target_std[valid].float().cpu(),
                "target": target[valid].float().cpu(),
                "predict_mask": predict_mask[valid].bool().cpu(),
                "category": category[valid].cpu(),
                "eval_mask": eval_mask[valid].bool().cpu(),
            }
        )

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            print(
                f"[{split}] batch={batch_idx + 1}, "
                f"cached_agents={writer.total_agents:,}, "
                f"shards={writer.shard_index}"
            )

    return writer.close()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(ckpt_path)
    hparams = dict(checkpoint.get("hyper_parameters", {}))
    if not hparams:
        raise RuntimeError("checkpoint 中缺少 hyper_parameters")

    model = QCNetFM(**hparams)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)

    device = torch.device(args.device)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

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
    cfg.setdefault("test_batch_size", args.val_batch_size)
    cfg["num_workers"] = args.num_workers
    cfg["shuffle"] = False
    cfg["pin_memory"] = True
    cfg["persistent_workers"] = args.num_workers > 0

    if not cfg.get("root"):
        raise ValueError("请通过 --root 指定数据根目录")

    dm = ArgoverseV2DataModule(**cfg)
    dm.setup(stage="fit")

    train_stats = build_split(
        model=model,
        loader=dm.train_dataloader(),
        split="train",
        out_dir=cache_dir,
        chunk_size=args.chunk_size,
        device=device,
        precision=args.precision,
        max_batches=args.max_train_batches,
        log_interval=args.log_interval,
    )
    val_stats = build_split(
        model=model,
        loader=dm.val_dataloader(),
        split="val",
        out_dir=cache_dir,
        chunk_size=args.chunk_size,
        device=device,
        precision=args.precision,
        max_batches=args.max_val_batches,
        log_interval=args.log_interval,
    )

    manifest = {
        "checkpoint": str(ckpt_path.resolve()),
        "hidden_dim": int(model.hidden_dim),
        "latent_dim": int(model.latent_dim),
        "num_intents": int(model.vae_num_intents),
        "num_future_steps": int(model.num_future_steps),
        "output_dim": int(model.output_dim),
        "trajectory_scale": 10.0,
        "z_mean": model.z_mean.detach().cpu().view(-1).tolist(),
        "z_std": model.z_std.detach().cpu().view(-1).tolist(),
        "precision_used_for_encoder": args.precision,
        "chunk_size": args.chunk_size,
        "train": train_stats,
        "val": val_stats,
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n缓存完成")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()