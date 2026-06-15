# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset


class LatentRegressionShardDataset(Dataset):
    def __init__(self, cache_dir: str, split: str) -> None:
        self.split_dir = Path(cache_dir) / split
        self.files = sorted(self.split_dir.glob("shard_*.pt"))
        if not self.files:
            raise FileNotFoundError(
                f"在 {self.split_dir} 中没有找到 shard_*.pt"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.files[index]
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")


def make_latent_regression_loader(
    cache_dir: str,
    split: str,
    shuffle: bool,
    num_workers: int = 4,
) -> DataLoader:
    dataset = LatentRegressionShardDataset(cache_dir, split)
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )