import os

from typing import Callable, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from datasets.vae_target_dataset import (VAETargetDataset,
                                         prepare_vae_data,
                                         vae_collate_fn)
from transforms import TargetBuilder


class ArgoverseV2DataModule(pl.LightningDataModule):

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_raw_dir: Optional[str] = None,
                 val_raw_dir: Optional[str] = None,
                 test_raw_dir: Optional[str] = None,
                 train_processed_dir: Optional[str] = None,
                 val_processed_dir: Optional[str] = None,
                 test_processed_dir: Optional[str] = None,
                 train_transform: Optional[Callable] = TargetBuilder(50, 60),
                 val_transform: Optional[Callable] = TargetBuilder(50, 60),
                 test_transform: Optional[Callable] = None,
                 vae_processed_dir: Optional[str] = None,
                 prototype_assignment_dir: Optional[str] = None,
                 prototype_assignment_strict: bool = False,
                 assignment_cache_size: int = 4,
                 **kwargs) -> None:
        super(ArgoverseV2DataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.vae_processed_dir = vae_processed_dir
        self.prototype_assignment_dir = prototype_assignment_dir
        self.prototype_assignment_strict = prototype_assignment_strict
        self.assignment_cache_size = assignment_cache_size

        print("[DM INIT] prototype_assignment_dir =", self.prototype_assignment_dir)
        print("[DM INIT] prototype_assignment_strict =", self.prototype_assignment_strict)
        print("[DM INIT] assignment_cache_size =", self.assignment_cache_size)

    def prepare_data(self) -> None:
        ArgoverseV2Dataset(self.root, 'train', self.train_raw_dir, self.train_processed_dir, self.train_transform, prototype_assignment_dir=self.prototype_assignment_dir, prototype_assignment_strict=self.prototype_assignment_strict, assignment_cache_size=self.assignment_cache_size)
        ArgoverseV2Dataset(self.root, 'val', self.val_raw_dir, self.val_processed_dir, self.val_transform, prototype_assignment_dir=self.prototype_assignment_dir, prototype_assignment_strict=self.prototype_assignment_strict, assignment_cache_size=self.assignment_cache_size)
        ArgoverseV2Dataset(self.root, 'test', self.test_raw_dir, self.test_processed_dir, self.test_transform)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ArgoverseV2Dataset(
            self.root,
            'train',
            self.train_raw_dir,
            self.train_processed_dir,
            self.train_transform,
            prototype_assignment_dir=self.prototype_assignment_dir,
            prototype_assignment_strict=self.prototype_assignment_strict,
            assignment_cache_size=self.assignment_cache_size,
        )

        self.val_dataset = ArgoverseV2Dataset(
            self.root,
            'val',
            self.val_raw_dir,
            self.val_processed_dir,
            self.val_transform,
            prototype_assignment_dir=self.prototype_assignment_dir,
            prototype_assignment_strict=self.prototype_assignment_strict,
            assignment_cache_size=self.assignment_cache_size,
        )

        self.test_dataset = ArgoverseV2Dataset(
            self.root,
            'test',
            self.test_raw_dir,
            self.test_processed_dir,
            self.test_transform,
        )
        print("[SETUP] train assignment dir =", self.train_dataset.prototype_assignment_dir)
        print("[SETUP] val assignment dir =", self.val_dataset.prototype_assignment_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers, prefetch_factor=2)

    def prepare_vae_data(self) -> None:
        """一次性后处理：从完整 .pkl 文件提取 VAE 所需的 target/mask 存为微型 .pt 文件。

        应在 VAE 训练开始前调用一次。如果 vae_processed_dir 中已有文件则跳过。
        """
        if self.vae_processed_dir is None:
            raise ValueError(
                "vae_processed_dir must be set to use VAE mode. "
                "Specify --vae_processed_dir in command line or configure it."
            )
        # 1. 安全获取 train_processed_dir
        processed_dir = getattr(self, 'train_processed_dir', None)

        # 2. 如果命令行没传，强行根据 root 补全路径
        if processed_dir is None:
            # 安全获取 root 路径
            root_dir = getattr(self, 'root', None)
            if root_dir is None:
                raise ValueError("无法推导路径: self.root 也是 None! 请在命令行强制传入 --train_processed_dir")
            processed_dir = os.path.join(root_dir, 'train', 'processed')

        print(f"🔍 [Debug] 即将从以下路径读取 .pkl 文件: {processed_dir}")
        prepare_vae_data(
            processed_dir=processed_dir,
            vae_dir=self.vae_processed_dir,
            target_builder=self.train_transform,
        )

    def vae_train_dataloader(self) -> TorchDataLoader:
        """VAE 训练专用 DataLoader: 加载轻量 .pt 文件而非完整 HeteroData。

        每个 .pt 文件约 24KB (vs .pkl 的几 MB), IO 开销降约 100 倍，
        且绕过 TargetBuilder CPU 瓶颈。
        """
        dataset = VAETargetDataset(self.vae_processed_dir)
        return TorchDataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            prefetch_factor=4,
            collate_fn=vae_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers, prefetch_factor=2)
