# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule
from predictors import QCNet, QCNetFM

import torch

# 添加这一行来启用 Tensor Cores 优化
# 'high' 是最推荐的设置：速度快，精度损失几乎可以忽略不计
torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    pl.seed_everything(2024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--model_type', type=str, choices=['qcnet', 'qcnet_fm'], default='qcnet')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--vae_processed_dir', type=str, default=None)
    parser.add_argument('--resume', action='store_true', default=False)
    # parse known args to determine model class before adding model-specific args
    known_args, _ = parser.parse_known_args()
    model_cls = QCNetFM if known_args.model_type == 'qcnet_fm' else QCNet
    model_cls.add_model_specific_args(parser)
    args = parser.parse_args()
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
    }[args.dataset](**vars(args))
    model = model_cls(**vars(args))
    fit_ckpt_path = None
    if args.ckpt_path is not None:
        if args.resume:
            # 【阶段内中断恢复】：带上 optimizer、epoch、lr scheduler 全部状态，继续训练
            print(f"🔄 [Resume] 正在从 {args.ckpt_path} 完全恢复训练进度 (包含优化器和 Epoch)...")
            fit_ckpt_path = args.ckpt_path
        else:
            # 【跨阶段全新启动】：只加载模型权重，剥离所有优化器状态，从 Epoch 0 重新开始
            print(f"🚀 [New Stage] 正在从 {args.ckpt_path} 仅加载预训练权重 (作为全新阶段的起点)...")
            model = model_cls.load_from_checkpoint(args.ckpt_path, strict=False, **vars(args))
            fit_ckpt_path = None

    if args.model_type == 'qcnet_fm' and args.vae_only:
        monitor_metric = 'val_vae_loss'
        monitor_mode = 'min'
    elif args.model_type == 'qcnet_fm' and not args.scorer_only:
        monitor_metric = 'val_fm_loss'
        monitor_mode = 'min'
    else:
        monitor_metric = 'val_minFDE'
        monitor_mode = 'min'
    model_checkpoint = ModelCheckpoint(monitor=monitor_metric, mode=monitor_mode, save_top_k=5, save_last=True, save_weights_only=False)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accumulate_grad_batches=5, precision='bf16-mixed',
                         accelerator=args.accelerator, devices=args.devices,
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs)

    # VAE-only mode: use lightweight .pt DataLoader to bypass TargetBuilder CPU bottleneck
    if args.model_type == 'qcnet_fm' and args.vae_only:
        if args.vae_processed_dir is None:
            raise ValueError('--vae_processed_dir must be set when vae_only=True')
        print(f'⚡ [Stage 0] Preprocessing VAE data to {args.vae_processed_dir}...')
        datamodule.setup(stage='fit')
        datamodule.prepare_vae_data()
        vae_train_loader = datamodule.vae_train_dataloader()
        val_loader = datamodule.val_dataloader()
        trainer.fit(model, train_dataloaders=vae_train_loader, val_dataloaders=val_loader,
                    ckpt_path=fit_ckpt_path)
    else:
        trainer.fit(model, datamodule, ckpt_path=fit_ckpt_path)
