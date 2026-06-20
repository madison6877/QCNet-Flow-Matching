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
import torch.multiprocessing
from latent_regression_cache import make_latent_regression_loader
# 强制多进程使用文件系统共享，避免 /dev/shm 内存不足导致的文件句柄丢失
torch.multiprocessing.set_sharing_strategy('file_system')

# 添加这一行来启用 Tensor Cores 优化
# 'high' 是最推荐的设置：速度快，精度损失几乎可以忽略不计
torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    pl.seed_everything(2024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument("--latent_cache_dir", type=str, default=None)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=14)
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
    parser.add_argument("--vae_weights_path", type=str, default=None)
    # parse known args to determine model class before adding model-specific args
    known_args, _ = parser.parse_known_args()
    model_cls = QCNetFM if known_args.model_type == 'qcnet_fm' else QCNet
    model_cls.add_model_specific_args(parser)
    args = parser.parse_args()
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
    }[args.dataset](**vars(args))
    model = model_cls(**vars(args))
    if args.model_type == "qcnet_fm" and args.vae_weights_path is not None:
        print(f"[VAE] Loading geometry-aware VAE from: {args.vae_weights_path}")

        vae_payload = torch.load(args.vae_weights_path, map_location="cpu")

        required_keys = {"latent_encoder", "z_mean", "z_std"}
        missing_payload_keys = required_keys - set(vae_payload.keys())
        if missing_payload_keys:
            raise RuntimeError(f"VAE weights file is missing keys: {sorted(missing_payload_keys)}")

        # 直接加载嵌套的 latent_encoder state_dict，不能把整个 vae_payload 传给 model.load_state_dict。
        incompatible = model.latent_encoder.load_state_dict(vae_payload["latent_encoder"], strict=True)

        if incompatible.missing_keys:
            raise RuntimeError(f"Missing latent_encoder keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected latent_encoder keys: {incompatible.unexpected_keys}")

        loaded_mean = vae_payload["z_mean"].reshape_as(model.z_mean)
        loaded_std = vae_payload["z_std"].reshape_as(model.z_std)

        with torch.no_grad():
            model.z_mean.copy_(loaded_mean.to(device=model.z_mean.device, dtype=model.z_mean.dtype))
            model.z_std.copy_(loaded_std.to(device=model.z_std.device, dtype=model.z_std.dtype))

            # 当前 UnnormDecoderWrapper 保存了自己的 mean/std 引用。显式同步，防止 decoder 使用旧 VAE 的统计量。
            if hasattr(model.latent_decoder, "mean"):
                model.latent_decoder.mean.copy_(loaded_mean.to(device=model.latent_decoder.mean.device, dtype=model.latent_decoder.mean.dtype))
            if hasattr(model.latent_decoder, "std"):
                model.latent_decoder.std.copy_(loaded_std.to(device=model.latent_decoder.std.device, dtype=model.latent_decoder.std.dtype))

        print("[VAE] Geometry-aware VAE loaded successfully.")
        print("[VAE] z_mean:", model.z_mean.flatten().tolist())
        print("[VAE] z_std :", model.z_std.flatten().tolist())

        if hasattr(model.latent_decoder, "mean"):
            mean_error = (model.latent_decoder.mean.cpu() - model.z_mean.cpu()).abs().max()
            std_error = (model.latent_decoder.std.cpu() - model.z_std.cpu()).abs().max()

            print("[VAE] decoder/stat mean max error:", float(mean_error))
            print("[VAE] decoder/stat std max error:", float(std_error))

            if mean_error > 1e-7 or std_error > 1e-7:
                raise RuntimeError("Decoder latent statistics are not synchronized.")

        # LatentSpaceDecoder 与 LatentSpaceEncoder 必须共享同一个 VAE。
        if hasattr(model.latent_decoder, "decoder"):
            assert model.latent_decoder.decoder.vae is model.latent_encoder.vae, "latent encoder and decoder do not share the same VAE"
    fit_ckpt_path = None
    if args.ckpt_path is not None:
        if args.resume:
            # 【阶段内中断恢复】：带上 optimizer、epoch、lr scheduler 全部状态，继续训练
            print(f"🔄 [Resume] 正在从 {args.ckpt_path} 完全恢复训练进度 (包含优化器和 Epoch)...")
            fit_ckpt_path = args.ckpt_path
        else:
            # 【跨阶段全新启动】：只加载模型权重，剥离所有优化器状态，从 Epoch 0 重新开始
            print(f"🚀 [New Stage] 正在从 {args.ckpt_path} 仅加载预训练权重 (作为全新阶段的起点)...")
            checkpoint = torch.load(args.ckpt_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            corrupted_keys = [
                'fm_decoder.to_vel.residual',
                'fm_decoder.to_vel.time_proj',
                'fm_decoder.to_vel.norm',
                'fm_decoder.to_vel.shortcut'
            ]
            clean_state_dict = {}
            for k, v in state_dict.items():
                if any(ck in k for ck in corrupted_keys):
                    print(f"🚯 已主动从 Checkpoint 中抹除冲突的老输出头权重: {k}")
                    continue  # 优雅拦截老噪声，不让它进入加载流
                clean_state_dict[k] = v
                
            # 3. 用提纯后的绝对干净的字典喂给模型，开启 strict=False 护航
            model.load_state_dict(clean_state_dict, strict=False)
            fit_ckpt_path = None
            # model = model_cls.load_from_checkpoint(args.ckpt_path, strict=False, **vars(args))
            # fit_ckpt_path = None

    if args.model_type == 'qcnet_fm' and args.vae_only:
        monitor_metric = 'val_vae_loss'
        monitor_mode = 'min'
    elif (args.model_type == "qcnet_fm" and args.latent_regression_only):
        monitor_metric = "val_latent_reg_loss"
        monitor_mode = "min"
    elif args.model_type == 'qcnet_fm' and not args.scorer_only:
        monitor_metric = 'val_fm_loss'
        monitor_mode = 'min'
    else:
        monitor_metric = 'val_minFDE'
        monitor_mode = 'min'
    model_checkpoint = ModelCheckpoint(monitor=monitor_metric, mode=monitor_mode, save_top_k=5, save_last=True, save_weights_only=False)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accumulate_grad_batches=2, precision='bf16-mixed',
                         accelerator=args.accelerator, devices=args.devices,
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs)

    
    if (args.model_type == "qcnet_fm" and args.latent_regression_only and args.latent_cache_dir is not None):
        train_loader = make_latent_regression_loader(
            cache_dir=args.latent_cache_dir,
            split="train",
            shuffle=True,
            num_workers=min(args.num_workers, 4),
        )

        val_loader = make_latent_regression_loader(
            cache_dir=args.latent_cache_dir,
            split="val",
            shuffle=False,
            num_workers=min(args.num_workers, 4),
        )

        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=fit_ckpt_path,
        )
    elif args.model_type == 'qcnet_fm' and args.vae_only:
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
