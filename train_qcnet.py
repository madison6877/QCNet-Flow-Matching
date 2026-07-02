from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import torch
import torch.multiprocessing

from datamodules import ArgoverseV2DataModule
from predictors import QCNet, QCNetFM
from latent_regression_cache import make_latent_regression_loader

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    pl.seed_everything(2030, workers=True)
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--latent_cache_dir", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--val_batch_size", type=int, required=True)
    parser.add_argument("--test_batch_size", type=int, required=True)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--persistent_workers", type=bool, default=True)
    parser.add_argument("--train_raw_dir", type=str, default=None)
    parser.add_argument("--val_raw_dir", type=str, default=None)
    parser.add_argument("--test_raw_dir", type=str, default=None)
    parser.add_argument("--train_processed_dir", type=str, default=None)
    parser.add_argument("--val_processed_dir", type=str, default=None)
    parser.add_argument("--test_processed_dir", type=str, default=None)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, required=True)
    parser.add_argument("--max_epochs", type=int, default=64)
    parser.add_argument("--model_type", type=str, choices=["qcnet", "qcnet_fm"], default="qcnet")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--vae_processed_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=False)

    known_args, _ = parser.parse_known_args()
    model_cls = QCNetFM if known_args.model_type == "qcnet_fm" else QCNet
    model_cls.add_model_specific_args(parser)
    args = parser.parse_args()

    datamodule = {"argoverse_v2": ArgoverseV2DataModule}[args.dataset](**vars(args))
    model = model_cls(**vars(args))
    fit_ckpt_path = None
    if args.ckpt_path is not None:
        if args.resume:
            print(f"🔄 [Resume] 正在从 {args.ckpt_path} 完全恢复训练进度（包含优化器和 Epoch）...")
            fit_ckpt_path = args.ckpt_path
        else:
            print(f"🚀 [New Stage] 从 {args.ckpt_path} 加载兼容的预训练模型权重，新模块保持当前初始化...")
            checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            incompatible = model.load_state_dict(state_dict, strict=False)
            missing_keys = list(incompatible.missing_keys)
            unexpected_keys = list(incompatible.unexpected_keys)

            allowed_missing_exact = {"latent_fm_loss.weights"}
            allowed_missing_prefixes = ("latent_regressor.", "fm_decoder.to_vel.")
            invalid_missing = [
                key for key in missing_keys
                if key not in allowed_missing_exact and not key.startswith(allowed_missing_prefixes)
            ]

            def is_allowed_unexpected(key):
                if key.startswith("fm_decoder.to_vel."):
                    return True
                if key.startswith("fm_decoder.blocks."):
                    return any(name in key for name in (".adaLN_seg.", ".seg_attn.", ".norm4."))
                return False

            allowed_unexpected = [key for key in unexpected_keys if is_allowed_unexpected(key)]
            invalid_unexpected = [key for key in unexpected_keys if not is_allowed_unexpected(key)]

            if invalid_missing:
                raise RuntimeError(
                    f"除新回归头、新速度头和固定 buffer 外，还有当前模型参数未加载：{invalid_missing}"
                )
            if invalid_unexpected:
                raise RuntimeError(
                    f"checkpoint 中存在未被允许的旧参数：{invalid_unexpected}"
                )

            print(f"✅ 成功加载兼容权重，忽略旧结构参数 {len(allowed_unexpected)} 个")
            print("✅ 当前新速度头保持重新初始化")

            if hasattr(model, "latent_fm_loss") and hasattr(model.latent_fm_loss, "weights"):
                print("✅ latent_fm_loss.weights =", model.latent_fm_loss.weights.detach().cpu())

            print(f"✅ 成功加载兼容权重，忽略旧 segment-attention 参数 {len(allowed_unexpected)} 个")
            fit_ckpt_path = None
    if args.model_type == "qcnet_fm" and args.vae_only:
        monitor_metric = "val_vae_loss"
        monitor_mode = "min"
    elif args.model_type == "qcnet_fm" and args.latent_regression_only:
        monitor_metric = "val_latent_reg_loss"
        monitor_mode = "min"
    elif args.model_type == "qcnet_fm" and not args.scorer_only:
        monitor_metric = "val_fm_loss"
        monitor_mode = "min"
    else:
        monitor_metric = "val_minFDE"
        monitor_mode = "min"

    model_checkpoint = ModelCheckpoint(
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=5,
        save_last=True,
        save_weights_only=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        accumulate_grad_batches=1,
        precision="bf16-mixed",
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[model_checkpoint, lr_monitor],
        max_epochs=args.max_epochs,
    )

    if args.model_type == "qcnet_fm" and args.latent_regression_only and args.latent_cache_dir is not None:
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
    elif args.model_type == "qcnet_fm" and args.vae_only:
        if args.vae_processed_dir is None:
            raise ValueError("--vae_processed_dir must be set when vae_only=True")
        print(f"⚡ [Stage 0] Preprocessing VAE data to {args.vae_processed_dir}...")
        datamodule.setup(stage="fit")
        datamodule.prepare_vae_data()
        vae_train_loader = datamodule.vae_train_dataloader()
        val_loader = datamodule.val_dataloader()
        trainer.fit(
            model,
            train_dataloaders=vae_train_loader,
            val_dataloaders=val_loader,
            ckpt_path=fit_ckpt_path,
        )
    else:
        trainer.fit(model, datamodule, ckpt_path=fit_ckpt_path)
