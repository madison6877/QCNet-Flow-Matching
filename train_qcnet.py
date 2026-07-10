from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import torch
import torch.multiprocessing

from datamodules import ArgoverseV2DataModule
from predictors import QCNet, QCNetFM

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("high")

def load_vae_weights(model, ckpt_path):
    print(f"🚀 从 {ckpt_path} 仅加载 VAE 权重和 latent 统计...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("state_dict", checkpoint)
    current_state = model.state_dict()
    allowed_prefixes = ("latent_encoder.", "latent_decoder.")
    allowed_exact = {"z_mean", "z_std"}
    filtered_state = {}
    missing_source_keys = []
    shape_mismatches = []
    ignored_keys = []

    for key, value in source_state.items():
        is_vae_key = key.startswith(allowed_prefixes) or key in allowed_exact
        if not is_vae_key:
            ignored_keys.append(key)
            continue
        if key not in current_state:
            missing_source_keys.append(key)
            continue
        if current_state[key].shape != value.shape:
            shape_mismatches.append(f"{key}: checkpoint={tuple(value.shape)}, current={tuple(current_state[key].shape)}")
            continue
        filtered_state[key] = value

    required_current_keys = [key for key in current_state if key.startswith(allowed_prefixes) or key in allowed_exact]
    unloaded_required_keys = [key for key in required_current_keys if key not in filtered_state]

    if shape_mismatches:
        raise RuntimeError("VAE checkpoint 与当前 VAE 结构存在尺寸不匹配：\n" + "\n".join(shape_mismatches))
    if missing_source_keys:
        raise RuntimeError("checkpoint 中的 VAE 参数在当前模型中不存在：\n" + "\n".join(missing_source_keys))
    if unloaded_required_keys:
        raise RuntimeError("当前 VAE 中存在未从 checkpoint 加载的必要参数：\n" + "\n".join(unloaded_required_keys))

    incompatible = model.load_state_dict(filtered_state, strict=False)
    loaded_numel = sum(value.numel() for value in filtered_state.values())

    print(f"✅ 成功加载 VAE tensor 数量：{len(filtered_state)}")
    print(f"✅ 成功加载 VAE 参数量：{loaded_numel:,}")
    print(f"✅ 忽略非 VAE tensor 数量：{len(ignored_keys)}")
    print("✅ 旧 fm_decoder.* 已全部忽略")
    print("✅ 当前新 fm_decoder 保持重新初始化")

    if hasattr(model, "z_mean"):
        print("✅ z_mean =", model.z_mean.detach().cpu())
    if hasattr(model, "z_std"):
        print("✅ z_std =", model.z_std.detach().cpu())

    return incompatible
if __name__ == "__main__":
    pl.seed_everything(2030, workers=True)
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--latent_cache_dir", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--val_batch_size", type=int, required=True)
    parser.add_argument("--test_batch_size", type=int, required=True)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=14)
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
    parser.add_argument("--prototype_assignment_dir", type=str, default=None)
    parser.add_argument("--prototype_assignment_strict", action="store_true", default=False)
    parser.add_argument("--assignment_cache_size", type=int, default=8)
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
            load_vae_weights(model=model,ckpt_path=args.ckpt_path)
            fit_ckpt_path = None
    if args.model_type == "qcnet_fm" and args.vae_only:
        monitor_metric = "val_vae_loss"
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
        accumulate_grad_batches=2,
        precision="bf16-mixed",
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[model_checkpoint, lr_monitor],
        max_epochs=args.max_epochs,
    )
  
    if args.model_type == "qcnet_fm" and args.vae_only:
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
