import os
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from tqdm import tqdm
import pytorch_lightning as pl
from torch_geometric.data import Batch

from datamodules import ArgoverseV2DataModule
from predictors import QCNetFM

def compute_min_ade(pred_traj: torch.Tensor, gt_traj: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """计算 minADE"""
    N_a, K, T_f, _ = pred_traj.shape
    gt_expanded = gt_traj.unsqueeze(1).expand(N_a, K, T_f, 2)
    dist = torch.norm(pred_traj - gt_expanded, dim=-1) # [N_a, K, T_f]
    mask_expanded = valid_mask.unsqueeze(1).expand(N_a, K, T_f)
    dist = dist * mask_expanded.float()
    ade = dist.sum(dim=-1) / (mask_expanded.float().sum(dim=-1) + 1e-5) # [N_a, K]
    min_ade, _ = ade.min(dim=-1) # [N_a]
    return min_ade.mean() 

if __name__ == '__main__':
    pl.seed_everything(2027, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--persistent_workers', type=bool, default=False)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='qcnet_fm')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--vae_processed_dir', type=str, default=None)
    
    
    # 自动注入 QCNetFM 参数
    QCNetFM.add_model_specific_args(parser)
    args, _ = parser.parse_known_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n🔄 1. 正在加载完整模型权重 (解除封印)...")
    model = QCNetFM(**vars(args))
    
    # 🌟 修复：删除 corrupted_keys 过滤，完美加载你 20 轮训练的每一丝权重！
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False) 
    
    model.eval()
    model.to(device)
    
    print("📦 2. 正在初始化验证集管道...")
    datamodule = ArgoverseV2DataModule(**vars(args))
    datamodule.setup(stage='validate')
    val_loader = datamodule.val_dataloader()
    
    all_latent_l2_errors = []
    
    # 根据真实 5维 Latent 的 Std 进行加噪
    dataset_std = torch.tensor([1.0592, 0.8184, 0.6869, 0.4002, 0.6022], device=device).view(1, 1, 5)
    alphas = [0.0, 0.5, 1.0, 2.0, 3.0]
    all_noisy_ades = {alpha: [] for alpha in alphas}

    print("\n" + "="*70)
    print("🩺 终极裁决：Latent Arrival Error & Decoder Robustness")
    print("="*70)

    with torch.no_grad():
        max_batches = 100
        
        for batch_idx, data in enumerate(tqdm(val_loader, total=max_batches, desc="评估中")):
            if batch_idx >= max_batches:
                break
                
            data = data.to(device)
            if isinstance(data, Batch):
                data['agent']['av_index'] += data['agent']['ptr'][:-1]
                
            eval_mask = data['agent']['category'] == 3
            if not eval_mask.any():
                continue
                
            target = data['agent']['target'][..., :model.output_dim] / 10.0
            predict_mask = data['agent']['predict_mask'][:, model.num_historical_steps:]
            
            target_eval = target[eval_mask]
            mask_eval = predict_mask[eval_mask]
            
            # =======================================================
            # 1. 获取 VAE 完美的均值 (mu)
            # =======================================================
            mu, _, _ = model.latent_encoder(target, return_latent=True, predict_mask=predict_mask)
            mu = mu.transpose(0, 1).contiguous()
            mu_eval = mu[eval_mask] # [N_eval, 1, 5]

            # =======================================================
            # [Exp A] Flow 潜空间误差评估 (精准的 5 维空间距离)
            # =======================================================
            scene_enc = model.encoder(data)
            # 这里调用真实的 10步 积分和 6 个 Mode 采样
            z_pred_full, _ = model.fm_decoder.sample(
                data, scene_enc, num_modes=model.num_modes, num_steps=model.fm_num_steps, latent_decoder=None
            ) # [N_a, 6, 1, 5]
            
            z_pred_eval = z_pred_full[eval_mask] # [N_eval, 6, 1, 5]

            mu_squeezed = mu_eval.squeeze(1)       # [N_eval, 5]
            z_squeezed = z_pred_eval.squeeze(2)    # [N_eval, 6, 5]
            
            dist = torch.norm(z_squeezed - mu_squeezed.unsqueeze(1), dim=-1) # [N_eval, 6]
            min_dist, _ = dist.min(dim=1) # [N_eval]
            all_latent_l2_errors.extend(min_dist.cpu().tolist())

            # =======================================================
            # [Exp B'] Decoder 真实分布尺度偏移容忍度测试
            # =======================================================
            num_scales = len(alphas)
            all_mu_noisy = []
            
            for alpha in alphas:
                if alpha > 0:
                    noise = torch.randn_like(mu_eval) * dataset_std
                    mu_noisy = mu_eval + alpha * noise
                else:
                    mu_noisy = mu_eval
                all_mu_noisy.append(mu_noisy)
                
            batched_mu_noisy = torch.cat(all_mu_noisy, dim=0) # [num_scales * N_eval, 1, 5]
            batched_noisy_traj = model.latent_decoder(batched_mu_noisy) # [num_scales * N_eval, T_f, 2]
            
            chunked_traj = torch.chunk(batched_noisy_traj, chunks=num_scales, dim=0)
            
            for idx, alpha in enumerate(alphas):
                traj_for_alpha = chunked_traj[idx] # [N_eval, T_f, 2]
                ade_val = compute_min_ade(traj_for_alpha.unsqueeze(1) * 10.0, target_eval * 10.0, mask_eval)
                all_noisy_ades[alpha].append(ade_val.item())


    # =======================================================
    # 📊 输出法庭裁决报告
    # =======================================================
    print("\n\n" + "█"*70)
    print("📈 论文级诊断报告 (Paper-Grade Diagnostic Report)")
    print("█"*70)

    avg_latent_error = sum(all_latent_l2_errors) / len(all_latent_l2_errors)
    print(f"\n[Exp A] Flow 5维潜空间到达误差 (Latent Arrival Error):")
    print(f"👉 满血权重 L2 Distance: {avg_latent_error:.4f}")

    print(f"\n[Exp B'] Decoder 基于真实分布的鲁棒性 (Decoder Robustness under Real-Std):")
    baseline_ade = sum(all_noisy_ades[0.0])/len(all_noisy_ades[0.0])
    
    for alpha in alphas:
        avg_ade = sum(all_noisy_ades[alpha]) / len(all_noisy_ades[alpha])
        degradation = avg_ade - baseline_ade
        print(f"  α = {alpha:.1f} | minADE: {avg_ade:.3f}m | 退化: +{degradation:.3f}m")
    print("█"*70 + "\n")