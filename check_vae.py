import torch
from argparse import ArgumentParser
from datamodules import ArgoverseV2DataModule
from predictors import QCNetFM
from tqdm import tqdm
from typing import Optional

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=1)  # 根据你的指令保持为 1
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
    
    # 🌟 自动注入 QCNetFM 特有的超参数（会在这里自动安全定义 --dataset 参数）
    QCNetFM.add_model_specific_args(parser)
    args = parser.parse_args()
    
    print("\n🔄 1. 正在从 Checkpoint 加载 VAE 模型权重 (启用手工纯净过滤)...")
    model = QCNetFM(**vars(args))
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
            continue  # 优雅过滤老噪声
        clean_state_dict[k] = v
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("📦 2. 正在初始化数据管道...")
    datamodule = ArgoverseV2DataModule(**vars(args))
    datamodule.setup(stage='fit')
    
    loader = datamodule.val_dataloader()
    all_z = []
    
    print("🏃 3. 开始遍历验证集并提取有效智能体的 Latent 向量...")
    with torch.no_grad():
        for data in tqdm(loader, desc="Encoding Track"):
            data = data.to(device)
            
            target = data['agent']['target'][..., :model.output_dim]
            target = target / 10.0
            predict_mask = data['agent']['predict_mask'][:, model.num_historical_steps:]
            
            valid_mask = predict_mask.any(dim=-1)
            if not valid_mask.any():
                continue
                
            target_valid = target[valid_mask]
            predict_mask_valid = predict_mask[valid_mask]
            
            z_target = model.latent_encoder.encode(target_valid, predict_mask=predict_mask_valid)
            z_target = z_target.view(-1, model.latent_dim)
            all_z.append(z_target.cpu())
            
    if len(all_z) == 0:
        print("❌ 错误：未能提取到任何有效智能体的 Latent 特征，请检查数据集或 predict_mask！")
        exit()
        
    all_z = torch.cat(all_z, dim=0)
    
    global_mean = all_z.mean().item()
    global_std = all_z.std().item()
    dim_means = all_z.mean(dim=0)
    dim_stds = all_z.std(dim=0)
    
    print("\n" + "="*65)
    print("📊       VAE 潜空间真实数据分布报告 (Latent Space Stats)      ")
    print("="*65)
    print(f"分析的有效车辆总数 (Total Agents Evaluated): {all_z.size(0)}")
    print(f"全局统计值 -> 均值 (Global Mean): {global_mean:.4f}  [标准高斯理想值: 0.0]")
    print(f"全局统计值 -> 标准差 (Global Std): {global_std:.4f}  [标准高斯理想值: 1.0]")
    print("-"*65)
    print("维度级明细 (Dimension-wise breakdown):")
    for i in range(model.latent_dim):
        print(f" 维度 {i:02d} | 均值 (Mean): {dim_means[i]:7.4f} | 标准差 (Std): {dim_stds[i]:7.4f}")
    print("="*65)
    
    print("\n💡 架构诊断助手提示：")
    if abs(global_mean) > 0.4 or global_std < 0.2:
        print("🚨 警告：观测到极其明显的潜空间坍塌或空间平移漂移！")
        print(f"诊断：由于你的 Beta (0.006) 较小，VAE 摆脱了标准高斯的约束，选择抱团挤在一个狭小的‘局部死结’里。")
        print("后果：Flow Matching 强行从标准球体 N(0,1) 拉直线跨越真空区去拟合这个死结，导致速度场剧烈扭曲崩溃，minADE 恶化到 1.15。")
        print("🎯 终极解药：请将本报告中输出的 `dim_means` 和 `dim_stds` 转化为常量张量，在训练 Flow 前进行【强制标准化】，并在最终解码前【反标准化】还原。")
    else:
        print("✅ 潜空间特征分布十分健康，基本符合标准高斯流形。")