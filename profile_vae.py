import math
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from predictors import QCNetFM
from datamodules import ArgoverseV2DataModule

def main():
    parser = ArgumentParser(description="VAE 潜空间频域探伤与可视化工具 (动态意图自适应版)")
    
    # 1. 完美复刻 train_qcnet.py 的参数解析，确保 100% 命中本地缓存
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
    parser.add_argument('--ckpt_path', type=str, required=True)
    
    # 自动加载所有模型相关参数，包括 --vae_num_intents
    QCNetFM.add_model_specific_args(parser)
    args, _ = parser.parse_known_args()

    # 🌟 动态获取意图数量 (K)
    K = args.vae_num_intents
    print(f"\n" + "="*50)
    print(f"🔧 当前检测到的意图数量 (K): {K}")
    print("="*50 + "\n")

    print("⏳ 1. 正在初始化 Argoverse V2 Datamodule...")
    datamodule = ArgoverseV2DataModule(**vars(args))
    datamodule.setup(stage='validate')
    val_loader = datamodule.val_dataloader()

    print("⏳ 2. 正在提取真实轨迹数据...")
    data = next(iter(val_loader))
    target = data['agent']['target'][..., :2] / 10.0  
    
    print(f"⏳ 3. 正在精准提取 VAE 模型权重 ({args.ckpt_path.split('/')[-1]})...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化空模型 (参数严格对齐当前 K)
    model = QCNetFM(**vars(args))
    
    # 读取 Checkpoint 字典
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    
    # 绝对过滤：只保留带有 'latent_encoder' 或 'latent_decoder' 的键！
    vae_state_dict = {k: v for k, v in state_dict.items() if 'latent_' in k}
    
    # 强行注入纯净的 VAE 灵魂
    model.load_state_dict(vae_state_dict, strict=False)
    
    model.eval()
    model.to(device)
    target = target.to(device)

    print("\n" + "="*50)
    print(f"🚀 4. 开始潜空间频域探伤 (Jerk Analysis, K={K})...")
    print("="*50)
    
    with torch.no_grad():
        z_target = model.latent_encoder.encode(target) 
        
        # 寻找真正需要预测的主车 (category == 3)
        eval_mask = data['agent']['category'] == 3
        
        # 核心拦截：寻找未来 60 步完全没有被 TargetBuilder 外推过的完美智能体！
        future_mask = data['agent']['predict_mask'][:, 50:] 
        fully_valid_mask = future_mask.all(dim=1)  
        
        perfect_agents = torch.nonzero(eval_mask & fully_valid_mask).view(-1)
        
        if len(perfect_agents) > 0:
            displacements = torch.norm(target[perfect_agents, -1, :2] - target[perfect_agents, 0, :2], dim=-1)
            best_idx = torch.argmax(displacements).item()
            agent_idx = perfect_agents[best_idx].item()
            print(f"\n🎯 成功锁定 100% 纯人类驾驶、无任何外推的完美主车 (Index: {agent_idx})")
        else:
            print("\n⚠️ 整个 Batch 中没有 100% 完整的类别 3 主车，退而求其次寻找真实点最多的主车...")
            valid_counts = future_mask[eval_mask].sum(dim=1)
            best_idx = torch.argmax(valid_counts).item()
            valid_indices = torch.nonzero(eval_mask).view(-1)
            agent_idx = valid_indices[best_idx].item()
            print(f"👉 锁定了有效点最多的主车 (Index: {agent_idx})")
            
        gt_traj = target[agent_idx].cpu().numpy() * 10.0  # 放大回物理尺度 (米)
        full_recon = model.latent_decoder(z_target)[agent_idx].cpu().numpy() * 10.0
        
        jerk_scores = []
        isolated_trajs = []
        
        # 🌟 动态遍历 K 个 Token
        for i in range(K):
            z_iso = torch.zeros_like(z_target)
            z_iso[:, i, :] = z_target[:, i, :] 
            traj_iso = model.latent_decoder(z_iso)  
            
            # 计算 Jerk
            vel = traj_iso[:, 1:] - traj_iso[:, :-1]
            acc = vel[:, 1:] - vel[:, :-1]
            jerk = acc[:, 1:] - acc[:, :-1]
            
            jerk_magnitude = jerk.norm(dim=-1).mean().item()
            jerk_scores.append(jerk_magnitude)
            
            print(f"👉 Token {i} 的平均急动度 (Jerk): {jerk_magnitude:.6f}")
            isolated_trajs.append(traj_iso[agent_idx].cpu().numpy() * 10.0)

    # ================= 画图模块 (动态自适应版) =================
    print(f"\n🎨 5. 正在生成 {K} 意图动态隔离子图...")
    
    # 🌟 动态计算最佳的子图行列数 (例如: 3->1x3或2x2, 4->2x2, 5->2x3, 6->2x3)
    cols = int(math.ceil(math.sqrt(K)))
    rows = int(math.ceil(K / cols))
    
    # 动态调整画布大小
    fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    
    # 兼容 K=1 或多维数组的展平
    if K == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
        
    # 预设极其丰富的颜色池，供任意数量的 K 循环使用
    base_colors = ['#FF6B6B', '#FFB86C', '#4ECDC4', '#45B7D1', '#9B59B6', '#F1C40F', '#E74C3C', '#3498DB', '#2ECC71', '#34495E']
    
    for i in range(K):
        ax = axs[i]
        color = base_colors[i % len(base_colors)]
        
        # 1. 画出 GT 和 Full Recon 作为背景参考
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'k--', label='Ground Truth', linewidth=2)
        ax.plot(full_recon[:, 0], full_recon[:, 1], 'gray', label='Full Recon', alpha=0.4, linewidth=6)
        
        # 2. 画出当前的 Token
        traj = isolated_trajs[i]
        ax.plot(traj[:, 0], traj[:, 1], color=color, label=f'Token {i}', linewidth=3)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, s=100, edgecolors='white', zorder=5)

        # 3. 动态配置标题和信息
        ax.set_title(f"Isolated: Token {i}\n(Jerk: {jerk_scores[i]:.6f})", fontsize=16, pad=15)
        ax.set_xlabel("X (meters)", fontsize=12)
        ax.set_ylabel("Y (meters)", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.axis('equal') 
        ax.legend(loc='best', fontsize=11)
        
    # 🌟 核心清理：如果排布的网格(如 2x3=6)多于实际的 K (如 5)，把空白的网格隐藏掉
    for j in range(K, len(axs)):
        fig.delaxes(axs[j])
        
    plt.tight_layout()
    output_filename = f'trajectory_disentanglement_k{K}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ 自适应隔离子图版可视化已生成，请查看: {output_filename}")

if __name__ == '__main__':
    main()