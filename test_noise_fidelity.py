import os
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from tqdm import tqdm
import pytorch_lightning as pl
from torch_geometric.data import Batch

from datamodules import ArgoverseV2DataModule
from predictors import QCNetFM

def compute_mean_cos(tensor: torch.Tensor, agent_idx: int) -> float:
    """计算指定智能体的 N 个平行噪声特征两两之间的平均余弦相似度"""
    # tensor shape: [N_a, K, hidden_dim]
    feats = tensor[agent_idx]  # [K, hidden_dim]
    norm_feats = F.normalize(feats, p=2, dim=-1)
    cos_sim_matrix = torch.mm(norm_feats, norm_feats.t())
    
    # 提取上三角元素（排除对角线自身与自身的相似度 1.0）
    K = feats.size(0)
    triu_indices = torch.triu_indices(row=K, col=K, offset=1)
    mean_cos = cos_sim_matrix[triu_indices[0], triu_indices[1]].mean().item()
    return mean_cos

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
    
    # 自动注入 QCNetFM 特有的超参数
    QCNetFM.add_model_specific_args(parser)
    args = parser.parse_args()
    
    # 强制验证集 Batch Size 为 1，防止 K=100 图展开时显存爆炸
    args.val_batch_size = 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n🔄 1. 正在从 Checkpoint 加载 VAE 模型权重 (启用手工纯净过滤)...")
    model = QCNetFM(**vars(args))
    
    # 手工纯净加载逻辑，完美避开 size mismatch
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
            continue
        clean_state_dict[k] = v
        
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    model.to(device)
    
    print("📦 2. 正在初始化数据管道...")
    datamodule = ArgoverseV2DataModule(**vars(args))
    datamodule.setup(stage='validate')
    val_loader = datamodule.val_dataloader()
    
    print("🏃 3. 抓取场景并开始噪声保真度测试...")
    
    # 获取一个 Batch 的数据
    data = next(iter(val_loader)).to(device)
    if isinstance(data, Batch):
        data['agent']['av_index'] += data['agent']['ptr'][:-1]
        
    # 找到我们需要重点观察的目标车（预测车辆 category == 3）
    eval_mask = data['agent']['category'] == 3
    if not eval_mask.any():
        print("未找到需要预测的主车，选用智能体 0 进行测试。")
        target_agent_idx = 0
    else:
        target_agent_idx = eval_mask.nonzero(as_tuple=True)[0][0].item()

    with torch.no_grad():
        # ---------------------------------------------------------
        # 🌟 核心魔术：动态篡改 K 值，生成 100 倍密度的 Context Graph
        # ---------------------------------------------------------
        K_TEST = 100
        original_k = model.fm_decoder.num_intents
        model.fm_decoder.num_intents = K_TEST # 欺骗 Builder 生成 K=100 的边
        
        scene_enc = model.encoder(data)
        ctx = model.fm_decoder._build_graph_context(data, scene_enc)
        
        model.fm_decoder.num_intents = original_k # 恢复原状
        
        N_a = ctx['pos_m'].size(0)
        H = model.fm_decoder.hidden_dim
        
        # 1. 采样 100 个完全不同的高斯噪声
        x_t_test = torch.randn(N_a, K_TEST, model.fm_decoder.latent_dim, device=device)
        
        # 2. 准备时间条件 (设 t = 0.5)
        t = torch.tensor([0.5], device=device).expand(N_a)
        t_emb = model.fm_decoder.t_emb(continuous_inputs=t.unsqueeze(-1), categorical_embs=None)
        t_emb_s = t_emb.unsqueeze(1).expand(N_a, K_TEST, H)
        
        # 3. 提取极其强烈的场景特征 x_m
        x_t_hist_unflat = ctx['x_t_hist'].view(N_a, model.num_historical_steps, H)
        x_m = x_t_hist_unflat[:, -1, :]  # [N_a, H]
        
        # 4. 生成 Dummy 的频率位置编码 (设为 0，避免干扰噪声纯度测试)
        dummy_freq_pos_emb = torch.zeros(K_TEST, H, device=device)
        
        print("\n" + "="*70)
        print("🧪 噪声保真度纵深诊断测试 (Noise Fidelity Diagnostic Test)")
        print("="*70)
        print(f"📌 测试目标智能体 Index: {target_agent_idx}")
        print(f"🎲 并行注入高斯噪声数量: {K_TEST}")
        print("-" * 70)
        
        # ================== 追踪开始 ==================
        
        # 【入口处】
        x = model.fm_decoder.x_proj_in(x_t_test)
        cos_init = compute_mean_cos(x, target_agent_idx)
        print(f"▶ 初始态 (x_proj) | 均值 Cosine 相似度: {cos_init:.4f}")
        
        # 【逐层穿透 DiT Blocks】
        for i, block in enumerate(model.fm_decoder.blocks):
            x = block(
                x=x, t_emb_s=t_emb_s, freq_pos_emb=dummy_freq_pos_emb, K=K_TEST,
                x_t=ctx['x_t_hist'], r_t2a_exp=ctx['r_t2a_exp'], edge_index_t2a_exp=ctx['edge_index_t2a_exp'],
                x_pl=ctx['x_pl'], r_pl2a_exp=ctx['r_pl2a_exp'], edge_index_pl2a_exp=ctx['edge_index_pl2a_exp'],
                r_a2a_exp=ctx['r_a2a_exp'], edge_index_a2a_exp=ctx['edge_index_a2a_exp'],
                edge_threat_exp=ctx['edge_threat_exp'], edge_map_exp=ctx['edge_map_exp'],
                x_m=x_m
            )
            cos_block = compute_mean_cos(x, target_agent_idx)
            print(f"▶ Block {i+1} 穿透后       | 均值 Cosine 相似度: {cos_block:.4f}")
            
        print("="*70)
        
        # 诊断结论
        final_cos = compute_mean_cos(x, target_agent_idx)
        print("\n💡 架构诊断助手结论：")
        if final_cos > 0.9:
            print("🚨 绝症确诊：严重条件坍塌 (Condition Collapse)！")
            print("现象：你的初始噪声虽然差异很大，但在穿透 DiT Blocks 的过程中，被历史、地图等庞大的 Cross-Attention 特征暴风洗刷，同化成了同一个东西。")
            print("后果：Flow Matching 完全失去了多模态能力，退化成了普通的 MLP 预测器。")
            print("解药：请立即实施【门控残差零初始化 (Zero-Gating)】和【AdaLN 条件注入】手术！")
        elif final_cos > 0.5:
            print("⚠️ 亚健康：中度特征同化！")
            print("现象：噪声保留了一部分特征，但仍然受到了场景特征的严重干预，多模态生成能力被大幅削弱。")
        else:
            print("✅ 极其健康：多模态潜能拉满！")
            print("现象：不同的噪声样本在经历所有注意力机制后，依然保持着强烈的个性。你的网络能够完美利用 Flow Matching 生成极其多样的轨迹！")
        print("\n")
        print("\n" + "="*70)
        print("📊 门控权重 (Zero-Gating) 学习结果大揭秘")
        print("="*70)
        for i, block in enumerate(model.fm_decoder.blocks):
            # 计算门控权重在 96 个通道上的平均绝对强度
            g_t2a  = block.adaLN_t2a[-1].weight[-block.hidden_dim:].abs().mean().item()
            g_pl2a = block.adaLN_pl2a[-1].weight[-block.hidden_dim:].abs().mean().item()
            g_a2a  = block.adaLN_a2a[-1].weight[-block.hidden_dim:].abs().mean().item()
            print(f"Block {i+1}: 历史(t2a)={g_t2a:.4f} | 地图(pl2a)={g_pl2a:.4f} | 交互(a2a)={g_a2a:.4f}")