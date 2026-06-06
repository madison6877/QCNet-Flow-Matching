# VAE Target Dataset: lightweight dataset that loads pre-computed trajectory targets
# and predict masks from micro .pt files (~24KB per scene vs ~several MB .pkl files).
# This bypasses the heavy HeteroData loading + TargetBuilder CPU bottleneck during VAE training.

import os
import pickle
from glob import glob
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import wrap_angle


def compute_vae_target(position: torch.Tensor,
                       heading: torch.Tensor,
                       velocity: torch.Tensor,
                       valid_mask: torch.Tensor,
                       predict_mask: torch.Tensor,
                       num_historical_steps: int = 50,
                       output_dim: int = 2) -> Dict[str, torch.Tensor]:
    """
    从原始 Agent 特征中计算 VAE 训练所需的 target（相对坐标）和 predict_mask。

    复用 transforms/target_builder.py 的核心逻辑（相对坐标变换 + 断层插值/外推），
    但作为纯函数，不依赖 HeteroData。

    Args:
        position:   [N_a, T_total, 3]  全局坐标
        heading:    [N_a, T_total]      航向角
        velocity:   [N_a, T_total, 2/3] 全局速度
        valid_mask: [N_a, T_total]      有效性掩码
        predict_mask:[N_a, T_total]     预测掩码
        num_historical_steps: 历史时间步数（默认 50）
        output_dim:           输出维度（2 或 3，默认 2）

    Returns:
        dict with keys 'target' [N_a, T_future, D] and 'predict_mask' [N_a, T_future]
    """
    dt = 0.1
    T_total = position.size(1)
    num_future_steps = T_total - num_historical_steps
    current_step_idx = num_historical_steps - 1

    # 1. 提取原点和航向角
    origin = position[:, current_step_idx]
    theta = heading[:, current_step_idx]

    # 2. 构建旋转矩阵
    cos, sin = theta.cos(), theta.sin()
    N = position.size(0)
    rot_mat = theta.new_zeros(N, 2, 2)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = -sin
    rot_mat[:, 1, 0] = sin
    rot_mat[:, 1, 1] = cos

    # 3. 计算相对坐标 target
    target = origin.new_zeros(N, num_future_steps, output_dim)
    target[..., :2] = torch.bmm(
        position[:, num_historical_steps:, :2] - origin[:, :2].unsqueeze(1),
        rot_mat,
    )
    if output_dim >= 3 and position.size(2) >= 3:
        target[..., 2] = (
            position[:, num_historical_steps:, 2] - origin[:, 2].unsqueeze(-1)
        )

    future_mask = predict_mask[:, num_historical_steps:]
    target_pos = target[..., :2]

    # 4. 断层填充：插值（中途断层）+ 等速外推（永久消失）
    vel_49_global = velocity[:, current_step_idx, :2]
    prev_vel_local = torch.bmm(vel_49_global.unsqueeze(1), rot_mat).squeeze(1)
    current_valid_mask = valid_mask[:, current_step_idx]

    for i in range(N):
        if not current_valid_mask[i] or future_mask[i].all():
            continue

        last_valid_pos = torch.zeros(2, device=origin.device, dtype=torch.float32)
        last_valid_t = -1

        t = 0
        while t < num_future_steps:
            if future_mask[i, t]:
                last_valid_pos = target_pos[i, t]
                last_valid_t = t
                t += 1
            else:
                next_valid_t = -1
                for search_t in range(t + 1, num_future_steps):
                    if future_mask[i, search_t]:
                        next_valid_t = search_t
                        break

                if next_valid_t != -1:
                    # 场景 A：中途断层 -> 双端线性插值
                    next_valid_pos = target_pos[i, next_valid_t]
                    steps = next_valid_t - last_valid_t
                    step_vector = (next_valid_pos - last_valid_pos) / steps
                    for fill_t in range(t, next_valid_t):
                        target_pos[i, fill_t] = last_valid_pos + step_vector * (fill_t - last_valid_t)
                    t = next_valid_t
                else:
                    # 场景 B：永久消失 -> 等速直线外推
                    if last_valid_t >= 1:
                        v_extrap = (target_pos[i, last_valid_t] - target_pos[i, last_valid_t - 1]) / dt
                    elif last_valid_t == 0:
                        v_extrap = target_pos[i, 0] / dt
                    else:
                        v_extrap = prev_vel_local[i]

                    for fill_t in range(t, num_future_steps):
                        target_pos[i, fill_t] = last_valid_pos + v_extrap * (fill_t - last_valid_t) * dt
                    break

    target[..., :2] = target_pos

    # 5. 无效 agent 置零
    future_predict_mask = predict_mask[:, num_historical_steps:]
    future_predict_mask = future_predict_mask.clone()
    future_predict_mask[~current_valid_mask] = False

    return {
        'target': target,
        'predict_mask': future_predict_mask,
    }


def prepare_vae_data(processed_dir: str,
                     vae_dir: str,
                     target_builder=None) -> None:
    """
    一次性后处理：遍历所有已处理的 .pkl 文件，使用 TargetBuilder 生成 target，
    存为微型 .pt 文件（每场景约 24KB，vs 原始 .pkl 的几 MB）。

    关键：与 validation 使用完全相同的 TargetBuilder，确保 train/val 的 target 一致。

    调用时机：VAE 训练开始前执行一次。如果 vae_dir 中已有足够多的 .pt 文件则跳过。

    Args:
        processed_dir: 已处理数据目录（包含 *.pkl 文件）
        vae_dir:       VAE 轻量数据输出目录
        target_builder: TargetBuilder 实例，用于生成与 validation 一致的 target
    """
    if target_builder is None:
        raise ValueError(
            "target_builder must be a TargetBuilder instance to ensure "
            "train/val target consistency."
        )
    from torch_geometric.data import HeteroData

    os.makedirs(vae_dir, exist_ok=True)

    pkl_files = sorted(glob(os.path.join(processed_dir, '*.pkl')))
    if len(pkl_files) == 0:
        raise FileNotFoundError(f"No .pkl files found in {processed_dir}")

    # 检查是否已有足够的 .pt 文件（允许跳过已完成的后处理）
    existing_pt = set(os.path.splitext(os.path.basename(f))[0]
                       for f in glob(os.path.join(vae_dir, '*.pt')))
    pkl_ids = [os.path.splitext(os.path.basename(f))[0] for f in pkl_files]
    missing = [f for f, fid in zip(pkl_files, pkl_ids) if fid not in existing_pt]

    if len(missing) == 0:
        print(f"[VAE Prep] All {len(pkl_files)} scenes already processed in {vae_dir}, skipping.")
        return

    num_historical_steps = target_builder.num_historical_steps

    print(f"[VAE Prep] Processing {len(missing)}/{len(pkl_files)} scenes into {vae_dir} ...")
    for pkl_path in tqdm(missing, desc='VAE Prep'):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # 反序列化 HeteroData 并用 TargetBuilder 生成与 validation 一致的 target
        hetero_data = HeteroData()
        for key in data:
            hetero_data[key] = data[key]
        hetero_data = target_builder(hetero_data)

        target = hetero_data['agent']['target'][..., :2]
        predict_mask = hetero_data['agent']['predict_mask'][:, num_historical_steps:]

        # 🌟 与 validation_step 保持完全一致：过滤掉 current_step 无效的 agent
        current_valid_mask = hetero_data['agent']['valid_mask'][:, num_historical_steps - 1]
        predict_mask = predict_mask.clone()
        predict_mask[~current_valid_mask] = False

        # 应用 /10.0 归一化（与 training_step 中一致）
        target = target / 10.0

        result = {'target': target, 'predict_mask': predict_mask}

        scene_id = os.path.splitext(os.path.basename(pkl_path))[0]
        torch.save(result, os.path.join(vae_dir, f'{scene_id}.pt'))

    print(f"[VAE Prep] Done. {len(missing)} scenes saved.")


class VAETargetDataset(Dataset):
    """
    轻量 VAE 训练数据集：每个 __getitem__ 只加载一个微型 .pt 文件（~24KB），
    包含 target [N_agents, T_future, D] 和 predict_mask [N_agents, T_future]。

    相比原始 ArgoverseV2Dataset（加载几 MB 的 .pkl + HeteroData 转换 + TargetBuilder），
    IO 开销降约 100 倍，CPU 瓶颈消除。
    """

    def __init__(self, vae_dir: str):
        self.file_list = sorted(glob(os.path.join(vae_dir, '*.pt')))
        if len(self.file_list) == 0:
            raise FileNotFoundError(
                f"No .pt files found in {vae_dir}. "
                f"Please run prepare_vae_data() first."
            )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = torch.load(self.file_list[idx], weights_only=True)
        return {
            'target': data['target'],
            'predict_mask': data['predict_mask'],
        }


def vae_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    VAE 数据集专用 collate：沿 agent 维度拼接 batch_size 个独立场景。
    
    Input:  list of {'target': [N_i, T, D], 'predict_mask': [N_i, T]}
    Output: dict  {'target': [sum(N_i), T, D], 'predict_mask': [sum(N_i), T]}
    """
    targets = torch.cat([item['target'] for item in batch], dim=0)
    masks = torch.cat([item['predict_mask'] for item in batch], dim=0)
    return {'target': targets, 'predict_mask': masks}