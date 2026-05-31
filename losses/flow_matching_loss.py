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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatchingLoss(nn.Module):

    def __init__(self,
                 reduction: str = 'mean') -> None:
        super(FlowMatchingLoss, self).__init__()
        self.reduction = reduction

    @staticmethod
    def sample_noise_and_time(target: torch.Tensor,
                               device: torch.device,
                               agent_batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        N_a, T_f, D = target.shape
        x_0 = torch.randn(N_a, T_f, D, device=device)
        if agent_batch is None:
            # Single scene: all agents share the same t
            t = torch.rand(1, device=device).expand(N_a)
        else:
            # Batched scenes: one t per scene, broadcast via agent_batch indexing
            B = int(agent_batch.max().item()) + 1
            t_scene = torch.rand(B, device=device)
            t = t_scene[agent_batch]
        return x_0, t

    @staticmethod
    def sample_noise_and_time_latent(num_intents: int,
                                     N_a: int,
                                     hidden_dim: int,
                                     device: torch.device,
                                     agent_batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """潜空间流匹配: 3 个独立噪声 + 共享扩散步长 t。

        Args:
            num_intents: K=3 频域 token 数量
            N_a:         智能体数量
            hidden_dim:  潜空间维度

        Returns:
            x_0: [N_a, num_intents, hidden_dim]  初始噪声
            t:   [N_a]                            共享扩散步长 (per-agent)
        """
        x_0 = torch.randn(N_a, num_intents, hidden_dim, device=device)
        if agent_batch is None:
            t = torch.rand(N_a, device=device)
        else:
            B = int(agent_batch.max().item()) + 1
            t_scene = torch.rand(B, device=device)
            t = t_scene[agent_batch]
        return x_0, t

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                x_0: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        # target velocity field: u_t = x_1 - x_0
        u_t = target - x_0

        # flatten all trajectories within each scene into a single vector
        loss = (pred - u_t).pow(2).reshape(pred.size(0), -1).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))


class LatentFlowMatchingLoss(nn.Module):
    """潜空间多频段流匹配损失。

    计算 3 个频段速度预测与目标速度的加权 MSE。
    权重可通过 w_low / w_mid / w_high 调节各频段的重要性。

    Input:
        pred:     [N_a, 3, H]  速度场预测 (DiT 输出)
        z_target: [N_a, 3, H]  目标潜向量 (VAE Encoder 产物)
        x_0:      [N_a, 3, H]  初始噪声

    Returns:
        loss: scalar  加权平方误差总和
    """

    def __init__(self, w_low: float = 1.0, w_mid: float = 1.0, w_high: float = 1.0) -> None:
        super(LatentFlowMatchingLoss, self).__init__()
        self.register_buffer('weights', torch.tensor([w_low, w_mid, w_high]))

    def forward(self,
                pred: torch.Tensor,
                z_target: torch.Tensor,
                x_0: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # u_t = z_true - x_0  
        u_t = z_target - x_0                           # [N_a, 3, H]
        se = (pred - u_t).pow(2).sum(dim=-1)           # [N_a, 3]  
        
        # 🌟 先在 Batch 维度上求均值，防止 Batch Size 波动导致梯度爆炸
        se_mean = se.mean(dim=0)                       # [3]
        
        # 🌟 分别乘以各自的权重
        loss_low = se_mean[0] * self.weights[0]
        loss_mid = se_mean[1] * self.weights[1]
        loss_high = se_mean[2] * self.weights[2]
        
        # 总 Loss
        total_loss = loss_low + loss_mid + loss_high
        
        # 记录分离的各项 (使用 .detach() 剥离梯度，防止内存泄漏)
        loss_dict = {
            'low': loss_low.detach(),
            'mid': loss_mid.detach(),
            'high': loss_high.detach()
        }
        
        # 返回总 loss 和 分项字典
        return total_loss, loss_dict
