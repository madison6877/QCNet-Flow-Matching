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
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    """VAE training loss: reconstruction MSE + KL divergence + orthogonality penalty.

    The orthogonality penalty encourages the 3 intent tokens to learn
    disentangled representations by penalizing pairwise cosine similarity.

    Total loss = recon_loss + beta * kl_loss + gamma * ortho_loss
    """

    def __init__(self,
                 beta: float = 1.0,
                 gamma: float = 3.0,
                 reduction: str = 'mean') -> None:
        super(VAELoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self,
                recon_x: torch.Tensor,
                mu: torch.Tensor,
                logvar: torch.Tensor,
                target_x: torch.Tensor,
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
      
        # 先算原始的 MSE 矩阵 (不求和)
        recon_loss_raw = F.mse_loss(recon_x, target_x, reduction='none')  # [N_a, T_f, D]

        if mask is not None:
            # 🌟 核心拦截：精准揪出有效的 Agent
            valid_agent_mask = mask.any(dim=-1) # [N_a] 
            # 真正的分母：有效 Agent 的数量
            num_valid_agents = valid_agent_mask.sum().clamp(min=1) 

            # ---- 1. Recon 重构损失计算 ----
            # 每个 agent 的所有有效坐标点 MSE 总和，然后对有效 agent 取平均
            per_elem_sum = (recon_loss_raw * mask.unsqueeze(-1)).sum(dim=(1, 2))
            recon_loss = per_elem_sum[valid_agent_mask].mean()

            # ---- 2. KL 散度损失计算 ----
            kl_per_agent = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(0, 2))
            # 🌟 过滤掉垃圾数据的 KL！
            kl_loss = kl_per_agent[valid_agent_mask].sum() / num_valid_agents

            # ---- 3. 正交约束损失计算 ----
            K = mu.size(0)                                                  
            if K > 1:
                mu_norm = F.normalize(mu, dim=-1)                               
                cos_sim = torch.einsum('inj,mnj->nim', mu_norm, mu_norm)        
                ortho_mask = ~torch.eye(K, dtype=torch.bool, device=mu.device)        
                ortho_per_agent = cos_sim.abs()[:, ortho_mask].mean(dim=-1)
                ortho_loss = ortho_per_agent[valid_agent_mask].sum() / num_valid_agents                      
            else:
                ortho_loss = torch.tensor(0.0, device=mu.device)
                
        else:
            # Fallback 逻辑保持原样
            recon_loss = recon_loss_raw.sum(dim=(1, 2)).mean()
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(0, 2)).mean()
            K = mu.size(0)
            if K > 1:
                mu_norm = F.normalize(mu, dim=-1)
                cos_sim = torch.einsum('inj,mnj->nim', mu_norm, mu_norm)
                ortho_mask = ~torch.eye(K, dtype=torch.bool, device=mu.device)
                ortho_loss = cos_sim.abs()[:, ortho_mask].mean()
            else:
                ortho_loss = torch.tensor(0.0, device=mu.device)

        # ---- Total loss ----
        total_loss = recon_loss + self.beta * kl_loss + self.gamma * ortho_loss

        return total_loss, {
            'loss_total': total_loss.detach(),
            'loss_recon': recon_loss.detach(),
            'loss_kl': kl_loss.detach(),
            'ortho_aux': ortho_loss.detach(),
        }