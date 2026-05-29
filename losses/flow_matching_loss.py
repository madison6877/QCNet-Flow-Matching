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