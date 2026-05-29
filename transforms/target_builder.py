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
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from utils import wrap_angle


import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from utils import wrap_angle


class TargetBuilder(BaseTransform):

    def __init__(self,
                 num_historical_steps: int,
                 num_future_steps: int) -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.dt = 0.1  # Argoverse 的时间步长

    def __call__(self, data: HeteroData) -> HeteroData:
        current_step_idx = self.num_historical_steps - 1

        # 1. 提取原点和航向角
        origin = data['agent']['position'][:, current_step_idx]
        theta = data['agent']['heading'][:, current_step_idx]
        
        # 2. 构建旋转矩阵
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        
        # 3. 初始化 Target Tensor (这是 target 出生的地方！)
        data['agent']['target'] = origin.new_zeros(data['agent']['num_nodes'], self.num_future_steps, 4)
        
        # 4. 正常计算相对坐标 target
        data['agent']['target'][..., :2] = torch.bmm(
            data['agent']['position'][:, self.num_historical_steps:, :2] - origin[:, :2].unsqueeze(1), 
            rot_mat
        )
        if data['agent']['position'].size(2) == 3:
            data['agent']['target'][..., 2] = (
                data['agent']['position'][:, self.num_historical_steps:, 2] - origin[:, 2].unsqueeze(-1)
            )
        data['agent']['target'][..., 3] = wrap_angle(
            data['agent']['heading'][:, self.num_historical_steps:] - theta.unsqueeze(-1)
        )

        future_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        target_pos = data['agent']['target'][..., :2]
        
        N = target_pos.size(0)
        T = self.num_future_steps
        
        # 获取 t=49 的局部速度，用作最坏情况下的保底外推速度
        vel_49_global = data['agent']['velocity'][:, current_step_idx, :2]
        prev_vel_local = torch.bmm(vel_49_global.unsqueeze(1), rot_mat).squeeze(1)

        current_valid_mask = data['agent']['valid_mask'][:, current_step_idx]

        for i in range(N):
            if not current_valid_mask[i] or future_mask[i].all():
                continue

            last_valid_pos = torch.zeros(2, device=origin.device, dtype=torch.float32)
            last_valid_t = -1
            
            t = 0
            while t < T:
                if future_mask[i, t]:
                    last_valid_pos = target_pos[i, t]
                    last_valid_t = t
                    t += 1
                else:
                    next_valid_t = -1
                    for search_t in range(t + 1, T):
                        if future_mask[i, search_t]:
                            next_valid_t = search_t
                            break
                    
                    if next_valid_t != -1:
                        # 🌟 场景 A：中途断层 -> 【双端线性插值】
                        next_valid_pos = target_pos[i, next_valid_t]
                        steps = next_valid_t - last_valid_t
                        step_vector = (next_valid_pos - last_valid_pos) / steps
                        
                        for fill_t in range(t, next_valid_t):
                            target_pos[i, fill_t] = last_valid_pos + step_vector * (fill_t - last_valid_t)
                        
                        t = next_valid_t
                    else:
                        # 🌟 场景 B：永久消失 -> 【等速直线外推】
                        if last_valid_t >= 1:
                            v_extrap = (target_pos[i, last_valid_t] - target_pos[i, last_valid_t - 1]) / self.dt
                        elif last_valid_t == 0:
                            v_extrap = target_pos[i, 0] / self.dt
                        else:
                            v_extrap = prev_vel_local[i]
                            
                        for fill_t in range(t, T):
                            target_pos[i, fill_t] = last_valid_pos + v_extrap * (fill_t - last_valid_t) * self.dt
                        
                        break
                        
        data['agent']['target'][..., :2] = target_pos

        data['agent']['predict_mask'][~current_valid_mask, self.num_historical_steps:] = False
        data['agent']['target'][~current_valid_mask, ...] = 0.0

        return data