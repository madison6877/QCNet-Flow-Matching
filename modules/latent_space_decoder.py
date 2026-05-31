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
import torch.nn as nn

from layers.VAE import VAE


class LatentSpaceDecoder(nn.Module):
    """封装 VAE Decoder 为独立模块，冻结参数。

    Input:  z [N_a, 3, H]   (batch-first, 流匹配产物)
    Output: traj [N_a, T_f, D]  (物理轨迹坐标)
    """

    def __init__(self, vae: VAE) -> None:
        super(LatentSpaceDecoder, self).__init__()
        self.vae = vae

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # VAE.decode expects [3, N_a, H] -> transpose
        z_seq = z.transpose(0, 1)  # [3, N_a, H]
        recon_x = self.vae.decode(z_seq)  # [N_a, T_f, 2]
        return recon_x