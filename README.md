# QCNet Structured Latent Residual Flow Matching

本项目在 QCNet 场景编码器之上，将未来轨迹的生成过程放到 VAE 潜空间中完成。与直接从高斯噪声生成完整 latent 不同，本项目先建立离散的 prototype 库，再由场景条件 selector 选择合适的 prototype，最后使用残差中心头和 flow matching 速度场补全 prototype 无法表达的连续变化。

本文档以当前仓库代码为准，核心实现位于：

- `predictors/qcnet_fm.py`：完整 Lightning 模型、prototype selector、损失、训练/验证/测试逻辑。
- `modules/qcnet_fm_decoder.py`：场景 token 聚合、残差中心头、潜空间速度场、CFG 和 Heun 采样。
- `train_geometry_aware_vae.py`：几何正则 VAE 的分阶段训练与导出。
- `build_latent_prototypes_and_assign.py`：从训练集 VAE cache 建立层次化 prototype 库。
- `merge_assignment_into_processed.py`：为 train/val 计算 GT-to-prototype assignment，并生成 sidecar shards。
- `merge_prototype_sidecar_into_processed.py`：把 sidecar 中的 assignment 字段写入一份新的 processed 数据集。

## 1. 方法概述

整个方法可以概括为：

```text
GT future trajectory
        │
        ▼
geometry-aware VAE encoder
        │ posterior mean
        ▼
centered-raw latent z_c
        │
        ├── 离线：聚类并建立 prototype bank
        │
        └── 离线：为每个 GT 分配 primary/secondary prototype

scene history + map + agents ──► QCNet encoder ──► structured scene tokens
                                                        │
                         ┌──────────────────────────────┼──────────────────────┐
                         ▼                              ▼                      ▼
                 prototype selector            residual center head      velocity field
                         │                              │                      │
                         ▼                              ▼                      ▼
                  selected prototype p          residual center c       residual delta
                         └──────────────────────────────┴──────────────────────┘
                                                        │
                                                        ▼
                                             z_hat = p + c + delta
                                                        │
                                                        ▼
                                                   VAE decoder
                                                        │
                                                        ▼
                                                multimodal trajectories
```

### 1.1 潜空间定义

轨迹首先除以 `trajectory_scale`，默认值为 10：

```math
\tilde{y} = y / s,\qquad s=\texttt{trajectory\_scale}.
```

VAE encoder 使用后验均值而不是随机采样值作为聚类与 flow matching 的目标：

```math
z_{\mathrm{raw}} = \mu_\phi(\tilde{y}).
```

prototype 库和 flow matching 实际工作在 **centered-raw latent**：

```math
z_c = z_{\mathrm{raw}}-\mu_z.
```

这里仅减去 `z_mean`，不会除以 `z_std`。VAE 几何正则内部使用 standardized latent：

```math
z_{\mathrm{std}}=(z_{\mathrm{raw}}-\mu_z)/\sigma_z,
```

但 prototype 聚类、离线 assignment、残差建模和 FM 都使用 `z_c`。这两个空间不能混用。

### 1.2 Prototype 与层次化 assignment

`build_latent_prototypes_and_assign.py` 只读取训练集的 VAE cache，建库过程分为两层：

1. 从物理轨迹构造 descriptor，包括 DCT 形状、分段位移、分段速度、分段转角和全局运动特征。
2. 在 descriptor 空间进行 coarse KMeans，得到运动语义粗分组。
3. 在每个 coarse group 内的 centered-raw latent 空间分配 leaf prototype 数量。
4. 使用局部 KMeans、可选 split-merge refinement，并把连续聚类中心替换为真实训练样本 medoid。
5. 用 VAE decoder 解码 prototype，保存轨迹表示与重叠诊断。

因此，最终 assignment 规则不是在所有 prototype 中直接全局最近邻，而是：

```math
g_i = \operatorname{NearestCoarse}(d_i),
```

```math
j_i = \arg\min_{j:\,g_j=g_i}\lVert z_{c,i}-p_j\rVert_2.
```

其中 `d_i` 是轨迹 descriptor，`p_j` 是第 `j` 个 prototype。

对靠近聚类边界的样本，代码还保存同一 coarse group 内的 Top-2 prototype。边界判据为：

```math
m_i=\frac{d_{2,i}-d_{1,i}}{d_{1,i}+\epsilon},
```

当 `m_i < support_margin_threshold` 时，将 primary 和 secondary 都作为 selector 的正支持，并依据距离和 temperature 生成软权重。

### 1.3 Selector

selector 为每个 agent 和每个 prototype 输出一个 logit。prototype token 由以下信息构成：

- prototype 的连续 latent 几何；
- prototype 的可学习离散 ID embedding；
- history、map、neighbor-agent 三组场景 token。

每层 selector block 依次执行 prototype-to-history、prototype-to-map、prototype-to-agent cross-attention，并可执行 prototype 间 self-attention。

普通样本以 primary prototype 为硬标签；边界样本使用 Top-2 软标签。当前损失为：

```math
\mathcal{L}_{sel}
=\lambda_{ce}\mathcal{L}_{soft/hard\ CE}
+\lambda_{rank}\mathcal{L}_{hard\ negative\ rank}.
```

rank loss 从 selector 当前得分最高的负 prototype 中选 hard negatives。

### 1.4 残差中心与 Delta Flow Matching

给定 GT latent 和分配的 prototype：

```math
r=z_c-p.
```

由于 `r` 仍可能具有明显的条件均值和较大的组内方差，残差中心头根据场景和 prototype 预测确定性中心：

```math
c_\psi=C_\psi(S,p).
```

速度场不直接生成完整残差，而是学习：

```math
\delta=r-\operatorname{stopgrad}(c_\psi).
```

中心头通过独立的 MSE 训练：

```math
\mathcal{L}_{center}
=\lVert c_\psi-r\rVert_2^2.
```

在 FM 分支中对中心值执行 detach，因此 FM loss 不会通过 `delta` 分支反向更新中心头。这样中心头只负责学习确定性条件均值，速度场只负责剩余不确定性。

FM 源分布使用 prototype 对应训练残差的局部标准差：

```math
x_0\sim\mathcal{N}
\left(0,\left(\alpha\sigma_j\right)^2I\right),
\qquad \alpha=\texttt{residual\_source\_scale}.
```

线性概率路径和目标速度为：

```math
x_t=(1-t)x_0+t\delta,\qquad
u_t=\delta-x_0.
```

速度场训练目标为：

```math
\mathcal{L}_{FM}
=\lVert v_\theta(x_t,t,S,p)-u_t\rVert_2^2.
```

总训练损失为：

```math
\mathcal{L}
=\mathcal{L}_{FM}
+\lambda_{ADE}\mathcal{L}_{ADE}
+\lambda_{FDE}\mathcal{L}_{FDE}
+\lambda_{sel}^{outer}\mathcal{L}_{sel}
+\lambda_{center}\mathcal{L}_{center}.
```

其中 ADE/FDE 是把当前 FM 终点 latent 解码回轨迹后的辅助损失。

### 1.5 推理

推理不需要 GT assignment：

1. selector 根据场景对完整 prototype 库打分；
2. 取 Top-K prototype，可为每个 prototype 重复采样多个 residual；
3. 残差中心头预测每个候选 prototype 的 `c`；
4. 从 prototype 局部残差尺度采样 `x_0`；
5. 使用 Heun 方法对速度场积分；
6. 得到最终 latent：

```math
\hat z_c=p+c+\hat\delta;
```

7. 加回 `z_mean` 后送入 VAE decoder；
8. selector 的 Top-K softmax 分数作为 mode probability。

当前 CFG 只作用于速度场分支：它移除速度场的 scene 条件，但始终保留 prototype latent 和 prototype ID。selector 和 residual center 仍使用完整场景条件：

```math
v_{cfg}
=v_{\text{prototype-only}}
+w\left(v_{\text{scene+prototype}}-v_{\text{prototype-only}}\right).
```

对速度场而言，`w=0` 为 prototype-only velocity，`w=1` 为普通条件 velocity，`w>1` 放大场景速度修正。即使 `w=0`，最终结果仍包含场景条件 selector 和场景条件 residual center 的影响。

## 2. 环境与数据

`environment.yml` 基于 Linux、Python 3.8、PyTorch 2.0.1、CUDA 11.8、PyG 2.3.0 和 PyTorch Lightning 2.0.4。prototype 建库依赖 `scikit-learn`，环境文件中已经包含。

```bash
conda env create -f environment.yml
conda activate QCNet
```

Argoverse 2 数据建议保持以下结构：

```text
DATA_ROOT/
├── train/
│   ├── raw/
│   └── processed/
├── val/
│   ├── raw/
│   └── processed/
└── test/
    ├── raw/
    └── processed/
```

后续示例使用以下环境变量。路径需要替换为本机真实路径：

```bash
export DATA_ROOT=/path/to/av2
export ARTIFACT_ROOT=/path/to/structured_latent_artifacts
export VAE_CACHE=$ARTIFACT_ROOT/vae_cache
export VAE_RUN=$ARTIFACT_ROOT/geometry_vae
export BANK_DIR=$ARTIFACT_ROOT/prototype_bank
export ASSIGNMENT_ROOT=$ARTIFACT_ROOT/prototype_assignments
export MERGED_ROOT=$ARTIFACT_ROOT/processed_with_prototypes
export LATENT_DIM=5
```

所有阶段必须使用一致的：

- `hidden_dim`
- `latent_dim`
- `vae_num_intents`
- `num_future_steps`
- `trajectory_scale`
- VAE checkpoint 和对应 `z_mean/z_std`

当前 prototype 建库脚本明确要求 `vae_num_intents=1`。

## 3. 完整运行顺序

正确的数据依赖顺序为：

```text
1. train_geometry_aware_vae.py
2. build_latent_prototypes_and_assign.py
3. merge_assignment_into_processed.py
4. merge_prototype_sidecar_into_processed.py
5. train_qcnet.py
```

特别注意：虽然两个 merge 脚本的文件名容易造成误解，但代码上必须先运行 `merge_assignment_into_processed.py` 生成 `train/val/shard_*.pt`，再运行 `merge_prototype_sidecar_into_processed.py` 消费这些 shards 并写入新的 processed 数据。反向执行时，后者没有可读取的 sidecar。

### 3.1 训练 geometry-aware VAE

下面是一套 AV2 配置模板。半径、batch size 和训练轮数可按实验调整。

```bash
python train_geometry_aware_vae.py \
  --root "$DATA_ROOT" \
  --train_processed_dir "$DATA_ROOT/train/processed" \
  --val_processed_dir "$DATA_ROOT/val/processed" \
  --test_processed_dir "$DATA_ROOT/test/processed" \
  --vae_processed_dir "$VAE_CACHE" \
  --output_dir "$VAE_RUN" \
  --train_batch_size 128 \
  --val_batch_size 32 \
  --test_batch_size 1 \
  --num_workers 8 \
  --accelerator auto \
  --devices 1 \
  --precision bf16-mixed \
  --dataset argoverse_v2 \
  --input_dim 2 \
  --output_dim 2 \
  --hidden_dim 128 \
  --latent_dim "$LATENT_DIM" \
  --vae_num_intents 1 \
  --num_historical_steps 50 \
  --num_future_steps 60 \
  --num_freq_bands 64 \
  --num_heads 8 \
  --head_dim 16 \
  --pl2pl_radius 150 \
  --pl2a_radius 50 \
  --a2a_radius 50 \
  --num_t2m_steps 10 \
  --pl2m_radius 150 \
  --a2m_radius 50 \
  --trajectory_scale 10 \
  --pretrain_epochs 12 \
  --joint_epochs 28 \
  --calibration_epochs 5 \
  --geometry_weight 0.01 \
  --geometry_warmup_epochs 5 \
  --geometry_num_agents 84 \
  --geometry_num_directions 20 \
  --geometry_perturbation 0.05 \
  --geometry_scale_weight 0.1 \
  --geometry_anchor_weight 0.1 \
  --geometry_anchor_init_batches 4 \
  --endpoint_loss_weight 5.0 \
  --joint_selection total \
  --final_selection total
```

脚本自动完成：

1. 普通 VAE warm-up；
2. encoder/decoder 联合几何训练；
3. 基于完整训练集重算精确 latent mean/std；
4. 冻结 encoder，对 decoder 进行 geometry calibration；
5. 导出多种 checkpoint、VAE-only 权重、latent 统计和 manifest。

主要输出：

```text
VAE_RUN/
├── geometry_aware_vae_final.ckpt
├── geometry_aware_vae_weights.pt
├── exact_latent_stats.json
├── joint_selected_with_exact_stats.ckpt
└── standardized_geometry_checkpoint_manifest.json
```

后续建库和 `train_qcnet.py --ckpt_path` 应使用完整的 `geometry_aware_vae_final.ckpt`。当前 `geometry_aware_vae_weights.pt` 是独立导出格式，不是 Lightning `state_dict` checkpoint，不能直接替代前者传给现有训练/建库入口。

如已有 VAE cache，脚本会自动跳过预处理。可用：

- `--skip_prepare_vae_data`：明确跳过；
- `--force_prepare_vae_data`：删除并重建 cache；
- `--init_ckpt`：仅 warm start 权重；
- `--resume_joint_ckpt`：恢复 joint 阶段的 VAE、优化器、scheduler 和训练进度。

### 3.2 建立 prototype bank

```bash
python build_latent_prototypes_and_assign.py \
  --checkpoint "$VAE_RUN/geometry_aware_vae_final.ckpt" \
  --vae_processed_dir "$VAE_CACHE" \
  --output_dir "$BANK_DIR" \
  --root "$DATA_ROOT" \
  --num_prototypes 128 \
  --num_coarse_groups 16 \
  --min_prototypes_per_coarse 2 \
  --agent_scope all_valid \
  --batch_size 128 \
  --num_workers 8 \
  --device auto \
  --precision bf16 \
  --local_budget_method greedy_split_gain \
  --split_merge_rounds 3 \
  --support_margin_threshold 0.15
```

该脚本只读取 `VAE_CACHE` 中的训练未来轨迹，不读取 train/val 的完整 processed 数据，也不会给 train/val 写 assignment。

主要输出：

```text
BANK_DIR/
├── prototype_bank.pt
├── bank_manifest.json
└── harmful_overlap_pairs.csv
```

`prototype_bank.pt` 主要包含：

| 字段 | 含义 |
|---|---|
| `prototype_latents_centered_raw` | `[M,1,D]`，最终 medoid prototype |
| `prototype_latents_raw` | 加回 `z_mean` 的 raw latent |
| `prototype_trajectories_normalized` | VAE 解码后的归一化轨迹 |
| `prototype_trajectories_m` | 以米为单位的 prototype 轨迹 |
| `prototype_coarse_ids` | 每个 leaf prototype 的 coarse group |
| `coarse_kmeans_centers_descriptor` | coarse descriptor 中心 |
| `descriptor_normalizer/config` | train/val assignment 必需的 descriptor 变换 |
| `support_temperature_per_coarse` | Top-2 边界软标签温度 |
| `z_mean/z_std` | 建库所用 VAE latent 统计 |
| `checkpoint` | 建库所用 VAE checkpoint |

### 3.3 为 train/val 生成 assignment sidecar

```bash
python merge_assignment_into_processed.py \
  --prototype_bank "$BANK_DIR/prototype_bank.pt" \
  --checkpoint "$VAE_RUN/geometry_aware_vae_final.ckpt" \
  --root "$DATA_ROOT" \
  --out_root "$ASSIGNMENT_ROOT" \
  --splits train val \
  --train_processed_dir "$DATA_ROOT/train/processed" \
  --val_processed_dir "$DATA_ROOT/val/processed" \
  --batch_size 64 \
  --num_workers 8 \
  --device auto \
  --precision bf16 \
  --shard_size 1024 \
  --compute_trajectory_metrics \
  --updated_bank_path "$BANK_DIR/prototype_bank_with_train_stats.pt"
```

这个脚本会：

1. 用同一 VAE 对 train/val GT 编码；
2. 按 prototype bank 的 coarse descriptor 规则选择 coarse group；
3. 在组内计算 primary/secondary prototype；
4. 生成 Top-2 support、边界标记和 latent/轨迹诊断；
5. 仅使用 train split 统计每个 prototype 的残差均值、方差和标准差；
6. 输出带 train residual statistics 的新 bank。

输出结构：

```text
ASSIGNMENT_ROOT/
├── train/shard_00000.pt
├── train/shard_00001.pt
├── val/shard_00000.pt
├── assignment_manifest.json
├── assignment_manifest.pt
└── prototype_residual_stats.csv

BANK_DIR/
└── prototype_bank_with_train_stats.pt
```

后续 residual FM 推荐使用 `prototype_bank_with_train_stats.pt`，因为 `_load_prototype_bank()` 会从其中读取 `prototype_residual_std_population` 作为各 prototype 的 FM 源分布尺度。如果仍使用原始 `prototype_bank.pt`，代码会退化为使用全局 `z_std`。

`merge_assignment_into_processed.py` 默认会检查 `ArgoverseV2Dataset._merge_prototype_assignment` 是否包含所需字段；当前仓库已经包含这些字段。只有旧版 Dataset 缺字段时，它才会修改该源码并创建 `.bak_before_top2_sidecar` 备份。可通过 `--no-patch_dataset_fields` 关闭这一行为。

### 3.4 将 sidecar 嵌入新的 processed 数据集

```bash
python merge_prototype_sidecar_into_processed.py \
  --assignment_root "$ASSIGNMENT_ROOT" \
  --out_root "$MERGED_ROOT" \
  --splits train val \
  --train_processed_dir "$DATA_ROOT/train/processed" \
  --val_processed_dir "$DATA_ROOT/val/processed"
```

该脚本不会修改原始 processed 数据，而是生成：

```text
MERGED_ROOT/
├── train/
│   ├── <scenario_id>.pkl
│   └── _SUCCESS
├── val/
│   ├── <scenario_id>.pkl
│   └── _SUCCESS
└── embedded_prototype_manifest.json
```

默认只写训练必需字段。如需一并写入匹配 ADE/FDE 等诊断字段，可加：

```bash
--include_diagnostics
```

默认严格要求 sidecar 和 processed 场景一一对应，并检查：

- `scenario_id`；
- 每个场景的 agent 数量；
- 每个字段首维是否等于 agent 数量；
- processed 中是否存在未合并场景；
- sidecar 中是否存在找不到源 processed 的场景。

只建议在调试小子集时使用 `--allow_partial`。若输出目录已存在且非空，需要显式添加 `--overwrite`。

### 3.5 训练 residual flow matching

```bash
python train_qcnet.py \
  --model_type qcnet_fm \
  --root "$DATA_ROOT" \
  --train_processed_dir "$MERGED_ROOT/train" \
  --val_processed_dir "$MERGED_ROOT/val" \
  --test_processed_dir "$DATA_ROOT/test/processed" \
  --train_batch_size 16 \
  --val_batch_size 16 \
  --test_batch_size 16 \
  --num_workers 8 \
  --accelerator auto \
  --devices 1 \
  --max_epochs 64 \
  --dataset argoverse_v2 \
  --input_dim 2 \
  --output_dim 2 \
  --hidden_dim 128 \
  --latent_dim "$LATENT_DIM" \
  --vae_num_intents 1 \
  --num_historical_steps 50 \
  --num_future_steps 60 \
  --num_modes 6 \
  --num_freq_bands 64 \
  --num_map_layers 1 \
  --num_agent_layers 2 \
  --num_dec_layers 1 \
  --num_selector_layers 1 \
  --num_heads 8 \
  --head_dim 16 \
  --dropout 0.1 \
  --pl2pl_radius 150 \
  --pl2a_radius 50 \
  --a2a_radius 50 \
  --num_t2m_steps 10 \
  --pl2m_radius 150 \
  --a2m_radius 50 \
  --trajectory_scale 10 \
  --ckpt_path "$VAE_RUN/geometry_aware_vae_final.ckpt" \
  --freeze_vae \
  --residual_fm \
  --prototype_bank_path "$BANK_DIR/prototype_bank_with_train_stats.pt" \
  --prototype_sampling_topk 6 \
  --residual_samples_per_prototype 1 \
  --prototype_selector_loss_weight 0.015 \
  --selector_hard_ce_weight 1.0 \
  --selector_rank_loss_weight 0.4 \
  --selector_rank_margin 0.2 \
  --selector_num_hard_negatives 6 \
  --selector_agent_chunk_size 64 \
  --num_hist_tokens 3 \
  --num_map_tokens 4 \
  --num_agent_tokens 4 \
  --num_modality_token_refiner_layers 1 \
  --num_scene_token_refiner_layers 0 \
  --num_center_layers 1 \
  --residual_center_loss_weight 4.0 \
  --prototype_std_floor 0.01 \
  --residual_source_scale 0.7 \
  --cfg_scene_dropout 0.15 \
  --cfg_guidance_scale 1.0 \
  --decoder_aux_ade_weight 0.016 \
  --decoder_aux_fde_weight 0.008 \
  --decoder_aux_warmup_epochs 5 \
  --fm_num_steps 3 \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --T_max 64
```

使用已经嵌入 assignment 的 `$MERGED_ROOT/train` 和 `$MERGED_ROOT/val` 时，不要再传 `--prototype_assignment_dir`，否则 Dataset 会在运行时重复读取 sidecar。

`--ckpt_path` 在未传 `--resume` 时只加载 VAE、`z_mean` 和 `z_std`，QCNet encoder、selector、残差中心头和 FM decoder 均重新初始化。若希望完整恢复 FM 模型、优化器和 epoch，使用：

```bash
--ckpt_path /path/to/fm_last.ckpt --resume
```

需要满足：

```text
prototype_sampling_topk × residual_samples_per_prototype >= num_modes
```

否则模型初始化会报错。

## 4. Assignment 字段

sidecar 和嵌入后的 `data["agent"]` 中包含：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `coarse_index` | `[N]` | agent 的轨迹 descriptor coarse group |
| `prototype_index` | `[N]` | primary prototype ID |
| `secondary_prototype_index` | `[N]` | second-nearest prototype ID，无效时为 -1 |
| `support_prototype_ids` | `[N,2]` | selector 正支持 prototype |
| `support_weights` | `[N,2]` | selector 软标签，非边界样本通常为 `[1,0]` |
| `support_size` | `[N]` | 有效支持数量，通常为 1 或 2 |
| `boundary_margin` | `[N]` | `(d2-d1)/(d1+eps)` |
| `is_boundary` | `[N]` | 是否采用 Top-2 支持 |
| `valid_agent_mask` | `[N]` | 是否参与 assignment 和 FM 训练 |
| `z_gt_centered_raw` | `[N,1,D]` | GT 的 centered-raw posterior mean |
| `z_residual` | `[N,1,D]` | 相对 primary prototype 的离线残差 |
| `match_latent_raw_l2` | `[N]` | primary latent 距离，诊断字段 |
| `second_match_latent_raw_l2` | `[N]` | secondary latent 距离，诊断字段 |
| `match_ade_m` / `match_fde_m` | `[N]` | primary prototype 轨迹误差 |
| `match_traj_score_m` | `[N]` | assignment 轨迹综合诊断 |

训练时不会直接信任缓存的 `z_residual`。模型读取 `z_gt_centered_raw` 和 support，根据当次采样的 primary/secondary prototype 重新计算：

```math
r=z_{gt}-p_{\text{selected}}.
```

因此边界样本在训练时可以按 support weight 随机选择 secondary prototype，验证时固定使用 primary prototype。

## 5. 两种 assignment 读取方式

项目支持两种互斥用法。

### 方式 A：离线嵌入 processed，推荐

先执行第 3.4 节，把 assignment 写进新 processed 数据：

```bash
--train_processed_dir "$MERGED_ROOT/train"
--val_processed_dir "$MERGED_ROOT/val"
```

此时不传 `--prototype_assignment_dir`。优点是训练阶段不需要维护 sidecar shard 索引和 LRU cache。

### 方式 B：运行时读取 sidecar

可以跳过第 3.4 节，继续使用原始 processed：

```bash
--train_processed_dir "$DATA_ROOT/train/processed"
--val_processed_dir "$DATA_ROOT/val/processed"
--prototype_assignment_dir "$ASSIGNMENT_ROOT"
--prototype_assignment_strict
--assignment_cache_size 8
```

`ArgoverseV2Dataset` 会按 `scenario_id` 从 shard 中读取记录，并在返回 batch 前合并到 `data["agent"]`。

不要同时使用已嵌入 assignment 的 processed 数据和 `--prototype_assignment_dir`。

## 6. 训练与验证指标

主要训练日志：

| 指标 | 含义 |
|---|---|
| `train_fm_loss` | delta 速度场损失 |
| `train_center_loss` | 残差中心 MSE |
| `train_center_rmse` | 中心预测误差 |
| `train_center_explained_energy` | 中心头解释的原始残差能量比例 |
| `train_original_residual_rmse` | 加中心头前的 residual 尺度 |
| `train_residual_target_rmse` | 中心头之后 delta 的尺度 |
| `train_selector_loss` | selector 内部 CE + rank loss |
| `train_prototype_acc` | Top-1 是否命中有效 support |
| `train_prototype_top6` | selector Top-K 是否命中 support |
| `train_boundary_agent_rate` | Top-2 边界样本比例 |
| `train_decoder_center_ADE_m/FDE_m` | 当前 FM 终点解码后的辅助轨迹误差 |
| `train_total_loss` | 所有加权项之和 |

比较中心头是否有效时，重点观察：

```text
train_residual_target_rmse < train_original_residual_rmse
train_center_explained_energy 持续为正并上升
```

`train_qcnet.py` 对非 scorer-only 的 QCNetFM 默认以 `val_fm_loss` 保存最优 checkpoint，而不是以 `val_minADE` 或 `val_minFDE` 保存。

## 7. 验证与测试

### 7.1 验证

```bash
python val.py \
  --model QCNetFM \
  --root /path/to/eval_root \
  --batch_size 16 \
  --num_workers 8 \
  --accelerator auto \
  --devices 1 \
  --ckpt_path /path/to/fm_checkpoint.ckpt
```

当前 `val.py` 不提供 `--val_processed_dir` 或 `--prototype_assignment_dir` 参数，而 residual FM 的 validation loss 需要 GT assignment。因此 `/path/to/eval_root/val/processed` 必须指向或包含第 3.4 节生成的 merged val 数据。训练阶段由 `train_qcnet.py --val_processed_dir` 进行验证通常更直接。

### 7.2 测试与提交

```bash
python test.py \
  --model QCNetFM \
  --root "$DATA_ROOT" \
  --batch_size 16 \
  --num_workers 8 \
  --accelerator auto \
  --devices 1 \
  --ckpt_path /path/to/fm_checkpoint.ckpt
```

test split 没有 GT，因此不需要离线 assignment。selector 会直接根据场景选择 prototype。提交文件写入 checkpoint 超参数保存的：

```text
submission_dir/submission_file_name.parquet
```

加载 checkpoint 时，训练时记录的 `prototype_bank_path` 必须仍然可访问。

## 8. 关键超参数

| 参数 | 作用 |
|---|---|
| `num_prototypes` | prototype 库总大小 |
| `num_coarse_groups` | 轨迹 descriptor 粗分组数 |
| `support_margin_threshold` | Top-2 边界判定阈值 |
| `prototype_sampling_topk` | 推理时 selector 选择的 prototype 数 |
| `residual_samples_per_prototype` | 每个 prototype 的 residual 重复采样数 |
| `use_prototype_local_std` | 是否使用 prototype 局部残差标准差 |
| `residual_source_scale` | 局部标准差上的额外缩放 |
| `prototype_std_floor` | 局部标准差下限 |
| `num_center_layers` | residual center prototype-query cross-attention 层数 |
| `residual_center_loss_weight` | 中心头 MSE 的外层权重 |
| `prototype_selector_loss_weight` | selector 总损失的外层权重 |
| `selector_rank_loss_weight` | selector hard-negative rank 权重 |
| `selector_agent_chunk_size` | selector 按 agent 分块，减小显存峰值；0 表示不分块 |
| `cfg_scene_dropout` | 训练时丢弃 scene 条件的概率 |
| `cfg_guidance_scale` | 推理 CFG 强度 |
| `fm_num_steps` | Heun 积分步数 |

增大 `num_prototypes` 通常能降低 prototype quantization residual，但会增加 selector 计算量和分类难度。中心头、prototype 数量和局部残差尺度应联合观察，而不是只看 prototype 最近邻误差。

## 9. 常见问题

### 9.1 `merge_prototype_sidecar_into_processed.py` 找不到 shard

先运行 `merge_assignment_into_processed.py`。前者只负责消费：

```text
ASSIGNMENT_ROOT/train/shard_*.pt
ASSIGNMENT_ROOT/val/shard_*.pt
```

### 9.2 建库提示 `vae_num_intents` 不等于 1

当前 `build_latent_prototypes_and_assign.py` 明确只支持：

```bash
--vae_num_intents 1
```

需要用相同配置重新训练 VAE，或扩展建库脚本对多 latent token 的处理。

### 9.3 VAE checkpoint 加载失败

检查是否误用了 `geometry_aware_vae_weights.pt`。现有入口需要带 `hyper_parameters` 和 `state_dict` 的完整 `.ckpt`：

```text
geometry_aware_vae_final.ckpt
```

### 9.4 Prototype 形状不匹配

模型要求 bank 中：

```text
prototype_latents_centered_raw.shape == [M, vae_num_intents, latent_dim]
```

并且训练 FM 时的 `latent_dim`、`vae_num_intents` 与 VAE/建库阶段完全一致。

### 9.5 使用原始 bank 后 residual noise 尺度不合理

原始 `prototype_bank.pt` 尚未包含按 train assignment 统计的 residual std。请使用：

```text
prototype_bank_with_train_stats.pt
```

该文件由 `merge_assignment_into_processed.py` 生成。

### 9.6 processed 与 assignment 不匹配

assignment 必须由同一版本 processed 数据生成。脚本按 `scenario_id` 和 `num_agents` 做严格检查；若 preprocessing、agent 过滤或数据版本改变，需要重新生成 assignment。

### 9.7 显存不足

可以依次尝试：

- 减小 train/val batch size；
- 减小 `selector_agent_chunk_size`；
- 减少 `num_prototypes`；
- 关闭 selector prototype self-attention；
- 降低 scene/prototype token 数或 refiner 层数；
- 使用 `bf16-mixed`。

### 9.8 文档中的 decoder 层数

`train_geometry_aware_vae.py` 顶部描述和部分导出 metadata 仍含有历史性的 “3-layer decoder” 文案；当前 `layers/VAE.py` 的实际默认结构是 2 个 `VAEEncoderBlock` 和 2 个 `VAEDecoderBlock`。此外，该训练脚本会把 QCNetFM 的 `num_dec_layers` 固定为 2。复现实验时应以实际 checkpoint 和代码结构为准。

## 10. 代码结构

```text
.
├── predictors/
│   ├── qcnet.py
│   └── qcnet_fm.py
├── modules/
│   ├── qcnet_encoder.py
│   ├── qcnet_fm_decoder.py
│   ├── latent_space_encoder.py
│   └── latent_space_decoder.py
├── layers/
│   └── VAE.py
├── losses/
│   ├── flow_matching_loss.py
│   └── vae_loss.py
├── datasets/
│   ├── argoverse_v2_dataset.py
│   └── vae_target_dataset.py
├── datamodules/
│   └── argoverse_v2_datamodule.py
├── train_geometry_aware_vae.py
├── build_latent_prototypes_and_assign.py
├── merge_assignment_into_processed.py
├── merge_prototype_sidecar_into_processed.py
├── train_qcnet.py
├── val.py
└── test.py
```

## 11. 最短复现检查清单

- [ ] 使用 `vae_num_intents=1` 训练 geometry-aware VAE。
- [ ] 后续阶段统一使用 `geometry_aware_vae_final.ckpt`。
- [ ] 使用同一 VAE cache 和 checkpoint 建立 prototype bank。
- [ ] 先生成 assignment sidecar，再嵌入 processed。
- [ ] FM 训练优先使用 `prototype_bank_with_train_stats.pt`。
- [ ] merged processed 模式下不再传 `prototype_assignment_dir`。
- [ ] VAE、bank 和 FM 的 latent 配置完全一致。
- [ ] test 阶段保证 checkpoint 中记录的 `prototype_bank_path` 可访问。
