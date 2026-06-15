#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""分析 VAE 编码均值 mu 的相关矩阵与特征值。

默认读取 VAE 预处理目录中的 .pt 文件，每个文件应包含：
    target:       [N, T, D]
    predict_mask: [N, T]

示例：
python analyze_vae_latent_corr.py \
  --ckpt /path/to/vae.ckpt \
  --vae_processed_dir /path/to/vae_processed \
  --device cuda \
  --output_dir latent_corr_results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from predictors import QCNetFM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--vae_processed_dir", type=str, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--target_scale",
        type=float,
        default=1.0,
        help="预处理数据通常为 1.0；若是原始米制轨迹则设为 0.1。",
    )
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--corr_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="latent_corr_results")
    return parser.parse_args()


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_batch(obj, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError(f"{path} 不是 dict，而是 {type(obj)}")
    if "target" not in obj or "predict_mask" not in obj:
        raise KeyError(f"{path} 缺少 target 或 predict_mask，keys={list(obj.keys())}")

    target = torch.as_tensor(obj["target"]).float()
    mask = torch.as_tensor(obj["predict_mask"]).bool()

    if target.ndim > 3:
        target = target.reshape(-1, target.shape[-2], target.shape[-1])
    if mask.ndim > 2:
        mask = mask.reshape(-1, mask.shape[-1])

    if target.ndim != 3 or mask.ndim != 2:
        raise ValueError(
            f"{path} 形状错误：target={tuple(target.shape)}, mask={tuple(mask.shape)}"
        )
    if target.shape[:2] != mask.shape:
        raise ValueError(
            f"{path} target/mask 不匹配：target={tuple(target.shape)}, mask={tuple(mask.shape)}"
        )
    return target, mask


@torch.inference_mode()
def collect_mu(
    model: QCNetFM,
    files: List[Path],
    device: torch.device,
    target_scale: float,
    max_samples: int,
) -> List[torch.Tensor]:
    """返回 K 个 Tensor，每个形状为 [N_valid, latent_dim]。"""
    chunks: List[List[torch.Tensor]] | None = None
    counts: List[int] | None = None

    model.eval()
    model.latent_encoder.eval()

    for file_idx, path in enumerate(files, 1):
        target, mask = extract_batch(safe_load(path), path)
        valid = mask.any(dim=-1)
        if not valid.any():
            continue

        target = (target * target_scale).to(device, non_blocking=True)
        mask_device = mask.to(device, non_blocking=True)
        valid_device = valid.to(device)

        # latent_encoder.encode() 返回 mu，形状 [N, K, H]
        mu = model.latent_encoder.encode(target, predict_mask=mask_device)
        if mu.ndim != 3:
            raise RuntimeError(f"encode() 返回形状异常：{tuple(mu.shape)}")

        mu = mu[valid_device].float().cpu()
        k_num = mu.size(1)

        if chunks is None:
            chunks = [[] for _ in range(k_num)]
            counts = [0 for _ in range(k_num)]

        for k in range(k_num):
            part = mu[:, k, :]
            if max_samples > 0:
                remain = max_samples - counts[k]
                if remain <= 0:
                    continue
                part = part[:remain]
            if part.numel() > 0:
                chunks[k].append(part)
                counts[k] += part.size(0)

        if file_idx == 1 or file_idx % 100 == 0 or file_idx == len(files):
            print(f"[{file_idx}/{len(files)}] 每个 intent 已收集样本数：{counts}")

        if max_samples > 0 and all(n >= max_samples for n in counts):
            break

    if chunks is None:
        raise RuntimeError("没有收集到有效样本。")

    result = []
    for k, parts in enumerate(chunks):
        if not parts:
            raise RuntimeError(f"intent {k} 没有样本。")
        result.append(torch.cat(parts, dim=0))
    return result


def analyze(mu: torch.Tensor, eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    x = mu.double()
    n, dim = x.shape
    if n < 2:
        raise ValueError("样本数必须至少为 2。")

    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=True).clamp_min(eps)
    z = (x - mean) / std

    corr = z.T @ z / (n - 1)
    corr = 0.5 * (corr + corr.T)
    corr.fill_diagonal_(1.0)

    eigvals, eigvecs = torch.linalg.eigh(corr)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals_nonneg = eigvals.clamp_min(0.0)
    ratio = eigvals_nonneg / eigvals_nonneg.sum().clamp_min(eps)
    cumulative = ratio.cumsum(dim=0)

    positive = eigvals_nonneg[eigvals_nonneg > eps]
    condition_number = (
        positive.max() / positive.min()
        if positive.numel() > 0
        else torch.tensor(float("inf"), dtype=x.dtype)
    )

    p = ratio[ratio > eps]
    effective_rank = torch.exp(-(p * torch.log(p)).sum())

    offdiag = corr.clone()
    offdiag.fill_diagonal_(0.0)

    return {
        "num_samples": torch.tensor(n),
        "mean": mean,
        "std": std,
        "z_mean": z.mean(dim=0),
        "z_std": z.std(dim=0, unbiased=True),
        "corr": corr,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "explained_ratio": ratio,
        "cumulative_ratio": cumulative,
        "max_abs_offdiag": offdiag.abs().max(),
        "condition_number": condition_number,
        "effective_rank": effective_rank,
    }


def print_result(name: str, s: Dict[str, torch.Tensor], threshold: float) -> None:
    np.set_printoptions(precision=4, suppress=True, linewidth=200, floatmode="fixed")
    corr = s["corr"]
    dim = corr.size(0)

    print("\n" + "=" * 88)
    print(name)
    print("=" * 88)
    print(f"样本数：{int(s['num_samples'].item()):,}")
    print(f"latent_dim：{dim}")
    print("\n原始 mu 均值：")
    print(s["mean"].numpy())
    print("\n原始 mu 标准差：")
    print(s["std"].numpy())
    print("\n标准化后均值：")
    print(s["z_mean"].numpy())
    print("\n标准化后标准差：")
    print(s["z_std"].numpy())
    print("\n相关系数矩阵：")
    print(corr.numpy())
    print("\n特征值（降序，总和应约等于 latent_dim）：")
    print(s["eigvals"].numpy())
    print("\n解释比例：")
    print(s["explained_ratio"].numpy())
    print("\n累计解释比例：")
    print(s["cumulative_ratio"].numpy())
    print(f"\n最大非对角 |corr|：{s['max_abs_offdiag'].item():.6f}")
    print(f"条件数：{s['condition_number'].item():.6f}")
    print(f"有效秩：{s['effective_rank'].item():.6f} / {dim}")

    pairs = []
    for i in range(dim):
        for j in range(i + 1, dim):
            value = corr[i, j].item()
            if abs(value) >= threshold:
                pairs.append((i, j, value))

    print(f"\n|corr| >= {threshold:.2f} 的维度对：")
    if not pairs:
        print("  无")
    else:
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for i, j, value in pairs:
            print(f"  dim {i:02d} <-> dim {j:02d}: corr={value:+.6f}")


def save_result(output_dir: Path, name: str, mu: torch.Tensor, s: Dict[str, torch.Tensor]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = name.lower().replace(" ", "_")

    corr = s["corr"].numpy()
    with (output_dir / f"{stem}_correlation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dim"] + [f"dim_{i:02d}" for i in range(corr.shape[1])])
        for i, row in enumerate(corr):
            writer.writerow([f"dim_{i:02d}"] + row.tolist())

    np.save(output_dir / f"{stem}_correlation.npy", corr)
    np.save(output_dir / f"{stem}_eigenvalues.npy", s["eigvals"].numpy())

    torch.save(
        {
            "mu": mu,
            **{k: v.cpu() if torch.is_tensor(v) else v for k, v in s.items()},
        },
        output_dir / f"{stem}_full.pt",
    )


def main() -> None:
    args = parse_args()
    ckpt = Path(args.ckpt)
    data_dir = Path(args.vae_processed_dir)
    output_dir = Path(args.output_dir)

    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)

    files = sorted(data_dir.rglob("*.pt"))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"{data_dir} 下没有 .pt 文件。")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。")

    print(f"checkpoint: {ckpt}")
    print(f"data dir:   {data_dir}")
    print(f"device:     {device}")
    print(f"pt files:   {len(files)}")
    print(f"target_scale: {args.target_scale}")

    model = QCNetFM.load_from_checkpoint(
        str(ckpt),
        map_location=device,
        strict=False,
    ).to(device)

    mu_by_intent = collect_mu(
        model=model,
        files=files,
        device=device,
        target_scale=args.target_scale,
        max_samples=args.max_samples,
    )

    for k, mu in enumerate(mu_by_intent):
        name = f"intent_{k}"
        stats = analyze(mu)
        print_result(name, stats, args.corr_threshold)
        save_result(output_dir, name, mu, stats)

    if len(mu_by_intent) > 1:
        combined = torch.cat(mu_by_intent, dim=0)
        stats = analyze(combined)
        print_result("combined_all_intents", stats, args.corr_threshold)
        save_result(output_dir, "combined_all_intents", combined, stats)

    print(f"\n结果已保存至：{output_dir.resolve()}")


if __name__ == "__main__":
    main()