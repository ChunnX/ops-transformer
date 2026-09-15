#!/usr/bin/env python3
"""Simple Torch precision test for mhc/block_attention_residuals."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
TORCH_EXTENSION = REPO_ROOT / "torch_extension"
if str(TORCH_EXTENSION) not in sys.path:
    sys.path.insert(0, str(TORCH_EXTENSION))


def reference(partial_block, block_res, proj_weight, norm_weight, norm_eps):
    values = torch.cat((block_res.float(), partial_block.float().unsqueeze(1)), dim=1)
    inv_norm = torch.rsqrt(values.square().mean(dim=-1) + norm_eps)
    score_weight = proj_weight.float().reshape(-1) * norm_weight.float()
    scores = (values * inv_norm.unsqueeze(-1) * score_weight).sum(dim=-1)
    probs = scores.softmax(dim=-1)
    return (probs.unsqueeze(-1) * values).sum(dim=1).to(partial_block.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu", choices=("cpu", "npu"))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument(
        "--storage-blocks",
        type=int,
        default=None,
        help="k in storage shape [T,k,H]; slice to [T,N,H]. Default: 2*N",
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--valid-block-num", type=int, default=-1)
    parser.add_argument("--norm-eps", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.tokens, args.blocks, args.hidden_size) < 1:
        raise ValueError("tokens, blocks and hidden-size must be positive")
    storage_blocks = args.storage_blocks if args.storage_blocks is not None else args.blocks * 2
    if storage_blocks < args.blocks:
        raise ValueError("storage-blocks (k) must be >= blocks (N)")
    if args.valid_block_num not in (-1, args.blocks):
        raise ValueError("valid-block-num must be -1 or equal to blocks")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)
    partial_block = torch.randn(args.tokens, args.hidden_size, dtype=dtype)
    block_res_storage = torch.randn(args.tokens, storage_blocks, args.hidden_size, dtype=dtype)
    block_res = block_res_storage[:, : args.blocks, :]
    proj_weight = torch.randn(args.hidden_size, dtype=dtype)
    norm_weight = torch.randn(args.hidden_size, dtype=dtype)
    expected = reference(partial_block, block_res, proj_weight, norm_weight, args.norm_eps)

    if args.device == "cpu":
        actual = expected
    else:
        import torch_npu  # noqa: F401  # registers the NPU backend
        # 已安装 cann_ops_transformer_custom 时避免与仓内 torch_extension 重复注册同一 op。
        from cann_ops_transformer_custom.ops.mhc.block_attention_residuals import block_attention_residuals

        device = torch.device(f"npu:{args.device_id}")
        block_res_storage_npu = block_res_storage.to(device)
        block_res_npu = block_res_storage_npu[:, : args.blocks, :]
        if block_res_npu.is_contiguous() and storage_blocks > args.blocks:
            raise RuntimeError("expected sliced block_res to stay non-contiguous on NPU")
        with torch.no_grad():
            actual = block_attention_residuals(
                partial_block.to(device), block_res_npu, proj_weight.to(device),
                norm_weight.to(device), args.valid_block_num, args.norm_eps,
            )
        torch.npu.synchronize()
        block_res = block_res_npu

    abs_err = (actual.float().cpu() - expected.float()).abs()
    max_abs = float(abs_err.max())
    max_rel = float((abs_err / expected.float().abs().clamp_min(1.0e-6)).max())
    print(
        f"shape={tuple(actual.shape)} block_res={tuple(block_res.shape)} "
        f"storage={tuple(block_res_storage.shape)} contiguous={block_res.is_contiguous()} "
        f"max_abs={max_abs:.6e} max_rel={max_rel:.6e}"
    )
    if not torch.isfinite(actual).all() or max_abs > 2.0e-2:
        raise AssertionError("forward precision check failed")
    print("PASS")


if __name__ == "__main__":
    main()
