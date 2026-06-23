#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import importlib.util
import os
import sys
import types
from typing import Dict, Tuple

import torch
import torch_npu  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug SparseFlashMla SCFA KV stride0 path with non-contiguous KV views.")
    parser.add_argument("--device-id", type=int, default=0, help="NPU device id.")
    parser.add_argument("--layout", choices=("tnd", "bsnd"), default="tnd",
                        help="Layout to debug. Default is TND because stride0 is read from KV T axis.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size. Use >=2 to exercise stride0.")
    parser.add_argument("--allow-batch-one", action="store_true",
                        help="Allow batch=1 for smoke debug. It cannot validate dim0 stride.")
    parser.add_argument("--q-len", type=int, default=38, help="Q sequence length.")
    parser.add_argument("--ori-len", type=int, default=166, help="ori_kv sequence length.")
    parser.add_argument("--cmp-len", type=int, default=41, help="cmp_kv sequence length.")
    parser.add_argument("--topk", type=int, default=512, help="cmp_sparse_indices topk, usually 512 or 1024.")
    parser.add_argument("--cmp-ratio", type=int, default=4, help="SCFA cmp ratio.")
    parser.add_argument("--num-heads-q", type=int, default=64, help="Query head count.")
    parser.add_argument("--num-heads-kv", type=int, default=1, help="KV head count.")
    parser.add_argument("--head-dim", type=int, default=512, help="Head dim.")
    parser.add_argument("--kv-pad", type=int, default=3, help="Padding tokens on KV dim1 for non-contiguous views.")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16", help="Q/KV dtype.")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance for view vs contiguous compare.")
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance for view vs contiguous compare.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--torch-extension-path", default=None,
                        help="Optional local torch_extension path. Omit it when using an installed CANN package.")
    parser.add_argument("--launch-blocking", action="store_true",
                        help="Set ASCEND_LAUNCH_BLOCKING=1 before running the op.")
    return parser.parse_args()


def register_sparse_flash_mla(torch_extension_path: str) -> None:
    if "cann_ops_transformer" in sys.modules:
        raise RuntimeError("cann_ops_transformer was imported before sparse_flash_mla-only registration.")

    if torch_extension_path is not None:
        path = os.path.abspath(torch_extension_path)
        package_dir = os.path.join(path, "cann_ops_transformer")
        print(f"[import] insert torch_extension path: {path}")
    else:
        spec = importlib.util.find_spec("cann_ops_transformer")
        if spec is None or spec.origin is None:
            raise RuntimeError("Cannot find installed cann_ops_transformer package.")
        package_dir = os.path.dirname(os.path.abspath(spec.origin))
        print("[import] use installed cann_ops_transformer package")

    module_path = os.path.join(package_dir, "ops", "sparse_flash_mla.py")
    if not os.path.exists(module_path):
        raise RuntimeError(f"Cannot find sparse_flash_mla.py under {package_dir}.")

    package = types.ModuleType("cann_ops_transformer")
    package.__file__ = os.path.join(package_dir, "__init__.py")
    package.__path__ = [package_dir]
    package.__package__ = "cann_ops_transformer"
    sys.modules["cann_ops_transformer"] = package

    ops_package = types.ModuleType("cann_ops_transformer.ops")
    ops_package.__file__ = os.path.join(package_dir, "ops", "__init__.py")
    ops_package.__path__ = [os.path.join(package_dir, "ops")]
    ops_package.__package__ = "cann_ops_transformer.ops"
    sys.modules["cann_ops_transformer.ops"] = ops_package

    spec = importlib.util.spec_from_file_location("cann_ops_transformer.ops.sparse_flash_mla", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load sparse_flash_mla module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    print(f"[import] registered sparse_flash_mla only from: {module_path}")


def dtype_from_name(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def make_padded_bsnd(shape: Tuple[int, int, int, int], pad_s: int, dtype: torch.dtype, device: str,
                     name: str, require_non_contiguous: bool = True) -> torch.Tensor:
    batch, seq_len, num_heads, head_dim = shape
    base_shape = (batch, seq_len + pad_s, num_heads, head_dim)
    base = (torch.randn(base_shape, dtype=torch.float32, device=device) * 0.02).to(dtype)
    view = base[:, :seq_len, :, :]
    print_tensor(name, view)
    if require_non_contiguous and view.is_contiguous():
        raise RuntimeError(f"{name} is contiguous; increase --kv-pad to exercise stride0.")
    return view


def make_padded_tnd(shape: Tuple[int, int, int], pad_n: int, dtype: torch.dtype, device: str,
                    name: str, require_non_contiguous: bool = True) -> torch.Tensor:
    total_seq, num_heads, head_dim = shape
    base_shape = (total_seq, num_heads + pad_n, head_dim)
    base = (torch.randn(base_shape, dtype=torch.float32, device=device) * 0.02).to(dtype)
    view = base[:, :num_heads, :]
    print_tensor(name, view)
    if require_non_contiguous and view.is_contiguous():
        raise RuntimeError(f"{name} is contiguous; increase --kv-pad to exercise stride0.")
    return view


def make_cu_seqlens(batch: int, seq_len: int, device: str) -> torch.Tensor:
    return torch.arange(batch + 1, dtype=torch.int32, device=device) * seq_len


def print_tensor(name: str, tensor: torch.Tensor) -> None:
    storage_offset = tensor.storage_offset() if hasattr(tensor, "storage_offset") else "unknown"
    print(
        f"[tensor] {name}: shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}, "
        f"contiguous={tensor.is_contiguous()}, storage_offset={storage_offset}, dtype={tensor.dtype}, "
        f"device={tensor.device}")


def make_common_inputs(args: argparse.Namespace, device: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    torch.manual_seed(args.seed)
    layout = args.layout.upper()
    require_non_contiguous = True
    if layout == "TND":
        q = (torch.randn((args.batch * args.q_len, args.num_heads_q, args.head_dim),
                         dtype=torch.float32, device=device) * 0.02).to(dtype).contiguous()
        ori_kv = make_padded_tnd((args.batch * args.ori_len, args.num_heads_kv, args.head_dim),
                                 args.kv_pad, dtype, device, "ori_kv_view", require_non_contiguous)
        cmp_kv = make_padded_tnd((args.batch * args.cmp_len, args.num_heads_kv, args.head_dim),
                                 args.kv_pad, dtype, device, "cmp_kv_view", require_non_contiguous)
        cu_seqlens_q = make_cu_seqlens(args.batch, args.q_len, device)
        cu_seqlens_ori_kv = make_cu_seqlens(args.batch, args.ori_len, device)
        cu_seqlens_cmp_kv = make_cu_seqlens(args.batch, args.cmp_len, device)
    else:
        q = (torch.randn((args.batch, args.q_len, args.num_heads_q, args.head_dim),
                         dtype=torch.float32, device=device) * 0.02).to(dtype).contiguous()
        require_non_contiguous = args.batch >= 2
        ori_kv = make_padded_bsnd((args.batch, args.ori_len, args.num_heads_kv, args.head_dim),
                                  args.kv_pad, dtype, device, "ori_kv_view", require_non_contiguous)
        cmp_kv = make_padded_bsnd((args.batch, args.cmp_len, args.num_heads_kv, args.head_dim),
                                  args.kv_pad, dtype, device, "cmp_kv_view", require_non_contiguous)
        cu_seqlens_q = None
        cu_seqlens_ori_kv = None
        cu_seqlens_cmp_kv = None
    seqused_q = torch.full((args.batch,), args.q_len, dtype=torch.int32, device=device)
    seqused_ori_kv = torch.full((args.batch,), args.ori_len, dtype=torch.int32, device=device)
    seqused_cmp_kv = torch.full((args.batch,), args.cmp_len, dtype=torch.int32, device=device)
    cmp_residual_kv = torch.full(
        (args.batch,), args.ori_len - args.cmp_len * args.cmp_ratio, dtype=torch.int32, device=device)
    cmp_sparse_indices = make_cmp_sparse_indices(args, device)
    sinks = torch.zeros((args.num_heads_q,), dtype=torch.float32, device=device)
    print_tensor("q", q)
    print_tensor("sinks", sinks)
    print_tensor("seqused_q", seqused_q)
    print_tensor("seqused_ori_kv", seqused_ori_kv)
    print_tensor("seqused_cmp_kv", seqused_cmp_kv)
    print_tensor("cmp_residual_kv", cmp_residual_kv)
    print_tensor("cmp_sparse_indices", cmp_sparse_indices)
    if cu_seqlens_q is not None:
        print_tensor("cu_seqlens_q", cu_seqlens_q)
        print_tensor("cu_seqlens_ori_kv", cu_seqlens_ori_kv)
        print_tensor("cu_seqlens_cmp_kv", cu_seqlens_cmp_kv)
    print(
        f"[stride-check] expected ori_kv stride0={ori_kv.stride()[0]}, "
        f"contiguous stride0={ori_kv.contiguous().stride()[0]}")
    print(
        f"[stride-check] expected cmp_kv stride0={cmp_kv.stride()[0]}, "
        f"contiguous stride0={cmp_kv.contiguous().stride()[0]}")
    return {
        "q": q,
        "ori_kv": ori_kv,
        "cmp_kv": cmp_kv,
        "sinks": sinks,
        "seqused_q": seqused_q,
        "seqused_ori_kv": seqused_ori_kv,
        "seqused_cmp_kv": seqused_cmp_kv,
        "cmp_residual_kv": cmp_residual_kv,
        "cmp_sparse_indices": cmp_sparse_indices,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_ori_kv": cu_seqlens_ori_kv,
        "cu_seqlens_cmp_kv": cu_seqlens_cmp_kv,
    }


def make_cmp_sparse_indices(args: argparse.Namespace, device: str) -> torch.Tensor:
    if args.layout.upper() == "TND":
        indices_shape = (args.batch * args.q_len, args.num_heads_kv, args.topk)
    else:
        indices_shape = (args.batch, args.q_len, args.num_heads_kv, args.topk)
    indices = torch.full(indices_shape, -1, dtype=torch.int32, device=device)
    valid = min(args.cmp_len, args.topk)
    if valid > 0:
        values = torch.arange(valid, dtype=torch.int32, device=device)
        if args.layout.upper() == "TND":
            indices[:, :, :valid] = values.reshape(1, 1, valid)
        else:
            indices[:, :, :, :valid] = values.reshape(1, 1, 1, valid)
    return indices


def build_metadata(args: argparse.Namespace, common: Dict[str, torch.Tensor]) -> torch.Tensor:
    metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
        num_heads_q=args.num_heads_q,
        num_heads_kv=args.num_heads_kv,
        head_dim=args.head_dim,
        cu_seqlens_q=common["cu_seqlens_q"],
        cu_seqlens_ori_kv=common["cu_seqlens_ori_kv"],
        cu_seqlens_cmp_kv=common["cu_seqlens_cmp_kv"],
        seqused_q=common["seqused_q"],
        seqused_ori_kv=common["seqused_ori_kv"],
        seqused_cmp_kv=common["seqused_cmp_kv"],
        cmp_residual_kv=common["cmp_residual_kv"],
        batch_size=args.batch,
        max_seqlen_q=args.q_len,
        max_seqlen_ori_kv=args.ori_len,
        max_seqlen_cmp_kv=args.cmp_len,
        ori_topk=0,
        cmp_topk=args.topk,
        cmp_ratio=args.cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q=args.layout.upper(),
        layout_kv=args.layout.upper(),
        has_ori_kv=True,
        has_cmp_kv=True)
    torch.npu.synchronize()
    print_tensor("scfa.metadata", metadata)
    return metadata


def run_sparse_flash_mla(args: argparse.Namespace, case_name: str, common: Dict[str, torch.Tensor],
                         ori_kv: torch.Tensor, cmp_kv: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
    out, softmax_lse = torch.ops.cann_ops_transformer.sparse_flash_mla(
        common["q"],
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=common["cmp_sparse_indices"],
        cu_seqlens_q=common["cu_seqlens_q"],
        cu_seqlens_ori_kv=common["cu_seqlens_ori_kv"],
        cu_seqlens_cmp_kv=common["cu_seqlens_cmp_kv"],
        seqused_q=common["seqused_q"],
        seqused_ori_kv=common["seqused_ori_kv"],
        seqused_cmp_kv=common["seqused_cmp_kv"],
        sinks=common["sinks"],
        metadata=metadata,
        cmp_residual_kv=common["cmp_residual_kv"],
        softmax_scale=args.head_dim ** -0.5,
        cmp_ratio=args.cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q=args.layout.upper(),
        layout_kv=args.layout.upper(),
        topk_value_mode=1,
        return_softmax_lse=False)
    torch.npu.synchronize()
    print_tensor(f"{case_name}.out", out)
    print_tensor(f"{case_name}.softmax_lse", softmax_lse)
    return out


def compare_outputs(case_name: str, view_out: torch.Tensor, contiguous_out: torch.Tensor,
                    atol: float, rtol: float) -> None:
    view_cpu = view_out.float().cpu()
    contiguous_cpu = contiguous_out.float().cpu()
    diff = (view_cpu - contiguous_cpu).abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / contiguous_cpu.abs().clamp_min(1e-8)).max().item())
    close = torch.isclose(view_cpu, contiguous_cpu, atol=atol, rtol=rtol)
    close_percent = float(close.float().mean().item() * 100.0)
    finite = bool(torch.isfinite(view_cpu).all().item())
    print(
        f"[compare] {case_name}: finite={finite}, max_abs={max_abs:.8f}, "
        f"max_rel={max_rel:.8f}, close_percent={close_percent:.4f}, atol={atol}, rtol={rtol}")
    if not finite:
        raise RuntimeError(f"{case_name} output contains NaN or Inf.")
    if not torch.allclose(view_cpu, contiguous_cpu, atol=atol, rtol=rtol):
        raise RuntimeError(
            f"{case_name} non-contiguous KV output differs from contiguous output. "
            f"max_abs={max_abs}, max_rel={max_rel}")


def run_scfa(args: argparse.Namespace, common: Dict[str, torch.Tensor]) -> None:
    print("[case] start scfa")
    metadata = build_metadata(args, common)
    view_out = run_sparse_flash_mla(
        args, "scfa.view", common, common["ori_kv"], common["cmp_kv"], metadata)
    contiguous_out = run_sparse_flash_mla(
        args, "scfa.contiguous", common, common["ori_kv"].contiguous(), common["cmp_kv"].contiguous(), metadata)
    compare_outputs("scfa", view_out, contiguous_out, args.atol, args.rtol)
    print("[case] pass scfa")


def main() -> None:
    args = parse_args()
    if args.launch_blocking:
        os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
    if args.layout.upper() == "BSND":
        if args.batch < 2 and not args.allow_batch_one:
            raise ValueError("--batch should be >= 2 because BSND validates dim0 stride through batch stride.")
        if args.batch < 2:
            print("[warn] batch=1 cannot validate BSND dim0 stride; use it only for smoke debug.")
    if args.kv_pad <= 0:
        raise ValueError("--kv-pad should be positive to create non-contiguous KV views.")

    register_sparse_flash_mla(args.torch_extension_path)
    device = f"npu:{args.device_id}"
    torch.npu.set_device(device)
    dtype = dtype_from_name(args.dtype)
    print(
        f"[config] device={device}, dtype={dtype}, batch={args.batch}, q_len={args.q_len}, "
        f"ori_len={args.ori_len}, cmp_len={args.cmp_len}, topk={args.topk}, cmp_ratio={args.cmp_ratio}, "
        f"num_heads_q={args.num_heads_q}, num_heads_kv={args.num_heads_kv}, head_dim={args.head_dim}, "
        f"kv_pad={args.kv_pad}, layout={args.layout.upper()}")

    common = make_common_inputs(args, device, dtype)
    run_scfa(args, common)
    print("[result] PASS: non-contiguous KV view outputs match contiguous KV outputs.")


if __name__ == "__main__":
    main()
