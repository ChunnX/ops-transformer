#!/usr/bin/env python3
# coding=utf-8
# ------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it
# under the terms and conditions of CANN Open Software License Agreement
# Version 2.0 (the "License").
# ------------------------------------------------------------------------------
"""AllGatherMatmulV3 (all_gather_quant_matmul) golden precision test.

Semantics (per rank i):
  1. AllGather: A_all = concat([A_0, ..., A_{R-1}])  -> [world_size*M, K]
  2. Matmul:    Y_i = dequant(A_all) x dequant(B_i)^T -> [world_size*M, N]  (bf16/fp16)

Op inputs (per rank):
  x1:        [M, K]            fp8 e4m3/e5m2, ND
  x2:        [N, K]            fp8 e4m3/e5m2, ND (transposed B, DN layout)
  x1_scale:  [M, ceil(K/64), 2]  fp8 e8m0
  x2_scale:  [N, ceil(K/64), 2]  fp8 e8m0

Run: qrun "python test_golden_allgathermm.py"
"""

import os
import argparse
import math

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Manager
from ml_dtypes import float8_e4m3fn as np_fp8_e4m3fn
from ml_dtypes import float8_e5m2 as np_fp8_e5m2
from ml_dtypes import float8_e8m0fnu as np_fp8_e8m0fnu

import cann_ops_transformer

os.environ.setdefault('HCCL_BUFFSIZE', '2000')  # comm window must hold ws*M*K data+scale per rank

MASTER_IP = "127.0.0.1"
MASTER_PORT = 50020

FP8_NUMPY = {"e4m3fn": np_fp8_e4m3fn, "e5m2": np_fp8_e5m2}
FP8_TORCH = {"e4m3fn": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}

# FP4 E2M1 code table (sans sign), exp bias = 1
FP4_CODE_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def gen_fp4_packed(shape, lo=1, hi=8):
    """Generate random fp4 e2m1 data packed as uint8 [..., K//2] (low nibble = even index)."""
    codes = np.random.randint(lo, hi, shape).astype(np.uint8)  # 1..7 -> 0.5..6.0
    signs = np.random.randint(0, 2, shape).astype(np.uint8)
    nib = (signs << 3) | codes
    flat = nib.reshape(-1)
    packed = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)
    return packed.reshape(shape[:-1] + (shape[-1] // 2,))


def decode_fp4(packed):
    """Decode packed uint8 [..., K//2] -> float32 [..., K]."""
    flat = packed.reshape(-1)
    nib = np.empty(flat.size * 2, dtype=np.uint8)
    nib[0::2] = flat & 0xF
    nib[1::2] = flat >> 4
    signs = (nib >> 3) & 1
    codes = nib & 0x7
    vals = FP4_CODE_TABLE[codes] * np.where(signs == 1, -1.0, 1.0).astype(np.float32)
    return vals.reshape(packed.shape[:-1] + (packed.shape[-1] * 2,))

Y_DTYPE_BF16 = 15
Y_DTYPE_FP16 = 5
MX_QUANT_MODE = 6
MX_GROUP_SIZES = [1, 1, 32]
MX_GROUP_SIZE = 32

ERROR_TOL = 1e-2
FIXED_SEED = 42

torch_npu.npu.config.allow_internal_format = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="all_gather_quant_matmul golden precision test (MX quant fp8, urma)")
    parser.add_argument("--m", type=int, default=1010, help="per-rank M")
    parser.add_argument("--k", type=int, default=2560, help="full K (not split across ranks)")
    parser.add_argument("--n", type=int, default=2560, help="output dim N")
    parser.add_argument("--rank-per-dev", type=int, default=4, help="ranks per NPU device")
    parser.add_argument("--server-num", type=int, default=1, help="server count")
    parser.add_argument("--dtype", type=str, default="e4m3fn", choices=["e4m3fn", "e5m2", "fp4e2m1"])
    parser.add_argument("--x1dtype", type=str, default=None, choices=["e4m3fn", "e5m2", "fp4e2m1"])
    parser.add_argument("--x2dtype", type=str, default=None, choices=["e4m3fn", "e5m2", "fp4e2m1"])
    parser.add_argument("--ydtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--bias", action="store_true", help="enable bias")
    args = parser.parse_args()
    if args.x1dtype is None:
        args.x1dtype = args.dtype
    if args.x2dtype is None:
        args.x2dtype = args.dtype
    return args


def set_device(rank, rank_per_dev):
    real_device = rank % rank_per_dev
    torch_npu.npu.set_device(real_device)
    print(f"[INFO] rank: {rank}, device: {real_device}")


def init_hccl_comm(rank, world_size):
    dist.init_process_group(
        backend='hccl',
        rank=rank,
        world_size=world_size,
        init_method=f'tcp://{MASTER_IP}:{MASTER_PORT}'
    )
    global_group = dist.new_group(ranks=list(range(world_size)))
    print(f"[INFO] device_{rank} HCCL comm group created")
    return global_group


def np_fp8_to_torch(np_arr, torch_dtype):
    return torch.from_numpy(np_arr.view(np.uint8)).view(torch_dtype)


def dequant_mx(data_fp8, scale_e8m0, k):
    """Dequant MX fp8 data [rows, K] with scale [rows, ceil(K/64), 2] -> fp32 [rows, K]."""
    rows = data_fp8.shape[0]
    scale_flat = scale_e8m0.reshape(rows, -1)                       # [rows, ceil(K/64)*2]
    scale_full = np.repeat(scale_flat, MX_GROUP_SIZE, axis=-1)[:, :k]  # [rows, K]
    return data_fp8.astype(np.float32) * scale_full.astype(np.float32)


def gen_low_precision(shape, dtype_name):
    """Generate [.., K] data: fp8 -> numpy fp8 array; fp4 -> packed uint8 array."""
    if dtype_name == "fp4e2m1":
        return gen_fp4_packed(shape)
    return np.random.uniform(1, 8, shape).astype(FP8_NUMPY[dtype_name])


def to_float32(arr, dtype_name):
    if dtype_name == "fp4e2m1":
        return decode_fp4(arr)
    return arr.astype(np.float32)


def to_npu_tensor(arr, dtype_name):
    if dtype_name == "fp4e2m1":
        return torch.from_numpy(arr).npu().view(torch.float4_e2m1fn_x2)
    return np_fp8_to_torch(arr, FP8_TORCH[dtype_name]).npu()


def generate_inputs(m, k, n, world_size, x1dtype, x2dtype):
    """Generate all ranks' inputs (identical on every rank thanks to fixed seeds)."""
    a_fp8_list, a_scale_list, b_fp8_list, b_scale_list = [], [], [], []
    for r in range(world_size):
        np.random.seed(FIXED_SEED + r)
        a_fp8_list.append(gen_low_precision((m, k), x1dtype))
        a_scale_list.append(np.random.uniform(1, 8, (m, math.ceil(k / 64), 2)).astype(np_fp8_e8m0fnu))
        b_fp8_list.append(gen_low_precision((n, k), x2dtype))  # [N, K] physical
        b_scale_list.append(np.random.uniform(1, 8, (n, math.ceil(k / 64), 2)).astype(np_fp8_e8m0fnu))
    return a_fp8_list, a_scale_list, b_fp8_list, b_scale_list


def compute_golden(inputs, rank, m, k, n, world_size, ydtype, x1dtype, x2dtype, bias_tensor=None):
    a_fp8_list, a_scale_list, b_fp8_list, b_scale_list = inputs

    def dequant(data, scale):
        rows = data.shape[0]
        data_f32 = to_float32(data, "fp4e2m1" if data.dtype == np.uint8 else "fp8")
        scale_flat = scale.reshape(rows, -1)
        scale_full = np.repeat(scale_flat, MX_GROUP_SIZE, axis=-1)[:, :k]
        return data_f32 * scale_full.astype(np.float32)

    a_deq_all = np.concatenate(
        [dequant(a_fp8_list[r], a_scale_list[r]) for r in range(world_size)], axis=0)  # [ws*M, K]
    b_deq_nk = dequant(b_fp8_list[rank], b_scale_list[rank])                              # [N, K]
    out = torch.from_numpy(a_deq_all) @ torch.from_numpy(b_deq_nk.T)                      # [ws*M, N]
    if bias_tensor is not None:
        out = out + bias_tensor.cpu().float()
    return out.to(torch.bfloat16 if ydtype == "bf16" else torch.float16)


def worker(queue, rank, args, world_size):
    try:
        set_device(rank, args.rank_per_dev)
        group = init_hccl_comm(rank, world_size)

        m, k, n = args.m, args.k, args.n
        y_dtype_val = Y_DTYPE_BF16 if args.ydtype == "bf16" else Y_DTYPE_FP16

        inputs = generate_inputs(m, k, n, world_size, args.x1dtype, args.x2dtype)
        a_fp8_list, a_scale_list, b_fp8_list, b_scale_list = inputs

        x1 = to_npu_tensor(a_fp8_list[rank], args.x1dtype)
        x2 = to_npu_tensor(b_fp8_list[rank], args.x2dtype).t()  # [N,K] → .t() view [K,N]，aclnn 内部翻回
        x1_scale = np_fp8_to_torch(a_scale_list[rank], torch.float8_e8m0fnu).npu()
        x2_scale = np_fp8_to_torch(b_scale_list[rank], torch.float8_e8m0fnu).npu()

        bias_tensor = None
        if args.bias:
            np.random.seed(FIXED_SEED + 999)
            bias_np = np.random.uniform(-1, 1, (n,)).astype(np.float32)
            bias_tensor = torch.from_numpy(bias_np).npu()

        print(f"[INFO] rank {rank} x1.shape={x1.shape} x2.shape={x2.shape} "
              f"x1_scale.shape={x1_scale.shape} x2_scale.shape={x2_scale.shape} "
              f"bias={'on' if bias_tensor is not None else 'off'}", flush=True)

        y, _, _ = cann_ops_transformer.all_gather_quant_matmul(
            x1,
            x2,
            group,
            x1_scale=x1_scale,
            x2_scale=x2_scale,
            bias=bias_tensor,
            y_dtype=y_dtype_val,
            group_sizes=MX_GROUP_SIZES,
            comm_mode="urma",
        )
        torch.npu.synchronize()

        golden = compute_golden(inputs, rank, m, k, n, world_size, args.ydtype, args.x1dtype, args.x2dtype, bias_tensor)
        golden_dtype = torch.bfloat16 if args.ydtype == "bf16" else torch.float16
        y_cpu = y.cpu().to(golden_dtype)

        expected_shape = (m * world_size, n)
        assert tuple(y.shape) == expected_shape, f"y shape {tuple(y.shape)} != {expected_shape}"

        abs_err = (y_cpu.float() - golden.float()).abs()
        denom = golden.float().abs().clamp_min(1e-8)
        rel_err = abs_err / denom
        numel = expected_shape[0] * expected_shape[1]
        over_tol = (abs_err > ERROR_TOL).sum().item()
        passed = torch.allclose(golden, y_cpu, rtol=ERROR_TOL, atol=ERROR_TOL, equal_nan=True)

        print(f"\n{'=' * 60}")
        print(f"[Rank {rank}] shape={expected_shape}, elements={numel}")
        print(f"  abs_err: max={abs_err.max().item():.6e}, mean={abs_err.mean().item():.6e}")
        print(f"  rel_err: max={rel_err.max().item():.6e}")
        print(f"  count(|abs_err| > {ERROR_TOL:g}): {over_tol} / {numel}")
        cr, cc = min(4, expected_shape[0]), min(4, n)
        print(f"  golden (top-left {cr}x{cc}):\n{golden[:cr, :cc]}")
        print(f"  npu    (top-left {cr}x{cc}):\n{y_cpu[:cr, :cc]}")
        print(f"  [{'PASS' if passed else 'FAIL'}] Rank {rank} (rtol=atol={ERROR_TOL})")

        dist.barrier()
        queue.put([rank, "PASS" if passed else "FAIL"])

    except Exception as e:
        print(f"[ERROR] rank {rank} failed: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            dist.barrier(timeout=5)
        except Exception:
            pass
        queue.put([rank, f"error: {str(e)}"])

    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        torch.npu.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    world_size = args.rank_per_dev * args.server_num

    print(f"[INFO] AllGatherMatmulV3 golden test, world_size={world_size}, "
          f"m={args.m}, k={args.k}, n={args.n}, x1={args.x1dtype}, x2={args.x2dtype}, y={args.ydtype}")

    mp.set_start_method('spawn', force=True)
    manager = Manager()
    result_queue = manager.Queue()

    proc_list = []
    for rank in range(world_size):
        proc = Process(target=worker, args=(result_queue, rank, args, world_size))
        proc.start()
        proc_list.append(proc)

    for proc in proc_list:
        proc.join()

    rank_results = [None] * world_size
    for _ in range(world_size):
        rank_id, result = result_queue.get()
        rank_results[rank_id] = result
        print(f"[INFO] rank {rank_id} result: {result}")

    all_pass = all(r == "PASS" for r in rank_results)
    print(f"\n{'=' * 60}")
    print("[PASS] all ranks matched golden!" if all_pass
          else f"[FAIL] failed ranks: {rank_results}")
    print(f"{'=' * 60}")
