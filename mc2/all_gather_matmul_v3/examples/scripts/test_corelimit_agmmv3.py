#!/usr/bin/env python3
# coding=utf-8
"""AllGatherMatmulV3 控核(core limit)专项验证.

场景:
  A. C/V 不规则数目(如 C=12 V=18)且 min(C,V) >= rank_size -> 期望精度 PASS
  B. min(C,V) < rank_size -> 期望 tiling 拦截 rank_size > min(aicNum, aivNum)

Run:
  qrun "python3 test_corelimit_agmmv3.py --world-size 4 --cube 12 --vector 18"
  qrun "python3 test_corelimit_agmmv3.py --world-size 4 --cube 2 --vector 18"
"""

import argparse
import os

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from ml_dtypes import float8_e4m3fn as np_fp8_e4m3fn
from ml_dtypes import float8_e8m0fnu as np_fp8_e8m0fnu

import cann_ops_transformer

os.environ.setdefault('HCCL_BUFFSIZE', '2000')

MASTER_IP = "127.0.0.1"
MASTER_PORT = 50022

M, K, N = 64, 256, 64
KG = (K + 63) // 64
MX_GROUP_SIZE = 32
ERROR_TOL = 1e-2


def gen_x1(rank):
    np.random.seed(42 + rank)
    return np.random.uniform(-1, 1, (M, K)).astype(np_fp8_e4m3fn)


def gen_x2(rank):
    np.random.seed(1000 + rank)
    return np.random.uniform(-1, 1, (N, K)).astype(np_fp8_e4m3fn)


def gen_scale(seed, rows):
    np.random.seed(seed)
    return np.random.uniform(1, 2, (rows, KG, 2)).astype(np_fp8_e8m0fnu)


def to_torch_fp8(arr):
    return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e4m3fn).npu()


def to_torch_scale(arr):
    return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e8m0fnu).npu()


def dequant_mx(data_fp8, scale_e8m0, k):
    rows = data_fp8.shape[0]
    scale_flat = scale_e8m0.reshape(rows, -1)
    scale_full = np.repeat(scale_flat, MX_GROUP_SIZE, axis=-1)[:, :k]
    return data_fp8.astype(np.float32) * scale_full.astype(np.float32)


def golden_for_rank(my_rank, world_size):
    gathered = []
    for r in range(world_size):
        gathered.append(dequant_mx(gen_x1(r), gen_scale(2000 + r, M), K))
    a_all = np.concatenate(gathered, axis=0)  # [world_size*M, K]
    b_me = dequant_mx(gen_x2(my_rank), gen_scale(3000 + my_rank, N), K)  # [N, K]
    return torch.from_numpy(a_all @ b_me.T).to(torch.bfloat16)


def worker(rank, world_size, cube, vector, use_stream_limit=False):
    torch_npu.npu.set_device(rank % 4)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size,
                            init_method=f'tcp://{MASTER_IP}:{MASTER_PORT}')
    group = dist.new_group(ranks=list(range(world_size)))

    if cube > 0 or vector > 0:
        if use_stream_limit:
            stream = torch.npu.current_stream()
            torch.npu.set_stream_limit(stream, cube, vector)
            print(f"[rank{rank}] set_stream_limit(cube={cube}, vector={vector})", flush=True)
        else:
            torch.npu.set_device_limit(rank % 4, cube, vector)
            print(f"[rank{rank}] set_device_limit(cube={cube}, vector={vector}) -> "
                  f"{torch.npu.get_device_limit(rank % 4)}", flush=True)

    x1 = to_torch_fp8(gen_x1(rank))
    x2 = to_torch_fp8(gen_x2(rank)).t()
    x1s = to_torch_scale(gen_scale(2000 + rank, M))
    x2s = to_torch_scale(gen_scale(3000 + rank, N))

    try:
        y, _, _ = cann_ops_transformer.all_gather_quant_matmul(
            x1, x2, group, x1_scale=x1s, x2_scale=x2s, comm_mode="urma")
        torch.npu.synchronize()
        golden = golden_for_rank(rank, world_size)
        y_cpu = y.cpu().to(torch.bfloat16)
        has_bad = torch.isnan(y_cpu.float()).any().item() or torch.isinf(y_cpu.float()).any().item()
        ok = torch.allclose(golden.float(), y_cpu.float(), rtol=ERROR_TOL, atol=ERROR_TOL)
        if has_bad:
            print(f"[rank{rank}] FAIL: 输出含 NaN/Inf", flush=True)
        elif ok:
            print(f"[rank{rank}] PASS: C={cube} V={vector} 精度通过", flush=True)
        else:
            diff = (golden.float() - y_cpu.float()).abs().max().item()
            print(f"[rank{rank}] FAIL: 精度不匹配 max_diff={diff}", flush=True)
    except Exception as e:
        print(f"[rank{rank}] 拦截/异常 [{type(e).__name__}] {str(e)[:250]}", flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--cube", type=int, default=-1, help="AIC(cube) 核数限制, -1 不限")
    parser.add_argument("--vector", type=int, default=-1, help="AIV(vector) 核数限制, -1 不限")
    parser.add_argument("--stream-limit", action="store_true", help="用 Stream 级控核(默认进程级)")
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    proc_list = []
    for rank in range(args.world_size):
        proc = Process(target=worker, args=(rank, args.world_size, args.cube, args.vector, args.stream_limit))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()
    print("[TEST] done")


if __name__ == "__main__":
    main()
