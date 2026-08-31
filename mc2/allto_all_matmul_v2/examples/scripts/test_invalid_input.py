#!/usr/bin/env python3
"""Diagnostic: feed an invalid dtype / invalid shape to all_to_all_quant_matmul
to check whether the HOST side rejects it (validation error) instead of the
device-side 507015 aicore exception.
"""

import os
import argparse
import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Manager

import cann_ops_transformer
from ml_dtypes import float8_e4m3fn as np_fp8
from ml_dtypes import float8_e8m0fnu as np_fp8e8m0

MASTER_IP = "127.0.0.1"
MASTER_PORT = 50123
WORLD_SIZE = 2

MX_QUANT_MODE = 6
MX_GROUP_SIZES = [1, 1, 32]
SCALE_DTYPE_FP8_E8M0 = 293
Y_DTYPE_BF16 = 15


def make_fp8(shape):
    arr = np.random.uniform(1, 8, shape).astype(np_fp8)
    return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e4m3fn)


def make_e8m0(shape):
    arr = np.random.uniform(1, 8, shape).astype(np_fp8e8m0)
    return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e8m0fnu)


def worker(queue, rank, mode):
    try:
        torch_npu.npu.set_device(rank % 2)
        dist.init_process_group(
            "hccl",
            rank=rank,
            world_size=WORLD_SIZE,
            init_method=f"tcp://{MASTER_IP}:{MASTER_PORT}",
        )
        group = dist.new_group(list(range(WORLD_SIZE)))

        bs, h, n = 1024, 4096, 8192

        if mode == "dtype":
            # INVALID dtype: fp16 instead of fp8
            x1 = torch.randn(bs, h, dtype=torch.float16).npu()
            x2 = torch.randn(n, h * WORLD_SIZE, dtype=torch.float16).npu()
            s1 = make_e8m0((bs, (h + 63) // 64, 2)).npu()
            s2 = make_e8m0((n, (h * WORLD_SIZE + 63) // 64, 2)).npu()
        else:  # shape: K dimension mismatch (x2 K_total != ka * world_size)
            x1 = make_fp8((bs, h)).npu()
            x2 = make_fp8((n, h * WORLD_SIZE + 64)).npu()  # wrong K_total
            s1 = make_e8m0((bs, (h + 63) // 64, 2)).npu()
            s2 = make_e8m0((n, ((h * WORLD_SIZE + 64) + 63) // 64, 2)).npu()

        try:
            y, a2a = cann_ops_transformer.all_to_all_quant_matmul(
                x1,
                x2,
                group,
                x1_scale=s1,
                x2_scale=s2,
                y_dtype=Y_DTYPE_BF16,
                x1_quant_mode=MX_QUANT_MODE,
                x2_quant_mode=MX_QUANT_MODE,
                group_sizes=MX_GROUP_SIZES,
                x1_scale_dtype=SCALE_DTYPE_FP8_E8M0,
                x2_scale_dtype=SCALE_DTYPE_FP8_E8M0,
                comm_mode="urma",
            )
            torch.npu.synchronize()
            queue.put((rank, "NO_ERROR", f"y.shape={y.shape}"))
        except Exception as e:
            msg = str(e)
            if "Unsupported dtype" in msg or "K dimension mismatch" in msg:
                queue.put((rank, "HOST_REJECTED", msg[:160]))
            elif "507015" in msg or "aicore" in msg:
                queue.put((rank, "DEVICE_507015", msg[:160]))
            else:
                queue.put((rank, "OTHER_ERROR", msg[:200]))

    except Exception as e:
        queue.put((rank, "SETUP_ERROR", str(e)[:160]))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.npu.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dtype", "shape"], default="dtype")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    q = mp.Manager().Queue()
    procs = [Process(target=worker, args=(q, r, args.mode)) for r in range(WORLD_SIZE)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    print("=" * 70)
    print(f"[test_invalid_input] mode={args.mode}  (expect HOST_REJECTED)")
    print("=" * 70)
    for _ in range(WORLD_SIZE):
        rank, tag, detail = q.get()
        print(f"  rank {rank}: [{tag}] {detail}")
    print("=" * 70)
