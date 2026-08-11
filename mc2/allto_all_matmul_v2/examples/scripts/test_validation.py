#!/usr/bin/env python3
"""Validation interception test for AlltoAllQuantMatmulV3.
Tests that invalid inputs are properly rejected.
Run: qrun bash -c "source conda && python test_validation.py"
"""

import argparse
import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Manager
import cann_ops_transformer
from ml_dtypes import float8_e4m3fn as np_fp8
from ml_dtypes import float8_e5m2 as np_fp8e5
from ml_dtypes import float8_e8m0fnu as np_fp8e8m0

MASTER_IP = "127.0.0.1"
MASTER_PORT = 50205
WORLD_SIZE = 2


def make_fp8(shape):
    arr = np.random.uniform(1, 8, shape).astype(np_fp8)
    return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e4m3fn)


def make_e8m0(shape):
    arr = np.random.uniform(1, 8, shape).astype(np_fp8e8m0)
    return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e8m0fnu)


def make_valid():
    bs = 1024; h = 4096; n = 8192; w = WORLD_SIZE
    x1 = make_fp8((bs, h)).npu()
    x2 = make_fp8((n, h*w)).npu()
    s1 = make_e8m0((bs, (h+63)//64, 2)).npu()
    s2 = make_e8m0((n, (h*w+63)//64, 2)).npu()
    return x1, x2, s1, s2


def call_op(x1, x2, g, s1, s2, **kw):
    return cann_ops_transformer.all_to_all_quant_matmul(
        x1, x2, g, x1_scale=s1, x2_scale=s2,
        y_dtype=15, x1_quant_mode=6, x2_quant_mode=6,
        group_sizes=[1, 1, 32], x1_scale_dtype=293, x2_scale_dtype=293,
        comm_mode="urma", alltoall_out_flag=False, **kw)


def check(desc, kwargs, expect_reject, g, x1, x2, s1, s2, results):
    try:
        call_op(x1, x2, g, s1, s2, **kwargs)
        results.put((desc, False, "no error"))
    except Exception as e:
        msg = str(e)
        if expect_reject in msg:
            results.put((desc, True, expect_reject))
        else:
            results.put((desc, False, msg[:150]))


def worker(queue, rank):
    try:
        npu_device = rank % 2
        torch_npu.npu.set_device(npu_device)
        dist.init_process_group("hccl", rank=rank, world_size=WORLD_SIZE,
                                init_method=f"tcp://{MASTER_IP}:{MASTER_PORT}")
        g = dist.new_group(list(range(WORLD_SIZE)))
        x1, x2, s1, s2 = make_valid()

        if rank == 0:
            # Baseline
            try:
                r = call_op(x1, x2, g, s1, s2)
                queue.put(("baseline", True, "OK"))
            except Exception as e:
                queue.put(("baseline", False, str(e)[:150]))

            # Validation tests
            tests = [
                ("comm_mode=ccu", {"comm_mode": "ccu"}, "must be urma"),
                ("bias dtype bf16", {"bias": torch.rand(8192, dtype=torch.bfloat16).npu()}, "bias dtype must be FLOAT"),
            ]
            for desc, kw, expect in tests:
                check(desc, kw, expect, g, x1, x2, s1, s2, queue)
    except Exception as e:
        queue.put(("init", False, str(e)[:150]))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.npu.empty_cache()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    q = mp.Manager().Queue()
    procs = [Process(target=worker, args=(q, r)) for r in range(WORLD_SIZE)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    passed = 0; failed = 0
    results = []
    while not q.empty():
        name, ok, detail = q.get()
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        results.append(f"  [{name}] {status} ({detail[:80]})")

    print("=" * 60)
    print("AlltoAllQuantMatmulV3: Validation Tests")
    print("=" * 60)
    for r in results:
        print(r)
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)