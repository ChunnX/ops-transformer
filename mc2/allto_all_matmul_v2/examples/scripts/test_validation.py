#!/usr/bin/env python3
"""Validation interception test for AlltoAllQuantMatmulV3.
Run: qrun bash -c "source conda && python test_validation.py"
"""

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import cann_ops_transformer
from ml_dtypes import float8_e4m3fn as np_fp8
from ml_dtypes import float8_e8m0fnu as np_fp8e8m0

MASTER_PORT = 50200


def init():
    dist.init_process_group("hccl", rank=0, world_size=1,
                            init_method=f"tcp://127.0.0.1:{MASTER_PORT}")


def make_group():
    return dist.new_group([0])


def make_fp8_tensor(shape, dtype):
    """Create a random fp8 tensor via numpy."""
    arr = np.random.uniform(1, 8, shape).astype(np_fp8)
    return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e4m3fn)


def make_e8m0_tensor(shape):
    arr = np.random.uniform(1, 8, shape).astype(np_fp8e8m0)
    return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e8m0fnu)


def make_valid():
    w = 2; bs = 1024; h = 4096; n = 8192
    x1 = make_fp8_tensor((bs, h), torch.float8_e4m3fn).npu()
    x2 = make_fp8_tensor((n, h*w), torch.float8_e4m3fn).npu()
    s1 = make_e8m0_tensor((bs, (h+63)//64, 2)).npu()
    s2 = make_e8m0_tensor((n, (h*w+63)//64, 2)).npu()
    return x1, x2, s1, s2


def call_op(x1, x2, g, s1, s2, **kw):
    return cann_ops_transformer.all_to_all_quant_matmul(
        x1, x2, g, x1_scale=s1, x2_scale=s2,
        y_dtype=15, x1_quant_mode=6, x2_quant_mode=6,
        group_sizes=[1, 1, 32], x1_scale_dtype=293, x2_scale_dtype=293,
        comm_mode="urma", alltoall_out_flag=False, **kw)


def check(desc, kwargs, expect_reject):
    try:
        x1, x2, s1, s2 = make_valid()
        call_op(x1, x2, g, s1, s2, **kwargs)
        print(f"  [{desc}] FAIL (no error)")
        return False
    except Exception as e:
        msg = str(e)
        if expect_reject in msg:
            print(f"  [{desc}] PASS (got: {expect_reject})")
            return True
        else:
            print(f"  [{desc}] FAIL (want '{expect_reject}' got '{msg[:120]}')")
            return False


if __name__ == "__main__":
    init()
    g = make_group()
    passed = 0; failed = 0

    print("=" * 60)
    print("AlltoAllQuantMatmulV3: Validation Interception Tests")
    print("=" * 60)

    # Baseline
    print("\n--- Baseline (should succeed) ---")
    try:
        x1, x2, s1, s2 = make_valid()
        r = call_op(x1, x2, g, s1, s2)
        print("  Baseline: OK")
    except Exception as e:
        print(f"  Baseline: FAIL ({str(e)[:100]})")
        exit(1)

    # Attr validations
    print("\n--- Attr validation ---")
    if check("comm_mode=ccu rejected", {"comm_mode": "ccu"}, "must be urma"): passed += 1
    else: failed += 1

    # Shape validations
    print("\n--- Shape validation ---")
    x1 = torch.rand(1024, 65, dtype=torch.float8_e4m3fn).npu()
    try:
        call_op(x1, x2, g, s1, s2)  # K not div 64
        print("  [K not div 64] FAIL")
        failed += 1
    except Exception as e:
        if "divisible by 64" in str(e): passed += 1; print("  [K not div 64] PASS")
        else: failed += 1; print(f"  [K not div 64] FAIL ({str(e)[:100]})")

    # K cross-check: x2.K != x1.K * world_size
    xBad = torch.rand(8192, 4096, dtype=torch.float8_e4m3fn).npu()
    sBad = torch.rand(8192, 64, 2, dtype=torch.float8_e8m0fnu).npu()
    try:
        call_op(x1, xBad, g, s1, sBad, y_dtype=15)
        print("  [K mismatch] FAIL")
        failed += 1
    except Exception as e:
        if "K mismatch" in str(e): passed += 1; print("  [K mismatch] PASS")
        else: failed += 1; print(f"  [K mismatch] FAIL ({str(e)[:100]})")

    # Scale dim not 3D
    sBad2 = torch.rand(1024, 64, dtype=torch.float8_e8m0fnu).npu()
    try:
        call_op(x1, x2, g, sBad2, s2)
        print("  [scale not 3D] FAIL")
        failed += 1
    except Exception as e:
        if "must be 3D" in str(e): passed += 1; print("  [scale not 3D] PASS")
        else: failed += 1; print(f"  [scale not 3D] FAIL ({str(e)[:100]})")

    # Bias validations
    print("\n--- Bias validation ---")
    if check("bias dtype not float", {"bias": torch.rand(8192, dtype=torch.bfloat16).npu()},
             "bias dtype must be FLOAT"): passed += 1
    else: failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    dist.destroy_process_group()