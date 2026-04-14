#!/usr/bin/env python3
"""测试 MxA8W4 golden 逻辑"""
import sys
sys.path.insert(0, '/mnt/workspace/gitCode/sunduiyang/opencode/cann-repo/ops-transformer/gmm/grouped_matmul_finalize_routing/tests/assets')

import numpy as np
from ml_dtypes import float8_e4m3fn, float4_e2m1fn, float8_e8m0fnu, bfloat16 as np_bfloat16
import torch
from golden import grouped_matmul_finalize_routing_golden, _mxa8w4_golden

def test_mxa8w4_basic():
    """基本 MxA8W4 测试：小维度，验证计算流程能跑通"""
    m = 8
    k = 64
    n = 32
    e = 2
    batch = 4
    top_k = 2
    bsdp = 2

    np.random.seed(42)

    # x1: float8_e4m3fn, shape=(m, k)
    x1 = np.random.uniform(-2, 2, (m, k)).astype(float8_e4m3fn)

    # x2: float4_e2m1fn, viewShape=(e, n, k), transposeX2=true
    x2 = np.random.uniform(-2, 2, (e, n, k)).astype(float4_e2m1fn)

    # scale: float8_e8m0fnu, shape=(e, n, ceil(k/64), 2)
    ceil_k_64 = int(np.ceil(k / 64))
    scale = np.random.uniform(0.5, 2.0, (e, n, ceil_k_64, 2)).astype(float8_e8m0fnu)

    # pertoken_scale: float8_e8m0fnu, shape=(m, ceil(k/64), 2)
    pertoken_scale = np.random.uniform(0.5, 2.0, (m, ceil_k_64, 2)).astype(float8_e8m0fnu)

    # bias: bfloat16, shape=(e, n)
    bias = np.random.uniform(-0.1, 0.1, (e, n)).astype(np_bfloat16)

    # group_list: int64, shape=(e,), count模式
    group_list = np.array([batch, batch], dtype=np.int64)  # 每个专家4个token

    # logit: float32, shape=(m,)
    logit = np.random.uniform(0.1, 1.0, (m,)).astype(np.float32)

    # row_index: int64, shape=(m,)
    row_index = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)

    # shared_input: bfloat16, shape=(bsdp, n)
    shared_input = np.random.uniform(-0.1, 0.1, (bsdp, n)).astype(np_bfloat16)

    shared_input_weight = 1.0
    shared_input_offset = 0

    kwargs = {
        'output_dtypes': [np.float32],
    }

    print(f"=== 测试 MxA8W4 golden (m={m}, k={k}, n={n}, e={e}) ===")
    print(f"x1: dtype={x1.dtype.name}, shape={x1.shape}")
    print(f"x2: dtype={x2.dtype.name}, shape={x2.shape}")
    print(f"scale: dtype={scale.dtype.name}, shape={scale.shape}")
    print(f"pertoken_scale: dtype={pertoken_scale.dtype.name}, shape={pertoken_scale.shape}")
    print(f"bias: dtype={bias.dtype.name}, shape={bias.shape}")
    print(f"group_list: {group_list}")
    print(f"logit: shape={logit.shape}")
    print(f"row_index: shape={row_index.shape}")
    print(f"shared_input: dtype={shared_input.dtype.name}, shape={shared_input.shape}")
    print()

    # 调用 golden
    result = grouped_matmul_finalize_routing_golden(
        x1, x2, scale, bias, pertoken_scale, group_list,
        shared_input, logit, row_index, None,
        dtype=0, shared_input_weight=shared_input_weight,
        shared_input_offset=shared_input_offset,
        transpose_x=False, transpose_w=True,
        output_bs=batch, group_list_type=1,
        tuning_config=[0],
        **kwargs
    )

    print(f"结果: shape={result.shape}, dtype={result.dtype}")
    print(f"输出统计: min={result.min():.6f}, max={result.max():.6f}, mean={result.mean():.6f}")
    print(f"是否有 nan: {np.isnan(result).any()}")
    print(f"是否有 inf: {np.isinf(result).any()}")
    print()

    assert result.shape == (batch, n), f"shape 不匹配: expected {(batch, n)}, got {result.shape}"
    assert not np.isnan(result).any(), "结果包含 NaN"
    assert not np.isinf(result).any(), "结果包含 Inf"
    print("=== 测试通过 ===")


def test_mxa8w4_larger():
    """较大维度 MxA8W4 测试"""
    m = 24
    k = 128
    n = 64
    e = 4
    batch = 6
    top_k = 4
    bsdp = 3

    np.random.seed(123)

    x1 = np.random.uniform(-2, 2, (m, k)).astype(float8_e4m3fn)
    x2 = np.random.uniform(-2, 2, (e, n, k)).astype(float4_e2m1fn)

    ceil_k_64 = int(np.ceil(k / 64))
    scale = np.random.uniform(0.5, 2.0, (e, n, ceil_k_64, 2)).astype(float8_e8m0fnu)
    pertoken_scale = np.random.uniform(0.5, 2.0, (m, ceil_k_64, 2)).astype(float8_e8m0fnu)
    bias = np.random.uniform(-0.1, 0.1, (e, n)).astype(np_bfloat16)
    group_list = np.array([6, 6, 6, 6], dtype=np.int64)
    logit = np.random.uniform(0.1, 1.0, (m,)).astype(np.float32)
    row_index = np.tile(np.arange(batch), top_k).astype(np.int64)
    np.random.shuffle(row_index)
    shared_input = np.random.uniform(-0.1, 0.1, (bsdp, n)).astype(np_bfloat16)

    kwargs = {'output_dtypes': [np.float32]}

    print(f"=== 测试 MxA8W4 golden 较大维度 (m={m}, k={k}, n={n}, e={e}) ===")
    print(f"x1: dtype={x1.dtype.name}, shape={x1.shape}")
    print(f"x2: dtype={x2.dtype.name}, shape={x2.shape}")
    print(f"scale: dtype={scale.dtype.name}, shape={scale.shape}")
    print(f"pertoken_scale: dtype={pertoken_scale.dtype.name}, shape={pertoken_scale.shape}")
    print()

    result = grouped_matmul_finalize_routing_golden(
        x1, x2, scale, bias, pertoken_scale, group_list,
        shared_input, logit, row_index, None,
        dtype=0, shared_input_weight=1.0,
        shared_input_offset=0,
        transpose_x=False, transpose_w=True,
        output_bs=batch, group_list_type=1,
        tuning_config=[0],
        **kwargs
    )

    print(f"结果: shape={result.shape}, dtype={result.dtype}")
    print(f"输出统计: min={result.min():.6f}, max={result.max():.6f}, mean={result.mean():.6f}")
    print(f"是否有 nan: {np.isnan(result).any()}")
    print(f"是否有 inf: {np.isinf(result).any()}")
    print()

    assert result.shape == (batch, n), f"shape 不匹配: expected {(batch, n)}, got {result.shape}"
    assert not np.isnan(result).any(), "结果包含 NaN"
    assert not np.isinf(result).any(), "结果包含 Inf"
    print("=== 测试通过 ===")


def test_mxa8w4_no_shared_input():
    """无 shared_input 的 MxA8W4 测试"""
    m = 8
    k = 64
    n = 32
    e = 2
    batch = 4
    top_k = 2

    np.random.seed(99)

    x1 = np.random.uniform(-2, 2, (m, k)).astype(float8_e4m3fn)
    x2 = np.random.uniform(-2, 2, (e, n, k)).astype(float4_e2m1fn)

    ceil_k_64 = int(np.ceil(k / 64))
    scale = np.random.uniform(0.5, 2.0, (e, n, ceil_k_64, 2)).astype(float8_e8m0fnu)
    pertoken_scale = np.random.uniform(0.5, 2.0, (m, ceil_k_64, 2)).astype(float8_e8m0fnu)
    bias = None
    group_list = np.array([batch, batch], dtype=np.int64)
    logit = np.random.uniform(0.1, 1.0, (m,)).astype(np.float32)
    row_index = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)

    kwargs = {'output_dtypes': [np.float32]}

    print(f"=== 测试 MxA8W4 无 shared_input (m={m}, k={k}, n={n}, e={e}) ===")

    result = grouped_matmul_finalize_routing_golden(
        x1, x2, scale, bias, pertoken_scale, group_list,
        None, logit, row_index, None,
        dtype=0, shared_input_weight=1.0,
        shared_input_offset=0,
        transpose_x=False, transpose_w=True,
        output_bs=batch, group_list_type=1,
        tuning_config=[0],
        **kwargs
    )

    print(f"结果: shape={result.shape}, dtype={result.dtype}")
    print(f"输出统计: min={result.min():.6f}, max={result.max():.6f}, mean={result.mean():.6f}")
    print(f"是否有 nan: {np.isnan(result).any()}")
    print(f"是否有 inf: {np.isinf(result).any()}")
    print()

    assert result.shape == (batch, n)
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()
    print("=== 测试通过 ===")


if __name__ == '__main__':
    tests = [
        ("基本测试", test_mxa8w4_basic),
        ("较大维度测试", test_mxa8w4_larger),
        ("无shared_input测试", test_mxa8w4_no_shared_input),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"!!! 测试失败 [{name}]: {e}")
            import traceback
            traceback.print_exc()
            print()

    print(f"结果: {passed} 通过, {failed} 失败")
    sys.exit(0 if failed == 0 else 1)
