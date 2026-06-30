#!/usr/bin/env python3
"""
测试: T.copy / T.tile.atomic_add 使用动态索引 sparse_indices 时的报错
模拟 gather 和 scatter 两种场景
"""
import torch
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

# ==============================================================================
# 测试 1: T.copy 用动态索引做 gather
# ==============================================================================
@tilelang.jit(pass_configs=pass_configs)
def test_copy_dynamic_gather(B, S1, K, DQuery, input_dtype, calc_dtype):
    """模拟: gather_key[b,s,i,:] = key[b, sparse_indices[b,s,0,i], :]"""

    @T.prim_func
    def main(
        Key: T.Tensor((B, S1, 1, DQuery), input_dtype),
        SparseIndices: T.Tensor((B, S1, 1, K), "int32"),
        GatherKey: T.Tensor((B, S1, K, DQuery), input_dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                key_ub = T.alloc_ub((1, DQuery), input_dtype)
                for s1_id in T.serial(S1):
                    for k_id in T.serial(K):
                        idx = SparseIndices[0, s1_id, 0, k_id]  # 运行时变量
                        if idx >= 0:
                            T.copy(Key[0, idx, 0, 0:DQuery], key_ub)      # ❌ 动态索引
                            T.copy(key_ub, GatherKey[0, s1_id, k_id, 0:DQuery])

    return main


# ==============================================================================
# 测试 2: T.tile.atomic_add 用动态索引做 scatter
# ==============================================================================
@tilelang.jit(pass_configs=pass_configs)
def test_atomic_add_dynamic_scatter(B, S1, S2, K, DQI, calc_dtype):
    """模拟: d_key_index[b, idx, 0, :] += scatter_add[b, s1_id, k_id, :]"""

    @T.prim_func
    def main(
        ScatterAdd: T.Tensor((B, S1, K, DQI), calc_dtype),
        SparseIndices: T.Tensor((B, S1, 1, K), "int32"),
        DKeyIndex: T.Tensor((B, S2, 1, DQI), calc_dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                val_ub = T.alloc_ub((1, DQI), calc_dtype)
                for s1_id in T.serial(S1):
                    for k_id in T.serial(K):
                        idx = SparseIndices[0, s1_id, 0, k_id]  # 运行时变量
                        if idx >= 0:
                            T.copy(ScatterAdd[0, s1_id, k_id, 0:DQI], val_ub)
                            T.tile.atomic_add(DKeyIndex[0, idx, 0, 0:DQI], val_ub)  # ❌

    return main


# ==============================================================================
# 测试 3: T.copy 用动态索引做 scatter
# ==============================================================================
@tilelang.jit(pass_configs=pass_configs)
def test_copy_dynamic_scatter(B, S1, S2, K, DQI, calc_dtype):
    """与测试2相同，但用 T.copy 替代 T.tile.atomic_add"""

    @T.prim_func
    def main(
        ScatterAdd: T.Tensor((B, S1, K, DQI), calc_dtype),
        SparseIndices: T.Tensor((B, S1, 1, K), "int32"),
        DKeyIndex: T.Tensor((B, S2, 1, DQI), calc_dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                val_ub = T.alloc_ub((1, DQI), calc_dtype)
                for s1_id in T.serial(S1):
                    for k_id in T.serial(K):
                        idx = SparseIndices[0, s1_id, 0, k_id]  # 运行时变量
                        if idx >= 0:
                            T.copy(ScatterAdd[0, s1_id, k_id, 0:DQI], val_ub)
                            T.copy(val_ub, DKeyIndex[0, idx, 0, 0:DQI])  # ❌ 动态索引

    return main


if __name__ == "__main__":
    B, S1, S2, K, DQuery, DQI = 1, 1, 2048, 2048, 512, 128
    input_dtype = "float16"
    calc_dtype = "float"

    print("=" * 70)
    print("测试 1: T.copy 动态索引做 gather")
    print("=" * 70)
    try:
        func1 = test_copy_dynamic_gather(B, S1, K, DQuery, input_dtype, calc_dtype)
        print("编译成功 (未触发预期错误)")
    except Exception as e:
        print(f"编译失败: {e}")

    print()
    print("=" * 70)
    print("测试 2: T.tile.atomic_add 动态索引做 scatter")
    print("=" * 70)
    try:
        func2 = test_atomic_add_dynamic_scatter(B, S1, S2, K, DQI, calc_dtype)
        print("编译成功 (未触发预期错误)")
    except Exception as e:
        print(f"编译失败: {e}")

    print()
    print("=" * 70)
    print("测试 3: T.copy 动态索引做 scatter")
    print("=" * 70)
    try:
        func3 = test_copy_dynamic_scatter(B, S1, S2, K, DQI, calc_dtype)
        print("编译成功 (未触发预期错误)")
    except Exception as e:
        print(f"编译失败: {e}")
