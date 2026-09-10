# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PoolKeyIndexer aclgraph 图模式 PA(PA_BBND) 布局缺陷回归用例。

场景来源(GLAM next 模型实测, 2026-08-29): aclgraph 入图(npugraph_ex 捕获/回放)
下异常报错, eager 模式正常。模型参数:

    batch=32, S1=3(TND, T1=96), S2 池化后=3, tail_k=3, n1=32, D=128,
    block_tables shape=(32, 116), block_size=288, mask_mode=3, pool_size=4,
    layout_q=tnd, layout_k=PA_BBND, topk=2048

本用例与既有 aclgraph 冒烟(test_aclgraph.py, 仅 BSND/TND 布局, 无 PA)互补,
首次将 aclgraph 捕获/回放语义与以下 PA 特有形态组合覆盖:

  1. PA_BBND block_table: shape (32, 116), 第二维 116 远大于实际所需块数
     (3 池 / 288 池每块 -> 每 batch 仅需 1 块), 其余 115 列为 -1 空槽,
     覆盖真实模型 block_table 按 KV cache 容量预分配、远超当前用量的形态;
  2. block_size=288(非 2 次幂, 16 对齐, 规格上限 1024 内);
  3. pool_size=4(非 8 倍数 —— 仅 arch35/950 支持, arch22 该值 eager 亦崩, 见
     pki_cases.py 定点用例注释);
  4. topk=2048 >> 有效池数 3(sparseCount=2048/4=512, 有效仅 3 池 ->
     覆盖 "池数 < sparseCount" 分支, 绝大多数输出为 -1);
  5. 值输入(pool_tail_k / actual_seq_q / actual_seq_k)为 NPU tensor, csrc 走
     Tensor 变体(aclnnPoolKeyIndexerTensorGetWorkspaceSize), 捕获期无阻塞 D2H
     (2026-08-27 改造后路径)。

用例结构(三测试共享同一输入构造器):
  - eager 基线: 算子结果 vs CPU golden, 固化"eager 正常"的事实基线;
  - 原生 NPUGraph 捕获(test_aclgraph.py 语义): torch_npu.npu.graph 捕获/回放;
  - TorchAir aclgraph 图模式(模型实际入图路径): torch.compile + npugraph_ex
    后端, 首调捕获 + 二调回放。

比对策略与全仓一致: indices 逐行有效索引集合比对(唯一精度门禁), values 仅
打印参考日志; 输出 dtype/shape 显式断言。

依赖: 原生 NPUGraph 用例依赖 torch_npu; TorchAir 用例依赖 torchair(缺失则该
用例 skip, 不影响其余); NPU 设备不可用则整文件 skip。
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

if not torch.npu.is_available():
    pytest.skip("NPU device is not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))

from pki_common import (  # noqa: E402
    PoolKeyIndexerModule,
    _op_tensor_args,
    assert_output_spec,
    compile_aclgraph,
    npu_device,
    pack_pa_key,
    run_golden,
    run_op_eager,
)
from result_compare import compare_indices, compare_values  # noqa: E402

from cann_ops_transformer import pool_key_indexer  # noqa: E402,F401

# ---------------------------------------------------------------------------
# 模型场景常量
# ---------------------------------------------------------------------------
BATCH = 32
S1 = 3  # 每 batch query token 数(TND, T1 = 96)
POOLS = 3  # 每 batch 池数(S2 池化后)
N1 = 32
D = 128
N2 = 1
PS = 4  # pool_size
BLOCK_SIZE = 288
MAX_BLOCKS = 116  # block_table 第二维(远大于实际所需 1 块)
TAIL = 3  # pool_tail_k(每 batch)
TOPK = 2048
MASK_MODE = 3

_DTYPES = [torch.float16, torch.bfloat16]
_DTYPE_IDS = [str(d).replace("torch.", "") for d in _DTYPES]


# ---------------------------------------------------------------------------
# 输入构造(统一 cfg + CPU 输入, 供标杆/算子/eager/aclgraph 共用)
# ---------------------------------------------------------------------------


def _build_cfg(dtype):
    return dict(
        id="aclgraph_pa_glmnext",
        layout_q="TND",
        layout_k="PA_BBND",
        dtype=dtype,
        q_lens=[S1] * BATCH,
        k_pools=[POOLS] * BATCH,
        n1=N1,
        topk=TOPK,
        pool_size=PS,
        mask_mode=MASK_MODE,
        return_value=True,
        pa_block_size=BLOCK_SIZE,
        key_non_contig=False,
        tail=[TAIL] * BATCH,
        seed=20260829,
    )


def _build_inputs(dtype):
    """构造模型场景 CPU 输入(PA 打包 + block_table 填充至 116 列空槽 -1)。

    返回 dict(key: query/pool_key/weights/pool_tail_k/actual_seq_q/
    actual_seq_k/block_table/npu_key_fn), 与 pki_common 的 run_golden /
    run_op_eager / _op_tensor_args 消费契约一致。
    """
    torch.manual_seed(_build_cfg(dtype)["seed"])

    # ---- query / weights(TND: T1 = BATCH * S1) ----
    query = torch.randn(BATCH * S1, N1, D, dtype=dtype)
    weights = torch.randn(BATCH * S1, N1, dtype=dtype)

    # ---- pool_key(PA_BBND): 每 batch 3 池, 先打包为 1 块/batch, 再填 block_table ----
    logical = torch.randn(BATCH * POOLS, N2, D, dtype=dtype)
    pool_key, bt_compact, _ = pack_pa_key(logical, [POOLS] * BATCH, BLOCK_SIZE)
    # 模型真实形态: block_table 按 KV cache 容量预分配 (B, 116), 仅首块有效
    block_table = torch.full((BATCH, MAX_BLOCKS), -1, dtype=torch.int32)
    block_table[:, : bt_compact.shape[1]] = bt_compact

    # ---- 值输入(TND 前缀和 / PA 非前缀和 / tail) ----
    actual_seq_q = torch.tensor([S1] * BATCH, dtype=torch.int64).cumsum(0)
    actual_seq_k = torch.tensor([POOLS] * BATCH, dtype=torch.int64)
    pool_tail_k = torch.full((BATCH,), TAIL, dtype=torch.int64)

    return {
        "query": query,
        "pool_key": pool_key,
        "weights": weights,
        "pool_tail_k": pool_tail_k,
        "actual_seq_q": actual_seq_q,
        "actual_seq_k": actual_seq_k,
        "block_table": block_table,
        "npu_key_fn": None,
    }


def _capture_with_warmup(fn, warmup_iters=3):
    """原生 NPUGraph 捕获流程: 侧流预热 + 捕获(口径同 test_aclgraph.py)。"""
    s = torch_npu.npu.Stream()
    s.wait_stream(torch_npu.npu.current_stream())
    with torch_npu.npu.stream(s):
        for _ in range(warmup_iters):
            fn()
    torch_npu.npu.current_stream().wait_stream(s)

    g = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(g):
        out = fn()
    return g, out


# ---------------------------------------------------------------------------
# 1. eager 基线(固化"eager 正常"的事实)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_aclgraph_pa_eager_baseline(dtype):
    """eager 单算子 vs CPU golden: 模型场景(PA_BBND + 大 block_table)基线。"""
    cfg = _build_cfg(dtype)
    inputs = _build_inputs(dtype)

    g_indices, g_values = run_golden(inputs, cfg, ref_device="cpu")
    n_indices, n_values = run_op_eager(inputs, cfg)

    assert_output_spec(n_indices, n_values, cfg)
    ok = compare_indices(g_indices, n_indices)
    assert ok, (
        f"eager(PA) baseline indices mismatch vs golden "
        f"(batch={BATCH} s1={S1} pools={POOLS} bs={BLOCK_SIZE} "
        f"topk={TOPK} ps={PS} mask={MASK_MODE})"
    )
    compare_values(g_values, n_values, dtype, label="eager-pa-values")
    print(
        f"=== ACLGRAPH-PA EAGER BASELINE PASSED ({_DTYPE_IDS[_DTYPES.index(dtype)]}) ==="
    )


# ---------------------------------------------------------------------------
# 2. 原生 NPUGraph 捕获/回放(torch_npu.npu.graph 语义, 缺陷触发点之一)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.aclgraph
def test_aclgraph_pa_native_npugraph(dtype):
    """原生 NPUGraph 捕获/回放: PA_BBND + 大 block_table 模型场景 vs CPU golden。

    值输入(pool_tail_k/actual_seq_q/k)为 NPU tensor, csrc 走 Tensor 变体
    (无阻塞 D2H, 捕获期校验跳过)。
    """
    cfg = _build_cfg(dtype)
    inputs = _build_inputs(dtype)
    dev = npu_device()

    q = inputs["query"].to(dev)
    k = inputs["pool_key"].to(dev)
    w = inputs["weights"].to(dev)
    ptk = inputs["pool_tail_k"].to(dev)
    asq = inputs["actual_seq_q"].to(dev)
    ask = inputs["actual_seq_k"].to(dev)
    bt = inputs["block_table"].to(dev)

    def run():
        return torch.ops.cann_ops_transformer.pool_key_indexer(
            q,
            k,
            w,
            ptk,
            actual_seq_q=asq,
            actual_seq_k=ask,
            block_table=bt,
            layout_q="TND",
            layout_k="PA_BBND",
            topk=TOPK,
            pool_size=PS,
            mask_mode=MASK_MODE,
            quant_mode=-1,
            return_value=True,
        )

    g, (si, sv) = _capture_with_warmup(run)
    g.replay()
    torch.npu.synchronize()
    n_indices, n_values = si.cpu(), sv.cpu()

    g_indices, g_values = run_golden(inputs, cfg, ref_device="cpu")

    assert_output_spec(n_indices, n_values, cfg)
    ok = compare_indices(g_indices, n_indices)
    assert ok, "aclgraph(native NPUGraph) PA replay indices mismatch vs golden"
    compare_values(g_values, n_values, dtype, label="aclgraph-pa-native-values")
    print(
        f"=== ACLGRAPH-PA NATIVE NPUGRAPH PASSED ({_DTYPE_IDS[_DTYPES.index(dtype)]}) ==="
    )


# ---------------------------------------------------------------------------
# 3. TorchAir aclgraph 图模式(torch.compile + npugraph_ex, 模型实际入图路径)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.aclgraph
def test_aclgraph_pa_torchair_npugraph(dtype):
    """TorchAir aclgraph(torchair npugraph_ex)捕获/回放: PA 模型场景 vs golden。

    显式 config.mode="npugraph_ex" 锁定 aclgraph 捕获/回放路径(模型实际
    入图方式), 首调捕获 + 二调回放, 回放输出与 CPU golden 比对。
    """
    torchair = pytest.importorskip("torchair")

    cfg = _build_cfg(dtype)
    inputs = _build_inputs(dtype)

    g_indices, g_values = run_golden(inputs, cfg, ref_device="cpu")

    model = compile_aclgraph(
        PoolKeyIndexerModule(
            layout_q="TND",
            layout_k="PA_BBND",
            topk=TOPK,
            pool_size=PS,
            mask_mode=MASK_MODE,
            quant_mode=-1,
            return_value=True,
        )
    )
    args = _op_tensor_args(inputs, npu_device())

    # 首调触发捕获, 二调走回放; aclgraph 输出为静态 buffer, 仅对末次输出断言
    model(*args)
    torch.npu.synchronize()
    n_indices, n_values = model(*args)
    torch.npu.synchronize()
    n_indices, n_values = n_indices.cpu(), n_values.cpu()

    assert_output_spec(n_indices, n_values, cfg)
    ok = compare_indices(g_indices, n_indices)
    assert ok, "aclgraph(torchair npugraph_ex) PA replay indices mismatch vs golden"
    compare_values(g_values, n_values, dtype, label="aclgraph-pa-torchair-values")
    print(
        f"=== ACLGRAPH-PA TORCHAIR NPUGRAPH PASSED ({_DTYPE_IDS[_DTYPES.index(dtype)]}) ==="
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
