# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PoolKeyIndexer aclgraph 图模式(torch_npu NPUGraph 捕获/回放)冒烟测试。

与 GE 图模式(test_pool_key_indexer_general_ge_graph.py, torchair converter 路径)
互补, 本文件验证算子在 aclgraph 流捕获语义下的行为。csrc 按值输入设备分流
(pool_key_indexer.cpp, 2026-08-27 改造):

  1. Device tensor 取值输入(NPU, 标准用法)—— Tensor 变体路径:
     pool_tail_k / actual_seq_q / actual_seq_k 为 NPU tensor 时, csrc 走
     aclnnPoolKeyIndexerTensorGetWorkspaceSize(aclTensor*), 无阻塞 D2H:
       - eager: 消除每次调用的 D2H 同步点;
       - NPUGraph 捕获/回放可用, device 输入为图输入, 捕获后修改 buffer
         内容 replay 自动生效(与 CPU 烘焙常量方案的本质差异);
       - 代价: tiling 跳过这些输入的 host 值校验(与 GE 图模式口径一致,
         kernel 运行期从 GM 读值)。
  2. Host 取值输入(CPU tensor)—— Array 变体路径:
     全部值输入在 CPU 时保持既有 aclnnPoolKeyIndexer(aclIntArray*)链路:
       - eager: tiling 值校验保留(非法 tail/非单调前缀和仍被拒绝);
       - NPUGraph 捕获可用, host 值在捕获期固化为图常量(捕获后改 CPU
         tensor 不影响回放, 需重新捕获), 属 aclgraph 常量语义。
  3. 混布输入(NPU+CPU): csrc 统一搬至计算设备后走 Tensor 变体,
     host->device 拷贝仅 eager 可用(捕获期内禁止同步拷贝)。

历史背景: 2026-08-27 改造前, csrc 无条件对值输入做 .cpu() 阻塞 D2H,
NPU 输入在捕获期内触发 EE1006(107027/107030 "operation not permitted
when a stream is capturing"), 仅 CPU 输入可捕获; tiling 侧对
ValueDepend 输入增加 host 可见性(placement)保护后, Tensor 变体的
device 输入跳过 host 值校验而非段错误。详见 docs/build_fix_record.md。

比对策略与 GE 图模式一致: indices 逐行有效索引集合比对(唯一精度门禁),
values 仅打印参考日志; 输出 dtype/shape 显式断言。

依赖: torch_npu(缺失则整文件 skip); NPU 设备(不可用则 skip);
不依赖 torchair(NPUGraph 为 torch_npu 原生能力)。
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

if not torch.npu.is_available():
    pytest.skip("NPU device is not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))

from result_compare import compare_indices, compare_values  # noqa: E402
from pool_key_indexer_reference import pool_key_indexer_reference  # noqa: E402

# Import torch extension (registers torch.ops.cann_ops_transformer.pool_key_indexer)
from cann_ops_transformer import pool_key_indexer  # noqa: E402,F401

HEAD_DIM = 128
N2 = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_with_warmup(fn, warmup_iters=3):
    """标准 NPUGraph 捕获流程: 侧流预热 + 捕获。

    Args:
        fn: 无参 callable, 内部执行待捕获的算子调用(闭包持有静态输入)。
        warmup_iters: 侧流预热次数(torch_npu 文档推荐 3 次, 确保 kernel
            加载/allocator 就绪在捕获期外完成)。

    Returns:
        (graph, fn 的返回值): 捕获得的图与捕获期产出的静态输出 tensor。
    """
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


def _assert_output_spec(indices, values, cfg):
    """输出 dtype/shape 与接口规格的显式断言(口径同 GE 图模式测试)。"""
    ps = cfg["pool_size"]
    topk = cfg["topk"]
    out_len = topk + ps - 1
    sparse = topk // ps
    assert indices.dtype == torch.int32, f"indices dtype {indices.dtype} != int32"
    if cfg["layout_q"] == "BSND":
        exp_idx = (len(cfg["q_lens"]), cfg["q_lens"][0], out_len)
        exp_val = (len(cfg["q_lens"]), cfg["q_lens"][0], sparse)
    else:
        exp_idx = (sum(cfg["q_lens"]), out_len)
        exp_val = (sum(cfg["q_lens"]), sparse)
    assert tuple(indices.shape) == exp_idx, (
        f"indices shape {tuple(indices.shape)} != {exp_idx}"
    )
    if cfg["return_value"]:
        assert values.dtype == torch.float32, (
            f"values dtype {values.dtype} != float32"
        )
        assert tuple(values.shape) == exp_val, (
            f"values shape {tuple(values.shape)} != {exp_val}"
        )
    else:
        assert values.numel() == 0, (
            f"return_value=False 时 values 应为空, got {tuple(values.shape)}"
        )


def _golden(cfg, inputs):
    return pool_key_indexer_reference(
        inputs["query"],
        inputs["pool_key"],
        inputs["weights"],
        inputs["pool_tail_k"],
        actual_seq_q=inputs["actual_seq_q"],
        actual_seq_k=inputs["actual_seq_k"],
        block_table=inputs["block_table"],
        layout_q=cfg["layout_q"],
        layout_k=cfg["layout_k"],
        topk=cfg["topk"],
        pool_size=cfg["pool_size"],
        mask_mode=cfg["mask_mode"],
        return_value=cfg["return_value"],
    )


# ---------------------------------------------------------------------------
# Device tensor 取值输入(NPU) —— Tensor 变体路径, 捕获/回放/变值
# ---------------------------------------------------------------------------


@pytest.mark.graph
def test_aclgraph_bsnd_device_value():
    """BSND/BSND + causal + NPU pool_tail_k: 捕获/回放 vs CPU golden + 变值回放。

    验证 Tensor 变体路径: 捕获期无 D2H, device 输入为图输入 —— 捕获后原地
    修改 pool_tail_k buffer, replay 自动读到新值(图输入语义, 非烘焙常量)。
    """
    torch.manual_seed(42)
    dtype = torch.float16
    ps, topk, mask_mode = 16, 128, 3
    batch, s1, pools, n1 = 2, 64, 16, 8
    dev = "npu:0"

    query = torch.randn(batch, s1, n1, HEAD_DIM, dtype=dtype, device=dev)
    pool_key = torch.randn(batch, pools, N2, HEAD_DIM, dtype=dtype, device=dev)
    weights = torch.randn(batch, s1, n1, dtype=dtype, device=dev)
    pool_tail_k = torch.randint(0, ps, (batch,), dtype=torch.int64, device=dev)

    def run():
        return torch.ops.cann_ops_transformer.pool_key_indexer(
            query,
            pool_key,
            weights,
            pool_tail_k,
            layout_q="BSND",
            layout_k="BSND",
            topk=topk,
            pool_size=ps,
            mask_mode=mask_mode,
            quant_mode=-1,
            return_value=True,
        )

    cfg = dict(
        layout_q="BSND",
        layout_k="BSND",
        pool_size=ps,
        topk=topk,
        q_lens=[s1] * batch,
        mask_mode=mask_mode,
        return_value=True,
    )
    inputs = dict(
        query=query.cpu(),
        pool_key=pool_key.cpu(),
        weights=weights.cpu(),
        pool_tail_k=pool_tail_k.cpu(),
        actual_seq_q=None,
        actual_seq_k=None,
        block_table=None,
    )
    print(f"\n=== AclGraph-BSND-device: batch={batch} s1={s1} pools={pools} "
          f"topk={topk} ps={ps} mask={mask_mode} dtype=fp16 ===")

    g, (si, sv) = _capture_with_warmup(run)
    g.replay()
    torch.npu.synchronize()
    npu_indices = si.cpu()
    npu_values = sv.cpu()
    _assert_output_spec(npu_indices, npu_values, cfg)

    g_indices, g_values = _golden(cfg, inputs)
    ok = compare_indices(g_indices, npu_indices)
    assert ok, "aclgraph(BSND device-value) replay indices mismatch vs golden"
    compare_values(g_values, npu_values, dtype, label="aclgraph-values")

    # ---- 变值回放: 原地修改 device buffer, replay 读到新值 ----
    old_sets = [tuple(sorted(r[r >= 0].tolist())) for r in npu_indices.reshape(-1, topk + ps - 1)]
    pool_tail_k.copy_((pool_tail_k + 1) % ps)
    g.replay()
    torch.npu.synchronize()
    new_indices = si.cpu()
    inputs["pool_tail_k"] = pool_tail_k.cpu()
    g_indices2, g_values2 = _golden(cfg, inputs)
    new_sets = [tuple(sorted(r[r >= 0].tolist())) for r in new_indices.reshape(-1, topk + ps - 1)]
    assert new_sets != old_sets, "buffer 值修改后 replay 输出未变化(疑似烘焙常量语义)"
    ok2 = compare_indices(g_indices2, new_indices)
    assert ok2, "aclgraph(BSND device-value) 变值回放 indices mismatch vs 新 golden"
    compare_values(g_values2, sv.cpu(), dtype, label="aclgraph-values-mutated")

    # 二次回放确定性(同一图对象重复 replay 结果稳定)
    g.replay()
    torch.npu.synchronize()
    assert torch.equal(si.cpu(), new_indices), "second replay output changed"
    print("=== ACLGRAPH BSND DEVICE-VALUE PASSED ===")


@pytest.mark.graph
def test_aclgraph_tnd_device_value():
    """TND/TND + NPU actual_seq_q/k + pool_tail_k: 变长前缀和捕获/回放 vs golden。

    覆盖 Tensor 变体的 optional tensor 输入转换与 TND 变长路径的图输入语义。
    """
    torch.manual_seed(7)
    dtype = torch.bfloat16
    ps, topk, mask_mode = 16, 256, 0
    q_lens = [48, 32]
    k_pools = [12, 20]
    n1 = 8
    t1 = sum(q_lens)
    dev = "npu:0"

    query = torch.randn(t1, n1, HEAD_DIM, dtype=dtype, device=dev)
    pool_key = torch.randn(sum(k_pools), N2, HEAD_DIM, dtype=dtype, device=dev)
    weights = torch.randn(t1, n1, dtype=dtype, device=dev)
    pool_tail_k = torch.tensor([3, ps - 1], dtype=torch.int64, device=dev)
    actual_seq_q = torch.tensor(q_lens, dtype=torch.int64, device=dev).cumsum(0)
    actual_seq_k = torch.tensor(k_pools, dtype=torch.int64, device=dev).cumsum(0)

    def run():
        return torch.ops.cann_ops_transformer.pool_key_indexer(
            query,
            pool_key,
            weights,
            pool_tail_k,
            actual_seq_q=actual_seq_q,
            actual_seq_k=actual_seq_k,
            layout_q="TND",
            layout_k="TND",
            topk=topk,
            pool_size=ps,
            mask_mode=mask_mode,
            quant_mode=-1,
            return_value=False,
        )

    cfg = dict(
        layout_q="TND",
        layout_k="TND",
        pool_size=ps,
        topk=topk,
        q_lens=q_lens,
        mask_mode=mask_mode,
        return_value=False,
    )
    print(f"\n=== AclGraph-TND-device: q_lens={q_lens} k_pools={k_pools} "
          f"topk={topk} ps={ps} mask={mask_mode} dtype=bf16 ===")

    g, (si, sv) = _capture_with_warmup(run)
    g.replay()
    torch.npu.synchronize()
    npu_indices = si.cpu()
    _assert_output_spec(npu_indices, sv.cpu(), cfg)

    g_indices, _ = _golden(
        cfg,
        dict(
            query=query.cpu(),
            pool_key=pool_key.cpu(),
            weights=weights.cpu(),
            pool_tail_k=pool_tail_k.cpu(),
            actual_seq_q=actual_seq_q.cpu(),
            actual_seq_k=actual_seq_k.cpu(),
            block_table=None,
        ),
    )
    ok = compare_indices(g_indices, npu_indices)
    assert ok, "aclgraph(TND device-value) replay indices mismatch vs golden"

    # 变值回放: 修改前缀和(等量重分配, 保持总长不变)后 replay 生效
    actual_seq_q.copy_(torch.tensor([64, 80], dtype=torch.int64, device=dev))
    g.replay()
    torch.npu.synchronize()
    mutated = si.cpu()
    ok2 = compare_indices(
        pool_key_indexer_reference(
            query.cpu(), pool_key.cpu(), weights.cpu(), pool_tail_k.cpu(),
            actual_seq_q=actual_seq_q.cpu(), actual_seq_k=actual_seq_k.cpu(),
            block_table=None, layout_q="TND", layout_k="TND", topk=topk,
            pool_size=ps, mask_mode=mask_mode, return_value=False,
        )[0],
        mutated,
    )
    assert ok2, "aclgraph(TND device-value) 变值回放 mismatch vs 新 golden"

    g.replay()
    torch.npu.synchronize()
    assert torch.equal(si.cpu(), mutated), "second replay output changed"
    print("=== ACLGRAPH TND DEVICE-VALUE PASSED ===")


# ---------------------------------------------------------------------------
# Host 取值输入(CPU tensor) —— Array 变体路径, 值烘焙为图常量
# ---------------------------------------------------------------------------


@pytest.mark.graph
def test_aclgraph_bsnd_host_value():
    """BSND/BSND + causal: CPU pool_tail_k(Array 变体)捕获/回放 vs CPU golden。

    CPU 值输入走 aclIntArray 链路: tiling 值校验保留, 捕获期无 D2H,
    host 值固化为图常量(改值需重新捕获)。
    """
    torch.manual_seed(42)
    dtype = torch.float16
    ps, topk, mask_mode = 16, 128, 3
    batch, s1, pools, n1 = 2, 64, 16, 8
    dev = "npu:0"

    query = torch.randn(batch, s1, n1, HEAD_DIM, dtype=dtype, device=dev)
    pool_key = torch.randn(batch, pools, N2, HEAD_DIM, dtype=dtype, device=dev)
    weights = torch.randn(batch, s1, n1, dtype=dtype, device=dev)
    pool_tail_k_cpu = torch.randint(0, ps, (batch,), dtype=torch.int64)

    def run():
        return torch.ops.cann_ops_transformer.pool_key_indexer(
            query,
            pool_key,
            weights,
            pool_tail_k_cpu,
            layout_q="BSND",
            layout_k="BSND",
            topk=topk,
            pool_size=ps,
            mask_mode=mask_mode,
            quant_mode=-1,
            return_value=True,
        )

    cfg = dict(
        layout_q="BSND",
        layout_k="BSND",
        pool_size=ps,
        topk=topk,
        q_lens=[s1] * batch,
        mask_mode=mask_mode,
        return_value=True,
    )
    print(f"\n=== AclGraph-BSND-host: batch={batch} s1={s1} pools={pools} "
          f"topk={topk} ps={ps} mask={mask_mode} dtype=fp16 ===")

    g, (si, sv) = _capture_with_warmup(run)
    g.replay()
    torch.npu.synchronize()
    npu_indices = si.cpu()
    npu_values = sv.cpu()
    _assert_output_spec(npu_indices, npu_values, cfg)

    g_indices, g_values = _golden(
        cfg,
        dict(
            query=query.cpu(),
            pool_key=pool_key.cpu(),
            weights=weights.cpu(),
            pool_tail_k=pool_tail_k_cpu,
            actual_seq_q=None,
            actual_seq_k=None,
            block_table=None,
        ),
    )
    ok = compare_indices(g_indices, npu_indices)
    assert ok, "aclgraph(BSND host-value) replay indices mismatch vs golden"
    compare_values(g_values, npu_values, dtype, label="aclgraph-values")

    # 二次回放确定性(同一图对象重复 replay 结果稳定)
    g.replay()
    torch.npu.synchronize()
    assert torch.equal(si.cpu(), npu_indices), "second replay output changed"
    print("=== ACLGRAPH BSND HOST-VALUE PASSED ===")


@pytest.mark.graph
def test_aclgraph_tnd_host_value():
    """TND/TND + non-causal: CPU 值输入(Array 变体)捕获/回放 vs CPU golden。

    actual_seq_q/actual_seq_k(前缀和)与 pool_tail_k 均为 CPU int64 tensor,
    覆盖变长路径的 host 取值输入; return_value=False 覆盖空 values 输出。
    """
    torch.manual_seed(7)
    dtype = torch.bfloat16
    ps, topk, mask_mode = 16, 256, 0
    q_lens = [48, 32]
    k_pools = [12, 20]
    n1 = 8
    t1 = sum(q_lens)
    dev = "npu:0"

    query = torch.randn(t1, n1, HEAD_DIM, dtype=dtype, device=dev)
    pool_key = torch.randn(sum(k_pools), N2, HEAD_DIM, dtype=dtype, device=dev)
    weights = torch.randn(t1, n1, dtype=dtype, device=dev)
    pool_tail_k_cpu = torch.tensor([3, ps - 1], dtype=torch.int64)
    actual_seq_q_cpu = torch.tensor(q_lens, dtype=torch.int64).cumsum(0)
    actual_seq_k_cpu = torch.tensor(k_pools, dtype=torch.int64).cumsum(0)

    def run():
        return torch.ops.cann_ops_transformer.pool_key_indexer(
            query,
            pool_key,
            weights,
            pool_tail_k_cpu,
            actual_seq_q=actual_seq_q_cpu,
            actual_seq_k=actual_seq_k_cpu,
            layout_q="TND",
            layout_k="TND",
            topk=topk,
            pool_size=ps,
            mask_mode=mask_mode,
            quant_mode=-1,
            return_value=False,
        )

    cfg = dict(
        layout_q="TND",
        layout_k="TND",
        pool_size=ps,
        topk=topk,
        q_lens=q_lens,
        mask_mode=mask_mode,
        return_value=False,
    )
    print(f"\n=== AclGraph-TND-host: q_lens={q_lens} k_pools={k_pools} "
          f"topk={topk} ps={ps} mask={mask_mode} dtype=bf16 ===")

    g, (si, sv) = _capture_with_warmup(run)
    g.replay()
    torch.npu.synchronize()
    npu_indices = si.cpu()
    _assert_output_spec(npu_indices, sv.cpu(), cfg)

    g_indices, _ = _golden(
        cfg,
        dict(
            query=query.cpu(),
            pool_key=pool_key.cpu(),
            weights=weights.cpu(),
            pool_tail_k=pool_tail_k_cpu,
            actual_seq_q=actual_seq_q_cpu,
            actual_seq_k=actual_seq_k_cpu,
            block_table=None,
        ),
    )
    ok = compare_indices(g_indices, npu_indices)
    assert ok, "aclgraph(TND host-value) replay indices mismatch vs golden"

    g.replay()
    torch.npu.synchronize()
    assert torch.equal(si.cpu(), npu_indices), "second replay output changed"
    print("=== ACLGRAPH TND HOST-VALUE PASSED ===")


# ---------------------------------------------------------------------------
# 值输入路由行为固化
# ---------------------------------------------------------------------------


def test_value_input_host_validation_routing():
    """csrc 值输入分流行为固化: CPU 输入保留 tiling 值校验(Array 变体)。

    NPU 输入(Tensor 变体)跳过 host 值校验的代价与 GE 图模式一致, 不在此
    断言(非法值直接进 kernel, 行为未定义); CPU 输入非法 tail 应仍被 tiling
    精确拒绝, 证明 Array 变体路径未被 Tensor 变体改造破坏。
    """
    torch.manual_seed(42)
    dev = "npu:0"
    ps, topk = 16, 128
    q = torch.randn(1, 64, 8, HEAD_DIM, dtype=torch.float16, device=dev)
    k = torch.randn(1, 16, N2, HEAD_DIM, dtype=torch.float16, device=dev)
    w = torch.randn(1, 64, 8, dtype=torch.float16, device=dev)

    # CPU 非法 tail(= pool_size, 越界) -> tiling 值校验拒绝
    with pytest.raises(Exception, match="pool_tail_k"):
        torch.ops.cann_ops_transformer.pool_key_indexer(
            q, k, w, torch.tensor([ps], dtype=torch.int64),
            layout_q="BSND", layout_k="BSND",
            topk=topk, pool_size=ps, mask_mode=3, quant_mode=-1,
            return_value=True,
        )
    print("=== VALUE-INPUT ROUTING (CPU validation preserved) PASSED ===")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
