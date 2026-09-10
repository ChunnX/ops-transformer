# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PoolKeyIndexer 冒烟测试套件(套件1): torch 接口可用性验证。

覆盖三种调用方式(用例代码即文档示例, 与
torch_extension/cann_ops_transformer/docs/zh/pool_key_indexer.md 的
"调用示例"章节保持一致):

  1. eager 单算子模式: cann_ops_transformer.pool_key_indexer Python 封装
     (BSND+因果掩码+池化 / TND+无掩码+pool_size=1 两个示例形态);
  2. eager torch.ops 路径: torch.ops.cann_ops_transformer.pool_key_indexer
     直调(亦为 GE 图模式 converter 的拦截入口);
  3. TorchAir GE 图模式: torch.compile + torchair NPU 后端(fullgraph,
     显式 mode="max-autotune" 锁定 GE/AscendIR converter 转换路径);
  4. TorchAir aclgraph 图模式: torch.compile + torchair npugraph_ex 后端
     (显式 mode="npugraph_ex" 锁定捕获/回放路径, 首调捕获/再调回放;
     原生 NPUGraph 捕获语义由 test_aclgraph.py 独立覆盖)。

判定标准: 调用成功、输出 shape/dtype 符合接口规格、含有效索引、同步无异常。
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

if not torch.npu.is_available():
    pytest.skip("NPU device is not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))

from cann_ops_transformer import pool_key_indexer  # noqa: E402

DEVICE = f"npu:{os.environ.get('PKI_DEVICE_ID', '0')}"


# ---------------------------------------------------------------------------
# 1. eager 单算子模式: BSND + 因果掩码 + 池化(文档示例一)
# ---------------------------------------------------------------------------


def test_smoke_eager_bsnd_causal_pooling():
    """BSND + rightDownCausal + pool_size=16, return_value=True。"""
    torch.manual_seed(42)
    B = 2
    S1 = 64
    S2 = 256  # pool级别Key序列长度
    N1 = 8
    N2 = 1
    D = 128
    topk = 128
    pool_size = 16
    mask_mode = 3  # 因果掩码
    return_value = True

    # 构造输入
    query = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
    pool_key = torch.randn(B, S2, N2, D, dtype=torch.float16).npu()
    weights = torch.randn(B, S1, N1, dtype=torch.float16).npu()
    pool_tail_k = torch.tensor(
        [0, 8], dtype=torch.int64
    ).npu()  # batch0尾块满, batch1尾块8个token

    # 执行pool_key_indexer
    sparse_indices, sparse_values = pool_key_indexer(
        query,
        pool_key,
        weights,
        pool_tail_k,
        layout_q="BSND",
        layout_k="BSND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=-1,
        return_value=return_value,
    )
    torch.npu.synchronize()

    out_len = topk + pool_size - 1
    sparse = topk // pool_size
    assert sparse_indices.dtype == torch.int32
    assert tuple(sparse_indices.shape) == (B, S1, out_len)
    assert sparse_values.dtype == torch.float32
    assert tuple(sparse_values.shape) == (B, S1, sparse)
    assert (sparse_indices >= 0).sum() > 0, "indices 不应全为无效值"
    print(f"sparse_indices shape: {tuple(sparse_indices.shape)}")
    print(f"sparse_values shape: {tuple(sparse_values.shape)}")


# ---------------------------------------------------------------------------
# 2. eager 单算子模式: TND + 无掩码 + 退化场景 pool_size=1(文档示例二)
# ---------------------------------------------------------------------------


def test_smoke_eager_tnd_pool_size1():
    """TND + 无掩码 + pool_size=1(退化为 lightning_indexer 行为)。"""
    torch.manual_seed(7)
    q_lens = [32, 48]
    k_lens = [128, 192]
    N1 = 8
    N2 = 1
    D = 128
    topk = 64
    pool_size = 1
    mask_mode = 0

    total_q = sum(q_lens)
    total_k = sum(k_lens)

    query = torch.randn(total_q, N1, D, dtype=torch.float16).npu()
    pool_key = torch.randn(total_k, N2, D, dtype=torch.float16).npu()
    weights = torch.randn(total_q, N1, dtype=torch.float16).npu()
    pool_tail_k = torch.tensor([0, 0], dtype=torch.int64).npu()
    # TND前缀和
    actual_seq_q = torch.tensor(
        [q_lens[0], q_lens[0] + q_lens[1]], dtype=torch.int64
    ).npu()
    actual_seq_k = torch.tensor(
        [k_lens[0], k_lens[0] + k_lens[1]], dtype=torch.int64
    ).npu()

    sparse_indices, sparse_values = pool_key_indexer(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        layout_q="TND",
        layout_k="TND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=-1,
        return_value=False,
    )
    torch.npu.synchronize()

    out_len = topk + pool_size - 1
    assert sparse_indices.dtype == torch.int32
    assert tuple(sparse_indices.shape) == (total_q, out_len)
    assert sparse_values.numel() == 0, "return_value=False 时 values 为空tensor"
    assert (sparse_indices >= 0).sum() > 0, "indices 不应全为无效值"
    print(f"sparse_indices shape: {tuple(sparse_indices.shape)}")


# ---------------------------------------------------------------------------
# 3. eager torch.ops 路径(GE 图模式 converter 的拦截入口)
# ---------------------------------------------------------------------------


def test_smoke_torch_ops_eager():
    """torch.ops.cann_ops_transformer.pool_key_indexer 直调(BSND + 因果 + 池化)。"""
    torch.manual_seed(11)
    B, S1, S2, N1, N2, D = 1, 64, 256, 8, 1, 128
    topk, pool_size, mask_mode = 128, 16, 3

    query = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
    pool_key = torch.randn(B, S2, N2, D, dtype=torch.float16).npu()
    weights = torch.randn(B, S1, N1, dtype=torch.float16).npu()
    pool_tail_k = torch.tensor([0], dtype=torch.int64).npu()

    sparse_indices, sparse_values = torch.ops.cann_ops_transformer.pool_key_indexer(
        query,
        pool_key,
        weights,
        pool_tail_k,
        layout_q="BSND",
        layout_k="BSND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=-1,
        return_value=True,
    )
    torch.npu.synchronize()

    out_len = topk + pool_size - 1
    sparse = topk // pool_size
    assert sparse_indices.dtype == torch.int32
    assert tuple(sparse_indices.shape) == (B, S1, out_len)
    assert sparse_values.dtype == torch.float32
    assert tuple(sparse_values.shape) == (B, S1, sparse)
    assert (sparse_indices >= 0).sum() > 0, "indices 不应全为无效值"


# ---------------------------------------------------------------------------
# 4. TorchAir GE 图模式(文档示例三)
# ---------------------------------------------------------------------------


class PoolKeyIndexerNetwork(torch.nn.Module):
    """pool_key_indexer 图模式封装(经 torch.ops 路径被 converter 拦截)。"""

    def forward(self, query, pool_key, weights, pool_tail_k):
        return torch.ops.cann_ops_transformer.pool_key_indexer(
            query,
            pool_key,
            weights,
            pool_tail_k,
            layout_q="BSND",
            layout_k="BSND",
            topk=128,
            pool_size=16,
            mask_mode=3,
            quant_mode=-1,
            return_value=True,
        )


def _install_torchair_path_probe():
    """安装 torchair 编译路径探针, 记录图编译期实例化的 concrete graph 类型。

    torchair 按 compiler_config.mode 路由编译后端:
      - "max-autotune"                 -> GeConcreteGraph(GE/AscendIR converter 路径);
      - "reduce-overhead"/"npugraph_ex" -> AclConcreteGraph(aclgraph 捕获回放路径)。
    探针用于断言用例实际走 GE 入图, 防止 CompilerConfig 默认 mode 变化导致
    用例静默漂移为 aclgraph。torchair 内部路径调整导致探针不可安装时返回
    None(仅跳过路径断言, 不影响功能验证)。
    """
    try:
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph
    except ImportError:
        return None

    record = {"ge": 0, "acl": 0, "uninstall": None}
    ge_init, acl_init = GeConcreteGraph.__init__, AclConcreteGraph.__init__

    def _patched_ge_init(self, *args, **kwargs):
        record["ge"] += 1
        return ge_init(self, *args, **kwargs)

    def _patched_acl_init(self, *args, **kwargs):
        record["acl"] += 1
        return acl_init(self, *args, **kwargs)

    def _uninstall():
        GeConcreteGraph.__init__ = ge_init
        AclConcreteGraph.__init__ = acl_init

    GeConcreteGraph.__init__ = _patched_ge_init
    AclConcreteGraph.__init__ = _patched_acl_init
    record["uninstall"] = _uninstall
    return record


def test_smoke_torchair_ge_graph():
    """TorchAir GE 图模式: torch.compile + torchair NPU 后端(fullgraph)。

    显式指定 config.mode = "max-autotune" 锁定 GE 入图(AscendIR converter
    转换路径, tiling 值校验跳过 + kernel 运行期从 GM 读值), 不依赖
    CompilerConfig 默认值(默认值随 torchair 版本演进可能漂移为
    reduce-overhead/npugraph_ex 即 aclgraph 捕获回放路径); aclgraph
    (捕获回放语义)由 test_aclgraph.py 独立覆盖。用例内安装路径探针,
    断言编译期实例化的是 GeConcreteGraph 而非 AclConcreteGraph,
    保证用例名实相符。
    """
    torchair = pytest.importorskip("torchair")

    torch.manual_seed(23)
    B = 1
    S1 = 64
    S2 = 256
    N1 = 8
    N2 = 1
    D = 128

    query = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
    pool_key = torch.randn(B, S2, N2, D, dtype=torch.float16).npu()
    weights = torch.randn(B, S1, N1, dtype=torch.float16).npu()
    pool_tail_k = torch.tensor([0], dtype=torch.int64).npu()

    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    # 显式锁定 GE 图模式: max-autotune -> GeConcreteGraph(converter 转换路径);
    # reduce-overhead/npugraph_ex -> AclConcreteGraph(aclgraph 捕获回放路径)。
    config.mode = "max-autotune"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(
        PoolKeyIndexerNetwork(), fullgraph=True, backend=npu_backend, dynamic=False
    )

    # torch.compile 为惰性编译, 首次调用触发图编译; 探针覆盖编译+执行窗口。
    probe = _install_torchair_path_probe()
    try:
        sparse_indices, sparse_values = npu_mode(query, pool_key, weights, pool_tail_k)
        torch.npu.synchronize()
    finally:
        if probe is not None:
            probe["uninstall"]()

    if probe is not None:
        assert probe["ge"] > 0, "torchair 未实例化 GeConcreteGraph, GE 入图路径未生效"
        assert probe["acl"] == 0, (
            "torchair 实例化了 AclConcreteGraph, 用例漂移为 aclgraph 路径"
        )

    topk, pool_size = 128, 16
    out_len = topk + pool_size - 1
    sparse = topk // pool_size
    assert sparse_indices.dtype == torch.int32
    assert tuple(sparse_indices.shape) == (B, S1, out_len)
    assert sparse_values.dtype == torch.float32
    assert tuple(sparse_values.shape) == (B, S1, sparse)
    assert (sparse_indices >= 0).sum() > 0, "indices 不应全为无效值"
    print(f"sparse_indices shape: {tuple(sparse_indices.shape)}")


# ---------------------------------------------------------------------------
# 5. TorchAir aclgraph 图模式(捕获/回放)
# ---------------------------------------------------------------------------


def test_smoke_torchair_aclgraph():
    """TorchAir aclgraph 图模式: torch.compile + torchair npugraph_ex 后端。

    显式指定 config.mode = "npugraph_ex" 锁定 aclgraph 入图
    (AclConcreteGraph 捕获/回放路径, 等价于已废弃的 reduce-overhead)。
    值输入(pool_tail_k 等)为 NPU tensor 时 csrc 走 aclnn Tensor 变体,
    捕获期无阻塞 D2H, device 输入为图输入, 回放期读取 buffer 当前内容
    (图输入语义, 非烘焙常量)。首次调用触发捕获, 二次调用走回放,
    回放输出仍需满足接口规格。路径探针断言编译期实例化的是
    AclConcreteGraph 而非 GeConcreteGraph, 保证用例名实相符。
    原生 NPUGraph 捕获语义(侧流预热 + torch_npu.npu.graph)由
    test_aclgraph.py 独立覆盖。
    """
    torchair = pytest.importorskip("torchair")

    torch.manual_seed(31)
    B = 1
    S1 = 64
    S2 = 256
    N1 = 8
    N2 = 1
    D = 128

    query = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
    pool_key = torch.randn(B, S2, N2, D, dtype=torch.float16).npu()
    weights = torch.randn(B, S1, N1, dtype=torch.float16).npu()
    pool_tail_k = torch.tensor([0], dtype=torch.int64).npu()

    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    # 显式锁定 aclgraph 图模式: npugraph_ex/reduce-overhead(已废弃) ->
    # AclConcreteGraph(捕获/回放路径); max-autotune -> GE converter 路径。
    config.mode = "npugraph_ex"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(
        PoolKeyIndexerNetwork(), fullgraph=True, backend=npu_backend, dynamic=False
    )

    # 首次调用触发捕获, 其后调用走回放; 探针覆盖捕获+回放窗口。
    # aclgraph 输出为静态 buffer, 回放复写, 仅对末次输出断言。
    probe = _install_torchair_path_probe()
    try:
        npu_mode(query, pool_key, weights, pool_tail_k)
        torch.npu.synchronize()
        sparse_indices, sparse_values = npu_mode(query, pool_key, weights, pool_tail_k)
        torch.npu.synchronize()
    finally:
        if probe is not None:
            probe["uninstall"]()

    if probe is not None:
        assert probe["acl"] > 0, (
            "torchair 未实例化 AclConcreteGraph, aclgraph 入图路径未生效"
        )
        assert probe["ge"] == 0, "torchair 实例化了 GeConcreteGraph, 用例漂移为 GE 路径"

    topk, pool_size = 128, 16
    out_len = topk + pool_size - 1
    sparse = topk // pool_size
    assert sparse_indices.dtype == torch.int32
    assert tuple(sparse_indices.shape) == (B, S1, out_len)
    assert sparse_values.dtype == torch.float32
    assert tuple(sparse_values.shape) == (B, S1, sparse)
    assert (sparse_indices >= 0).sum() > 0, "indices 不应全为无效值"
    print(f"sparse_indices shape: {tuple(sparse_indices.shape)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
