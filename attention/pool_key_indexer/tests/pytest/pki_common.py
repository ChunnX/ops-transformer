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
PoolKeyIndexer 测试共享工具(冒烟/精度/GE 图/性能套件公用)。

提供:
  - 设备选择: PKI_DEVICE_ID 环境变量(默认 0), 避免多用户环境设备冲突;
  - 用例输入构造: build_case_inputs(cfg), 统一 cfg dict 格式
    (layout_q/layout_k/dtype/q_lens/k_pools/n1/topk/pool_size/mask_mode/
     return_value/pa_block_size/key_non_contig/tail/seed);
  - PA 物理块打包: pack_pa_key(一致性 block_table 逆映射, 0 池 batch 不占块);
  - 运行封装: run_op_eager / run_op_ge_graph(算子) 与 run_golden(统一标杆,
    ref_device 可选 cpu/npu 灵活切换);
  - 输出规格断言: assert_output_spec(dtype/shape, 接口文档口径)。
"""

import os

import torch
import torch_npu  # noqa: F401  (NPU 后端注册副作用)

from cann_ops_transformer import pool_key_indexer  # noqa: F401  (注册 torch.ops)

from pool_key_indexer_reference import pool_key_indexer_reference

HEAD_DIM = 128
N2 = 1


# ---------------------------------------------------------------------------
# 设备选择
# ---------------------------------------------------------------------------


def get_device_id():
    """测试设备号: 环境变量 PKI_DEVICE_ID(默认 0)。"""
    return int(os.environ.get("PKI_DEVICE_ID", "0"))


def npu_device():
    return f"npu:{get_device_id()}"


# ---------------------------------------------------------------------------
# PA 物理块打包(一致性 block_table)
# ---------------------------------------------------------------------------


def pack_pa_key(pool_key_rows, k_pool_lens, block_size):
    """逻辑 (sum_pools, N2, D) 池数据打包为 PA 物理块布局。

    与 LIV2 语义一致: PA key 块内每行 = 1 个池(key 不做 token 展开)。
    每 batch 的池独立连续填块, 尾块不足补零; 物理块随机排布一次,
    block_table 由同一置换的逆映射反推, 保证 bt[b][i] 指向的物理块真实
    存放 batch b 第 i 组池(算子与标杆共用同一份物理布局)。
    0 池 batch 不占任何物理块。返回 (key_pa, block_table, block_num)。
    """
    blocks_per_batch = [(pl + block_size - 1) // block_size for pl in k_pool_lens]
    block_num = sum(blocks_per_batch)
    n2, d = pool_key_rows.shape[1], pool_key_rows.shape[2]
    blocks = torch.zeros(block_num, block_size, n2, d, dtype=pool_key_rows.dtype)
    pool_start = blk_start = 0
    for pools, nb in zip(k_pool_lens, blocks_per_batch):
        if nb == 0:
            continue
        seg = pool_key_rows[pool_start : pool_start + pools]
        pad = nb * block_size - pools
        if pad:
            seg = torch.cat([seg, torch.zeros(pad, n2, d, dtype=seg.dtype)], dim=0)
        blocks[blk_start : blk_start + nb] = seg.view(nb, block_size, n2, d)
        pool_start += pools
        blk_start += nb
    perm = torch.randperm(block_num)  # 物理块 j 存放旧块 perm[j]
    inv = torch.empty_like(perm)  # 旧块 o 放到物理块 inv[o]
    inv[perm] = torch.arange(block_num)
    bt = torch.full((len(k_pool_lens), max(blocks_per_batch)), -1, dtype=torch.int32)
    cur = 0
    for b, nb in enumerate(blocks_per_batch):
        for i in range(nb):
            bt[b, i] = inv[cur]
            cur += 1
    return blocks[perm].contiguous(), bt, block_num


def make_pa_key_non_contig(key_pa, block_size, block_num, dtype):
    """构造 PA key 0 轴非连续输入(keyStride0 运行时 stride 分支)。

    物理块嵌入 2 倍缓冲的偶数行, 视图 stride[0] = 2 * block_size * N2 * D。
    返回 (cpu_view, npu_builder):
      - cpu_view: 供标杆直接消费(torch 按 stride 索引);
      - npu_builder: NPU 侧先搬大缓冲再切片 —— 直接对非稠密视图 .to(npu)
        会被拍平为连续, 丢失 stride 信息, 无法覆盖 kernel 非连续寻址。
    """
    big = torch.zeros(2 * block_num, block_size, N2, HEAD_DIM, dtype=dtype)
    big[0::2] = key_pa
    view = big[::2]

    def npu_builder(dev, _big=big):
        return _big.to(dev)[::2]

    return view, npu_builder


# ---------------------------------------------------------------------------
# 用例输入构造(统一 cfg dict 格式)
# ---------------------------------------------------------------------------


def build_case_inputs(cfg):
    """根据 case 配置构造全部 CPU 输入 + 可选 NPU key 构造器。

    cfg 字段:
      layout_q/layout_k: BSND / TND / PA_BBND
      dtype: torch.float16 / torch.bfloat16
      q_lens:  每 batch query token 数(BSND 要求各 batch 相同)
      k_pools: 每 batch 池数(BSND 要求各 batch 相同; TND/PA 允许 0 池 batch)
      n1/topk/pool_size/mask_mode/return_value: 算子属性
      pa_block_size:  PA 物理块大小(16 的倍数, 上限 1024)
      key_non_contig: PA key 0 轴非连续(仅 PA 支持)
      tail:  pool_tail_k; None=随机, list=每 batch 定值
      seed:   随机种子(确定性可复现)
    """
    torch.manual_seed(cfg["seed"])
    dtype = cfg["dtype"]
    ps = cfg["pool_size"]
    q_lens = list(cfg["q_lens"])
    k_pools = list(cfg["k_pools"])
    batch = len(q_lens)
    n1 = cfg["n1"]

    actual_seq_q = None
    actual_seq_k = None
    block_table = None
    npu_key_fn = None

    # ---- query / weights ----
    if cfg["layout_q"] == "BSND":
        s1 = q_lens[0]
        assert all(q == s1 for q in q_lens), "BSND 要求各 batch S1 相同"
        query = torch.randn(batch, s1, n1, HEAD_DIM, dtype=dtype)
        weights = torch.randn(batch, s1, n1, dtype=dtype)
    else:  # TND
        t1 = sum(q_lens)
        assert t1 > 0, "query 不允许空 tensor"
        query = torch.randn(t1, n1, HEAD_DIM, dtype=dtype)
        weights = torch.randn(t1, n1, dtype=dtype)
        actual_seq_q = torch.tensor(q_lens, dtype=torch.int64).cumsum(0)

    # ---- pool_key ----
    if cfg["layout_k"] == "PA_BBND":
        assert sum(k_pools) > 0, "PA 全部 batch 0 池时 pool_key 为空, 不允许"
        logical = torch.randn(sum(k_pools), N2, HEAD_DIM, dtype=dtype)
        pool_key, block_table, block_num = pack_pa_key(
            logical, k_pools, cfg["pa_block_size"]
        )
        actual_seq_k = torch.tensor(k_pools, dtype=torch.int64)  # 非前缀和
        if cfg.get("key_non_contig"):
            pool_key, npu_key_fn = make_pa_key_non_contig(
                pool_key, cfg["pa_block_size"], block_num, dtype
            )
    elif cfg["layout_q"] == "TND":  # TND/TND
        assert sum(k_pools) > 0, "pool_key 不允许空 tensor"
        pool_key = torch.randn(sum(k_pools), N2, HEAD_DIM, dtype=dtype)
        actual_seq_k = torch.tensor(k_pools, dtype=torch.int64).cumsum(0)  # 前缀和
    else:  # BSND/BSND: 不允许 S2 pad, 各 batch 池数必须相同
        pools = k_pools[0]
        assert all(p == pools for p in k_pools), "BSND 要求各 batch 池数相同"
        assert pools > 0
        pool_key = torch.randn(batch, pools, N2, HEAD_DIM, dtype=dtype)

    # ---- pool_tail_k ----
    tail = cfg.get("tail")
    if tail is None:
        pool_tail_k = torch.randint(0, ps, (batch,), dtype=torch.int64)
    else:
        pool_tail_k = torch.tensor(list(tail), dtype=torch.int64)
        assert bool((pool_tail_k >= 0).all()) and bool((pool_tail_k < ps).all())

    return {
        "query": query,
        "pool_key": pool_key,
        "weights": weights,
        "pool_tail_k": pool_tail_k,
        "actual_seq_q": actual_seq_q,
        "actual_seq_k": actual_seq_k,
        "block_table": block_table,
        "npu_key_fn": npu_key_fn,
    }


# ---------------------------------------------------------------------------
# 运行封装: 算子(eager / GE 图) 与 统一标杆(cpu / npu)
# ---------------------------------------------------------------------------


def _to_dev(inputs, dev):
    moved = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            moved[k] = v.to(dev)
        elif k == "npu_key_fn":
            moved[k] = v  # callable, 不搬移
        else:
            moved[k] = v
    return moved


def _op_tensor_args(inputs, dev):
    """算子 tensor 参数搬到 dev, 返回 7 个位置参数(供 GE 图模块 forward)。"""
    key = (
        inputs["npu_key_fn"](dev)
        if inputs["npu_key_fn"] is not None
        else inputs["pool_key"].to(dev)
    )
    return (
        inputs["query"].to(dev),
        key,
        inputs["weights"].to(dev),
        inputs["pool_tail_k"].to(dev),
        (
            inputs["actual_seq_q"].to(dev)
            if inputs["actual_seq_q"] is not None
            else None
        ),
        (
            inputs["actual_seq_k"].to(dev)
            if inputs["actual_seq_k"] is not None
            else None
        ),
        (inputs["block_table"].to(dev) if inputs["block_table"] is not None else None),
    )


def _op_call_args(inputs, cfg, dev):
    """构造 eager 算子调用参数(tensor 搬到 dev, 非连续 key 走 npu_key_fn)。"""
    q, k, w, ptk, asq, ask, bt = _op_tensor_args(inputs, dev)
    kwargs = dict(
        actual_seq_q=asq,
        actual_seq_k=ask,
        block_table=bt,
        layout_q=cfg["layout_q"],
        layout_k=cfg["layout_k"],
        topk=cfg["topk"],
        pool_size=cfg["pool_size"],
        mask_mode=cfg["mask_mode"],
        quant_mode=-1,
        return_value=cfg["return_value"],
    )
    return (q, k, w, ptk), kwargs


def run_op_eager(inputs, cfg, device_id=None):
    """eager 调用 torch extension, 返回 CPU 输出。"""
    dev = f"npu:{device_id if device_id is not None else get_device_id()}"
    args, kwargs = _op_call_args(inputs, cfg, dev)
    indices, values = pool_key_indexer(*args, **kwargs)
    torch.npu.synchronize()
    return indices.cpu(), values.cpu()


class PoolKeyIndexerModule(torch.nn.Module):
    """GE 图模式 wrapper(走 torch.ops 路径, converter 拦截点)。"""

    def __init__(self, **op_kwargs):
        super().__init__()
        self._op_kwargs = op_kwargs

    def forward(
        self,
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=None,
        actual_seq_k=None,
        block_table=None,
    ):
        return torch.ops.cann_ops_transformer.pool_key_indexer(
            query,
            pool_key,
            weights,
            pool_tail_k,
            actual_seq_q=actual_seq_q,
            actual_seq_k=actual_seq_k,
            block_table=block_table,
            **self._op_kwargs,
        )


def compile_ge_graph(module):
    """torchair GE 图后端编译(fullgraph, 静态 shape, converter 转换路径)。"""
    import torchair

    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    return torch.compile(module, fullgraph=True, backend=npu_backend, dynamic=False)


def compile_aclgraph(module):
    """torchair aclgraph(npugraph_ex, capture&replay)后端编译。

    值输入(pool_tail_k/actual_seq_q/actual_seq_k)为 NPU tensor 时, csrc
    走 aclnn Tensor 变体(2026-08-27 改造后无阻塞 D2H), 捕获/回放可用,
    device 值输入为图输入, 回放期读取 buffer 当前内容。冒烟级调用示例见
    test_smoke.py::test_smoke_torchair_aclgraph; 原生 NPUGraph 捕获语义
    的深度验证(变值回放等)见 test_aclgraph.py。
    """
    import torchair

    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    config.mode = "npugraph_ex"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    return torch.compile(module, fullgraph=True, backend=npu_backend, dynamic=False)


def run_op_ge_graph(inputs, cfg, device_id=None):
    """GE 图模式(torchair)调用算子, 返回 CPU 输出。"""
    dev = f"npu:{device_id if device_id is not None else get_device_id()}"
    model = compile_ge_graph(
        PoolKeyIndexerModule(
            layout_q=cfg["layout_q"],
            layout_k=cfg["layout_k"],
            topk=cfg["topk"],
            pool_size=cfg["pool_size"],
            mask_mode=cfg["mask_mode"],
            quant_mode=-1,
            return_value=cfg["return_value"],
        )
    )
    indices, values = model(*_op_tensor_args(inputs, dev))
    torch.npu.synchronize()
    return indices.cpu(), values.cpu()


def run_golden(inputs, cfg, ref_device="cpu"):
    """统一标杆调用(cpu 金标 / npu 组合标杆, 由 ref_device 指定)。

    返回 CPU 输出(与算子输出可比)。切换标杆类型 = 切换 ref_device,
    标杆内部按输入所在设备自动选择实现(pool_key_indexer_reference)。
    """
    dev = ref_device if ref_device != "cpu" else "cpu"
    moved = {
        "query": inputs["query"].to(dev),
        # 非连续视图: 标杆按数据语义计算, 搬运产生的连续化不影响结果
        "pool_key": inputs["pool_key"].to(dev),
        "weights": inputs["weights"].to(dev),
        "pool_tail_k": inputs["pool_tail_k"].to(dev),
        "actual_seq_q": (
            inputs["actual_seq_q"].to(dev)
            if inputs["actual_seq_q"] is not None
            else None
        ),
        "actual_seq_k": (
            inputs["actual_seq_k"].to(dev)
            if inputs["actual_seq_k"] is not None
            else None
        ),
        "block_table": (
            inputs["block_table"].to(dev) if inputs["block_table"] is not None else None
        ),
    }
    g_indices, g_values = pool_key_indexer_reference(
        moved["query"],
        moved["pool_key"],
        moved["weights"],
        moved["pool_tail_k"],
        actual_seq_q=moved["actual_seq_q"],
        actual_seq_k=moved["actual_seq_k"],
        block_table=moved["block_table"],
        layout_q=cfg["layout_q"],
        layout_k=cfg["layout_k"],
        topk=cfg["topk"],
        pool_size=cfg["pool_size"],
        mask_mode=cfg["mask_mode"],
        return_value=cfg["return_value"],
    )
    return g_indices.cpu(), g_values.cpu()


# ---------------------------------------------------------------------------
# 输出规格断言(接口文档口径)
# ---------------------------------------------------------------------------


def assert_output_spec(npu_indices, npu_values, cfg):
    """校验输出 dtype/shape 与接口规格一致(int32/float32/空 values)。"""
    ps = cfg["pool_size"]
    topk = cfg["topk"]
    sparse = topk // ps
    out_len = topk + ps - 1

    assert npu_indices.dtype == torch.int32, (
        f"indices dtype {npu_indices.dtype} != int32"
    )
    if cfg["layout_q"] == "BSND":
        exp_idx = (len(cfg["q_lens"]), cfg["q_lens"][0], out_len)
        exp_val = (len(cfg["q_lens"]), cfg["q_lens"][0], sparse)
    else:
        exp_idx = (sum(cfg["q_lens"]), out_len)
        exp_val = (sum(cfg["q_lens"]), sparse)
    assert tuple(npu_indices.shape) == exp_idx, (
        f"indices shape {tuple(npu_indices.shape)} != {exp_idx}"
    )
    if cfg["return_value"]:
        assert npu_values.dtype == torch.float32, (
            f"values dtype {npu_values.dtype} != float32"
        )
        assert tuple(npu_values.shape) == exp_val, (
            f"values shape {tuple(npu_values.shape)} != {exp_val}"
        )
    else:
        assert npu_values.numel() == 0, (
            f"return_value=False 时 values 应为空, got {tuple(npu_values.shape)}"
        )
