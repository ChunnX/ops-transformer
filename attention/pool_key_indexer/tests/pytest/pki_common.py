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
    if pool_key_rows.dtype == torch.float8_e4m3fn:
        # fp8 无 index 语义: 经 uint8 位视图散布(字节级等价)
        return (
            blocks.view(torch.uint8)[perm].contiguous().view(torch.float8_e4m3fn),
            bt,
            block_num,
        )
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
    if dtype == torch.float8_e4m3fn:
        # fp8 无 index 赋值 semantics: 经 uint8 位视图赋值(字节级搬运等价)
        big.view(torch.uint8)[0::2] = key_pa.view(torch.uint8)
    else:
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
      dtype: torch.float16 / torch.bfloat16 / torch.float8_e4m3fn(量化)
      q_lens:  每 batch query token 数(BSND 要求各 batch 相同)
      k_pools: 每 batch 池数(BSND 要求各 batch 相同; TND/PA 允许 0 池 batch)
      n1/topk/pool_size/mask_mode/return_value: 算子属性
      pa_block_size:  PA 物理块大小(16 的倍数, 上限 2048)
      key_non_contig: PA key 0 轴非连续(仅 PA 支持)
      tail:  pool_tail_k; None=随机, list=每 batch 定值
      seed:   随机种子(确定性可复现)
      quant_mode: -1(非量化, 默认) / 0(FP8 per-token-head) / 1(mxFP8)
    """
    torch.manual_seed(cfg["seed"])
    dtype = cfg["dtype"]
    ps = cfg["pool_size"]
    q_lens = list(cfg["q_lens"])
    k_pools = list(cfg["k_pools"])
    batch = len(q_lens)
    n1 = cfg["n1"]
    quant_mode = cfg.get("quant_mode", -1)
    is_quant = quant_mode in (0, 1)
    # fp8 dtype 无 randn/clamp 计算 semantics —— fp8 输入一律从 fp32 原值 cast
    need_fp32_gen = is_quant or (dtype == torch.float8_e4m3fn)

    actual_seq_q = None
    actual_seq_k = None
    block_table = None
    npu_key_fn = None
    q_descale = None
    k_descale = None

    # ---- query / weights / q_descale ----
    # 量化场景: 用 float32 随机数据生成后再量化 —— CPU 金标直接用 float32 原值
    # 计算避免二次量化误差, NPU 侧用 fp8 量化值(反量化等价 float32 原值)。
    # q_descale(mode=0)/mx scale(mode=1) 均以可区分的确定性小范围生成。
    if need_fp32_gen:
        # float32 原值: 幅度控制在 fp8_e4m3fn 表示精度内, 保证量化-反量化
        # 往返的相对误差可控(CPU 金标直接用 float32 原值计算)
        # weights: 量化场景规格允许 FP16/BF16, cfg["weights_dtype"] 可指定
        # (默认 FP16); 金标侧统一转 float32 计算(两种承载 dtype 数值近似)
        w_dtype = cfg.get("weights_dtype", torch.float16)
        if cfg["layout_q"] == "BSND":
            q_fp32 = (
                torch.randn(batch, q_lens[0], n1, HEAD_DIM, dtype=torch.float32) * 0.25
            )
            weights = torch.randn(batch, q_lens[0], n1, dtype=w_dtype)
        else:
            q_fp32 = torch.randn(sum(q_lens), n1, HEAD_DIM, dtype=torch.float32) * 0.25
            weights = torch.randn(sum(q_lens), n1, dtype=w_dtype)
        q_fp32 = q_fp32.clamp(-3.5, 3.5)
        query = q_fp32.to(dtype)
        query_fp32 = q_fp32
        if cfg["layout_q"] == "TND":
            actual_seq_q = torch.tensor(q_lens, dtype=torch.int64).cumsum(0)
    elif cfg["layout_q"] == "BSND":
        s1 = q_lens[0]
        assert all(q == s1 for q in q_lens), "BSND 要求各 batch S1 相同"
        query = torch.randn(batch, s1, n1, HEAD_DIM, dtype=dtype)
        weights = torch.randn(batch, s1, n1, dtype=dtype)
        query_fp32 = None
    else:  # TND
        t1 = sum(q_lens)
        assert t1 > 0, "query 不允许空 tensor"
        query = torch.randn(t1, n1, HEAD_DIM, dtype=dtype)
        weights = torch.randn(t1, n1, dtype=dtype)
        query_fp32 = None
        actual_seq_q = torch.tensor(q_lens, dtype=torch.int64).cumsum(0)

    # ---- q_descale ----
    if quant_mode == 0:
        # mode=0: (B,S1,N1)/(T1,N1) float32
        if cfg["layout_q"] == "BSND":
            q_descale = (
                torch.rand(batch, q_lens[0], n1, dtype=torch.float32) * 3.0 + 0.25
            )
        else:
            q_descale = torch.rand(sum(q_lens), n1, dtype=torch.float32) * 3.0 + 0.25
    elif quant_mode == 1:
        # mode=1: (B,S1,N1,D/64,2)/(T1,N1,D/64,2) E8M0(raw 编码 0..254)
        tail = (HEAD_DIM // 64, 2)
        if cfg["layout_q"] == "BSND":
            base = (batch, q_lens[0], n1)
        else:
            base = (sum(q_lens), n1)
        q_descale = torch.randint(120, 136, tuple(base) + tail, dtype=torch.uint8)

    # ---- pool_key / k_descale ----
    if cfg["layout_k"] == "PA_BBND":
        assert sum(k_pools) > 0, "PA 全部 batch 0 池时 pool_key 为空, 不允许"
        if need_fp32_gen:
            logical = (
                (torch.randn(sum(k_pools), N2, HEAD_DIM, dtype=torch.float32) * 0.25)
                .clamp(-3.5, 3.5)
                .to(dtype)
            )
        else:
            logical = torch.randn(sum(k_pools), N2, HEAD_DIM, dtype=dtype)
        pool_key, block_table, block_num = pack_pa_key(
            logical, k_pools, cfg["pa_block_size"]
        )
        actual_seq_k = torch.tensor(k_pools, dtype=torch.int64)  # 非前缀和
        if cfg.get("key_non_contig"):
            pool_key, npu_key_fn = make_pa_key_non_contig(
                pool_key, cfg["pa_block_size"], block_num, dtype
            )
        k_ds_logical = _make_k_descale_logical(quant_mode, k_pools)
        if k_ds_logical is not None:
            # PA: 与 pool_key 同一 block_table 打包(物理块对齐)
            if quant_mode == 0:
                # 规格: (blockNum, blockSize, N2) 3D
                k_descale = _pack_pa_descale(
                    k_ds_logical.reshape(sum(k_pools), 1),
                    k_pools,
                    cfg["pa_block_size"],
                    block_table,
                    pad_value=0.0,
                ).squeeze(-1)  # 去掉打包用的尾维 1
            else:
                # 规格: (blockNum, blockSize, N2, D/64, 2) 5D
                k_descale = _pack_pa_descale(
                    k_ds_logical,
                    k_pools,
                    cfg["pa_block_size"],
                    block_table,
                    pad_value=0,
                )
    elif cfg["layout_q"] == "TND":  # TND/TND
        assert sum(k_pools) > 0, "pool_key 不允许空 tensor"
        if need_fp32_gen:
            pool_key = (
                (torch.randn(sum(k_pools), N2, HEAD_DIM, dtype=torch.float32) * 0.25)
                .clamp(-3.5, 3.5)
                .to(dtype)
            )
        else:
            pool_key = torch.randn(sum(k_pools), N2, HEAD_DIM, dtype=dtype)
        actual_seq_k = torch.tensor(k_pools, dtype=torch.int64).cumsum(0)  # 前缀和
        k_ds_logical = _make_k_descale_logical(quant_mode, k_pools)
        if k_ds_logical is not None:
            if quant_mode == 0:
                k_descale = k_ds_logical.reshape(sum(k_pools), N2)
            else:
                k_descale = k_ds_logical.reshape(
                    sum(k_pools), N2, HEAD_DIM // 64, 2
                )  # (T2,N2,D/64,2)
    else:  # BSND/BSND: 不允许 S2 pad, 各 batch 池数必须相同
        pools = k_pools[0]
        assert all(p == pools for p in k_pools), "BSND 要求各 batch 池数相同"
        assert pools > 0
        if need_fp32_gen:
            pool_key = (
                (torch.randn(batch, pools, N2, HEAD_DIM, dtype=torch.float32) * 0.25)
                .clamp(-3.5, 3.5)
                .to(dtype)
            )
        else:
            pool_key = torch.randn(batch, pools, N2, HEAD_DIM, dtype=dtype)
        k_ds_logical = _make_k_descale_logical(quant_mode, k_pools)
        if k_ds_logical is not None:
            if quant_mode == 0:
                k_descale = k_ds_logical.reshape(batch, pools, N2)
            else:
                k_descale = k_ds_logical.reshape(batch, pools, N2, HEAD_DIM // 64, 2)

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
        "q_descale": q_descale,
        "k_descale": k_descale,
        "query_fp32": query_fp32 if is_quant else None,
    }


def _make_k_descale_logical(quant_mode, k_pools):
    """构造逻辑连续的 k_descale(逐池展开顺序, 打包/reshape 前的中间形态)。

    mode=0: (sum_pools,) float32 —— 每池 1 个 scale
    mode=1: (sum_pools, D/64, 2) uint8 —— 每池 D/32 个 E8M0 raw 编码
    """
    total = sum(k_pools)
    if total == 0:
        return None
    if quant_mode == 0:
        return torch.rand(total, dtype=torch.float32) * 3.0 + 0.25
    if quant_mode == 1:
        return torch.randint(120, 136, (total, HEAD_DIM // 64, 2), dtype=torch.uint8)
    return None


def _pack_pa_descale(descale_rows, k_pool_lens, block_size, block_table, pad_value=0):
    """逻辑 k_descale(sum_pools, ...) 按 pool_key 的 block_table 打包到 PA 布局。

    与 pack_pa_key 的物理排布一致: 逐 batch 前缀填块后按 bt[b][i] 的逆映射
    放置物理块(同一置换), 保证 bt[b][i] 指向的物理块与 pool_key 的池数据对齐。
    descale_rows: (sum_pools, [D/64, 2]) 或 (sum_pools, 1);
    返回 (blockNum, blockSize, 1, [D/64, 2]) 形态(N2 维=1)。
    """
    blocks_per_batch = [(pl + block_size - 1) // block_size for pl in k_pool_lens]
    block_num = sum(blocks_per_batch)
    tail_dims = descale_rows.shape[1:]
    # 先按"逐 batch 前缀填块"的原始逻辑顺序组块
    ordered = torch.full(
        (block_num, block_size, 1, *tail_dims), pad_value, dtype=descale_rows.dtype
    )
    row_start = blk_start = 0
    for pools, nb in zip(k_pool_lens, blocks_per_batch):
        if nb == 0:
            continue
        seg = descale_rows[row_start : row_start + pools]
        pad = nb * block_size - pools
        if pad:
            seg = torch.cat(
                [seg, torch.full((pad, *tail_dims), pad_value, dtype=seg.dtype)], dim=0
            )
        ordered[blk_start : blk_start + nb] = seg.view(nb, block_size, 1, *tail_dims)
        row_start += pools
        blk_start += nb
    # 按 bt 逆映射散布: 物理块 p 存放逻辑块 inv[p]
    # (pack_pa_key 中 bt[b][i] = inv[cur], 即逻辑块 cur 放在物理块 inv[cur])
    blocks_before = [0]
    for pl in k_pool_lens:
        blocks_before.append(blocks_before[-1] + (pl + block_size - 1) // block_size)
    out = torch.empty_like(ordered)
    placed = torch.zeros(block_num, dtype=torch.bool)
    for b in range(block_table.shape[0]):
        for i in range(block_table.shape[1]):
            v = int(block_table[b, i])
            if v >= 0:
                out[v] = ordered[blocks_before[b] + i]
                placed[v] = True
    # bt 覆盖不到的物理块(理论上不存在: block_num == sum blocks_per_batch)
    for p in range(block_num):
        if not placed[p]:
            out[p] = ordered[p]
    return out.contiguous()


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
    qd = inputs["q_descale"].to(dev) if inputs.get("q_descale") is not None else None
    kd = inputs["k_descale"].to(dev) if inputs.get("k_descale") is not None else None
    if qd is not None and cfg.get("quant_mode", -1) == 1:
        # E8M0: torch 无 float8_e8m0fnu 计算语义, 以 uint8 视图搬运后
        # view 回 e8m0 dtype(与 QLIv2 golden 的 make_e8m0_tensor_from_raw 一致)
        qd = _view_as_e8m0(qd)
    if kd is not None and cfg.get("quant_mode", -1) == 1:
        kd = _view_as_e8m0(kd)
    kwargs = dict(
        actual_seq_q=asq,
        actual_seq_k=ask,
        block_table=bt,
        q_descale=qd,
        k_descale=kd,
        layout_q=cfg["layout_q"],
        layout_k=cfg["layout_k"],
        topk=cfg["topk"],
        pool_size=cfg["pool_size"],
        mask_mode=cfg["mask_mode"],
        quant_mode=cfg.get("quant_mode", -1),
        return_value=cfg["return_value"],
    )
    return (q, k, w, ptk), kwargs


def _view_as_e8m0(t):
    """uint8 tensor(连续) -> torch.float8_e8m0fnu 视图(dtype 重解释)。"""
    if t.dtype == torch.uint8:
        return t.view(torch.float8_e8m0fnu)
    return t


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


def _e8m0_to_scale(t):
    """E8M0 raw(uint8) -> float32 scale: 2^(raw-127), raw==255 -> NaN。"""
    raw = t.detach().cpu().to(torch.uint8)
    scale = torch.pow(2.0, raw.to(torch.float32) - 127.0)
    scale = torch.where(raw == 255, torch.full_like(scale, float("nan")), scale)
    return scale


def _capture_run(cfg, run):
    """标准 NPUGraph 捕获流程: 侧流预热 + 捕获(与 test_aclgraph.py 口径一致)。"""
    torch_npu.npu.set_device(get_device_id())
    s = torch_npu.npu.Stream()
    s.wait_stream(torch_npu.npu.current_stream())
    with torch_npu.npu.stream(s):
        for _ in range(3):
            run()
    torch_npu.npu.current_stream().wait_stream(s)

    g = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(g):
        out = run()
    return g, out


def run_op_aclgraph(inputs, cfg, device_id=None):
    """aclgraph(原生 NPUGraph)捕获/回放调用算子, 返回 CPU 输出。

    值输入(pool_tail_k/actual_seq_q/k)搬至 NPU, csrc 走 aclnn Tensor 变体
    (捕获期无阻塞 D2H), device 值输入为图输入。侧流预热 + 捕获 + 回放,
    回放输出与 CPU golden 比对。return_value=False 时 values 为空 tensor,
    与 assert_output_spec 的 empty 口径兼容。
    """
    dev = f"npu:{device_id if device_id is not None else get_device_id()}"
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

    def run():
        return pool_key_indexer(q, k, w, ptk, **kwargs)

    g, (indices, values) = _capture_run(cfg, run)
    g.replay()
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
    # 量化 descale: mode=1 的 E8M0 在标杆侧解码为 float32 scale(尾维 [D/64,2]
    # 展平为 D/32, 与 reference 的 mx 语义对齐)
    q_descale = None
    k_descale = None
    quant_mode = cfg.get("quant_mode", -1)
    if inputs.get("q_descale") is not None:
        q_descale = inputs["q_descale"].to(dev)
        if quant_mode == 1:
            q_descale = _e8m0_to_scale(q_descale).reshape(*q_descale.shape[:-2], -1)
    if inputs.get("k_descale") is not None:
        k_descale = inputs["k_descale"].to(dev)
        if quant_mode == 1:
            k_descale = _e8m0_to_scale(k_descale).reshape(*k_descale.shape[:-2], -1)
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
        q_descale=q_descale,
        k_descale=k_descale,
        quant_mode=quant_mode,
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
