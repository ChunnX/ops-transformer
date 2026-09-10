#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PoolKeyIndexer ATK 自定义执行方式 (BaseApi) + 三节点共享输入规整工具。

注册名: "pool_key_indexer" (对应 YAML api_type)

执行拓扑(双标杆精度标准 pki_double_benchmark):
  - operator  (pyaclnn): aclnn 接口调用自定义算子, 原始精度输入;
  - npu_golden(npu)     : NPU 组合标杆(与 pytest 的 _reference_npu 一致), 原始精度输入;
  - 真值      (cpu_benchmark, ATK benchmark 任务自动触发): CPU 金标 + 升精度输入
    (fp16/bf16 -> fp32), 与 pytest --ref-device cpu 的金标一致。

本 Api 按输入所在设备自动路由(与 pytest 的 pool_key_indexer_reference 一致):
  - 输入在 CPU(真值任务, 升精度后)  -> CPU 金标实现(_reference_cpu);
  - 输入在 NPU(npu_golden 节点)     -> NPU 组合标杆实现(_reference_npu)。

输入规整(prepare_pki_inputs)为全节点共用(算子适配器/标杆/真值):
ATK 随机生成的 actualSeq / blockTable 需要确定性规整为合法形态,
规整只依赖输入数据本身(种子固定), 保证三方消费到完全一致的输入。
"""

import math

import numpy as np
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from pool_key_indexer_reference import (
    pool_key_indexer_reference,
    _e8m0_decode,
    _reference_npu,
    _reference_npu_blocked,
)

LAYOUT_MAP = {0: "BSND", 1: "TND", 2: "PA_BBND"}

# 输入索引 (与 YAML inputs 顺序一致)
IDX_QUERY = 0
IDX_POOL_KEY = 1
IDX_WEIGHTS = 2
IDX_POOL_TAIL_K = 3
IDX_ACT_SEQ_Q = 4
IDX_ACT_SEQ_K = 5
IDX_BLOCK_TABLE = 6
IDX_Q_DESCALE = 7
IDX_K_DESCALE = 8
IDX_LAYOUT_Q = 9
IDX_LAYOUT_K = 10
IDX_TOPK = 11
IDX_POOL_SIZE = 12
IDX_MASK_MODE = 13
IDX_QUANT_MODE = 14
IDX_RETURN_VALUE = 15
IDX_TND_UNEQUAL = 16
IDX_ZERO_Q_BATCH = 17
IDX_ZERO_POOL_BATCH = 18
IDX_KEY_NON_CONTIG = 19


def _parse_layout(raw):
    if isinstance(raw, (int, float)):
        return LAYOUT_MAP.get(int(raw), "BSND")
    return str(raw)


def _parse_bool(raw, default=False):
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(int(raw))
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "1")
    return default


def _tensor_or_none(t):
    """空 tensor(shape[0]) -> None(可选输入未启用)。"""
    if t is None:
        return None
    if hasattr(t, "shape") and t.numel() == 0:
        return None
    return t


def _normalize_prefix_seq(seq, total):
    """ATK 随机生成的 TND actualSeq 规整为合法前缀和(末项 = total)。

    只依赖 len(seq) 与 total, 确定性输出, 三节点一致。
    """
    b = len(seq)
    if b <= 1:
        return torch.tensor([total], dtype=torch.int64)
    lens = [total // b] * b
    lens[-1] += total - sum(lens)
    return torch.tensor(lens, dtype=torch.int64).cumsum(0).to(torch.int64)


def _distribute(total, b):
    """total 均分到 b 份, 余数逐份 +1 分散(不堆在末 batch)。"""
    base = total // b
    rem = total - base * b
    return [base + 1 if i < rem else base for i in range(b)]


def _split_batch_lens(b, total, mode="uniform", max_ratio=2.0):
    """将 total 确定性拆分为 b 份逐 batch 长度(与随机 tensor 值解耦)。

    mode:
      - "uniform": 均分(默认)
      - "unequal": 头 batch 占大头, 其余均分(不等长 batch)
      - "zero_head": 首 batch 为 0, 其余按 unequal 拆(0 长度/0 池 batch)
    b<2 或余量不足时退化为 uniform。

    max_ratio: 单 batch 长度相对均匀份额的上限。必须设上限:
    NPU 组合标杆按 (B, S1m, Pm) 稠密化, 若头 batch 按 total 的固定比例
    (如 60%)取值, 大 batch 用例单 batch 峰值放大 ~b/2 倍, 稠密矩阵
    随 batch 维再乘 B, 内存 O(b^2) 爆炸(实测 51 batch 时 dot 矩阵
    ~77GB 导致 NPU OOM)。上限 2x 保证峰值 <= 均匀基线的 ~4x。
    (zero_head 的峰值上限按有效份额 total/(b-1) 计, 略超 2x*total/b
    属预期。)
    """
    if b <= 0 or total < 0:
        return []
    if b == 1:
        return [total]
    if mode == "zero_head":
        if total < b - 1:  # 其余 batch 各至少 1 个
            mode = "uniform"
        else:
            return [0] + _split_batch_lens(b - 1, total, "unequal", max_ratio)
    if mode == "unequal":
        eff = b - 1
        head = max(1, int(total * 0.6))
        head = min(head, int(max_ratio * total) // b)
        head = min(head, total - eff)
        head = max(head, 1)
        return [head] + _distribute(total - head, eff)
    return _distribute(total, b)


def make_pa_key_non_contig(key):
    """PA key 0 轴非连续视图: 物理块嵌入 2 倍缓冲偶数行(与 pytest
    make_pa_key_non_contig 语义一致), stride(0) = 2 * blockSize * N2 * D。

    必须在目标设备上调用(非稠密视图跨设备 .to() 会被拍平为连续,
    丢失 stride 信息)。"""
    bn, bs, n2, d = key.shape
    big = torch.zeros(2 * bn, bs, n2, d, dtype=key.dtype, device=key.device)
    big[0::2] = key
    return big[::2]


def _construct_block_table(batch_size, act_seq_k, block_num, block_size):
    """PA_BBND 合法 block_table 构造(与 pytest pack_pa_key 语义一致)。

    每 batch 池独立连续填块, 物理块随机排布一次(种子固定),
    bt[b][i] 指向的物理块真实存放 batch b 第 i 组池; 0 池 batch 不占块。
    """
    max_block_num_per_seq = 0
    block_num_per_batch = []
    for b in range(batch_size):
        cur_k = (
            int(act_seq_k[b].item())
            if torch.is_tensor(act_seq_k)
            else int(act_seq_k[b])
        )
        cur_blocks = math.ceil(cur_k / block_size) if cur_k > 0 else 0
        block_num_per_batch.append(cur_blocks)
        max_block_num_per_seq = max(max_block_num_per_seq, cur_blocks)

    rng = np.random.RandomState(block_num * 31 + batch_size * 7 + max_block_num_per_seq)
    block_id_list = rng.permutation(block_num).astype(np.int32)
    bt = np.full((batch_size, max_block_num_per_seq), -1, dtype=np.int32)
    cur_id = 0
    for b in range(batch_size):
        for i in range(block_num_per_batch[b]):
            if cur_id < block_num:
                bt[b][i] = block_id_list[cur_id]
                cur_id += 1
    return torch.from_numpy(bt).to(torch.int32)


def prepare_pki_inputs(input_data: InputDataset, build_non_contig=True):
    """解析并规整 ATK 输入, 算子/标杆/真值三方一致消费。

    build_non_contig: 是否在 key 所在设备构造 PA 非连续视图。
    pyaclnn 适配器传 False(输入在 npu:0, 需搬到目标设备后再构造,
    避免跨设备 .to() 拍平视图), 由适配器自行构造并读取 keyStride0。

    返回 dict:
      query/pool_key/weights/pool_tail_k/actual_seq_q/actual_seq_k/block_table
      layout_q/layout_k/topk/pool_size/mask_mode/quant_mode/return_value
      tnd_unequal/zero_q_batch/zero_pool_batch/key_non_contig
      q_descale/k_descale(算子侧: mode=0 fp32 scale / mode=1 uint8 E8M0 raw)
      q_descale_scales/k_descale_scales(标杆侧: mode=1 已解码 [.., D/32] float;
      mode=0 与原 tensor 相同; 非量化恒 None)
    """
    args = list(input_data.args) if input_data.args else []
    kw = dict(input_data.kwargs) if input_data.kwargs else {}

    def pick(idx, name):
        if idx < len(args) and args[idx] is not None:
            return args[idx]
        return kw.get(name)

    def pick_attr(idx, name, default):
        value = pick(idx, name)
        return default if value is None else value

    query = pick(IDX_QUERY, "query")
    key = pick(IDX_POOL_KEY, "poolKey")
    weights = pick(IDX_WEIGHTS, "weights")
    pool_tail_k = _tensor_or_none(pick(IDX_POOL_TAIL_K, "poolTailK"))
    act_seq_q = _tensor_or_none(pick(IDX_ACT_SEQ_Q, "actualSeqQ"))
    act_seq_k = _tensor_or_none(pick(IDX_ACT_SEQ_K, "actualSeqK"))
    block_table = _tensor_or_none(pick(IDX_BLOCK_TABLE, "blockTable"))

    # 属性双路读取: 生成态在 args, 落盘加载态在 kwargs(按名)
    layout_q = _parse_layout(pick_attr(IDX_LAYOUT_Q, "layout_q", "BSND"))
    layout_k = _parse_layout(pick_attr(IDX_LAYOUT_K, "layout_k", "BSND"))
    topk = int(pick_attr(IDX_TOPK, "topk", 2048))
    pool_size = int(pick_attr(IDX_POOL_SIZE, "pool_size", 1))
    mask_mode = int(pick_attr(IDX_MASK_MODE, "mask_mode", 3))
    quant_mode = int(pick_attr(IDX_QUANT_MODE, "quant_mode", -1))
    return_value = _parse_bool(pick_attr(IDX_RETURN_VALUE, "return_value", False))
    # 场景标志(旧 JSON 无字段时回退 False)
    tnd_unequal = _parse_bool(pick_attr(IDX_TND_UNEQUAL, "tnd_unequal", False))
    zero_q_batch = _parse_bool(pick_attr(IDX_ZERO_Q_BATCH, "zero_q_batch", False))
    zero_pool_batch = _parse_bool(
        pick_attr(IDX_ZERO_POOL_BATCH, "zero_pool_batch", False)
    )
    key_non_contig = _parse_bool(pick_attr(IDX_KEY_NON_CONTIG, "key_non_contig", False))

    # TND actualSeq: 按场景标志确定性拆分(不等长/0长度 batch), 前缀和
    if layout_q == "TND" and act_seq_q is not None:
        mode = (
            "zero_head" if zero_q_batch else ("unequal" if tnd_unequal else "uniform")
        )
        lens = _split_batch_lens(len(act_seq_q), query.shape[0], mode)
        act_seq_q = torch.tensor(lens, dtype=torch.int64).cumsum(0).to(torch.int64)
    if layout_k == "TND" and act_seq_k is not None:
        mode = (
            "zero_head"
            if zero_pool_batch
            else ("unequal" if tnd_unequal else "uniform")
        )
        lens = _split_batch_lens(len(act_seq_k), key.shape[0], mode)
        act_seq_k = torch.tensor(lens, dtype=torch.int64).cumsum(0).to(torch.int64)

    # PA_BBND: actualSeqK 为每 batch 池数(非前缀和), 构造合法 block_table
    if layout_k == "PA_BBND":
        b_pa = len(act_seq_k) if act_seq_k is not None else query.shape[0]
        if zero_pool_batch or tnd_unequal:
            # 不等长/0池: 按 blockNum 拆分物理块(总量精确匹配 shape),
            # 每 batch 池数 = 块数 * blockSize
            mode = "zero_head" if zero_pool_batch else "unequal"
            blocks = _split_batch_lens(b_pa, key.shape[0], mode)
            act_seq_k = torch.tensor(
                [bl * key.shape[1] for bl in blocks], dtype=torch.int64
            )
        elif act_seq_k is None:
            act_seq_k = torch.full((b_pa,), key.shape[1], dtype=torch.int64)
        block_table = _construct_block_table(
            b_pa, act_seq_k, key.shape[0], key.shape[1]
        )

    # PA key 0 轴非连续: 在 key 所在设备构造视图
    # (pyaclnn 适配器传 build_non_contig=False, 由其在目标设备自行构造)
    if (
        build_non_contig
        and key_non_contig
        and layout_k == "PA_BBND"
        and key.is_contiguous()
    ):
        key = make_pa_key_non_contig(key)

    if pool_tail_k is None:
        b = query.shape[0] if layout_q == "BSND" else len(act_seq_q)
        pool_tail_k = torch.zeros((b,), dtype=torch.int64)

    # ---- 量化 descale 规整(quant_mode=0/1; 非量化恒为 None, 行为不变) ----
    # q_descale/k_descale:
    #   - prep["q_descale"]/["k_descale"]: 算子侧原始形态
    #       mode=0: FLOAT scale(真值任务升精度后可能是 fp64, 统一回 fp32, 精确);
    #       mode=1: uint8 E8M0 raw(适配层视图为 float8_e8m0fnu 下传);
    #   - prep["q_descale_scales"]/["k_descale_scales"]: 标杆侧解码形态
    #       mode=0: 同原 tensor; mode=1: 2^(raw-127) 解码 + 尾维 [D/64,2]
    #       展平为 [D/32](与 pytest run_golden 的 e8m0 预处理一致)
    q_descale = _tensor_or_none(pick(IDX_Q_DESCALE, "qDescale"))
    k_descale = _tensor_or_none(pick(IDX_K_DESCALE, "kDescale"))
    if quant_mode not in (0, 1):
        q_descale = None
        k_descale = None
    q_descale_scales = None
    k_descale_scales = None
    if q_descale is not None or k_descale is not None:
        if quant_mode == 0:
            if q_descale is not None:
                q_descale = q_descale.to(torch.float32)
                q_descale_scales = q_descale
            if k_descale is not None:
                k_descale = k_descale.to(torch.float32)
                k_descale_scales = k_descale
        else:  # mxFP8: raw -> float scale, 尾维 [D/64, 2] 展平为 D/32
            if q_descale is not None:
                q_descale_scales = _e8m0_decode(q_descale).reshape(
                    *q_descale.shape[:-2], -1
                )
            if k_descale is not None:
                k_descale_scales = _e8m0_decode(k_descale).reshape(
                    *k_descale.shape[:-2], -1
                )

    # 规整产生的 tensor 归一到 query 所在设备(CPU 真值 / NPU 标杆与算子),
    # 保证 reference/算子侧输入设备一致
    if hasattr(query, "device"):
        dev = query.device
        act_seq_q = act_seq_q.to(dev) if act_seq_q is not None else None
        act_seq_k = act_seq_k.to(dev) if act_seq_k is not None else None
        block_table = block_table.to(dev) if block_table is not None else None
        pool_tail_k = pool_tail_k.to(dev)
        q_descale = q_descale.to(dev) if q_descale is not None else None
        k_descale = k_descale.to(dev) if k_descale is not None else None
        q_descale_scales = (
            q_descale_scales.to(dev) if q_descale_scales is not None else None
        )
        k_descale_scales = (
            k_descale_scales.to(dev) if k_descale_scales is not None else None
        )

    return dict(
        query=query,
        pool_key=key,
        weights=weights,
        pool_tail_k=pool_tail_k,
        actual_seq_q=act_seq_q,
        actual_seq_k=act_seq_k,
        block_table=block_table,
        layout_q=layout_q,
        layout_k=layout_k,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=quant_mode,
        return_value=return_value,
        tnd_unequal=tnd_unequal,
        zero_q_batch=zero_q_batch,
        zero_pool_batch=zero_pool_batch,
        key_non_contig=key_non_contig,
        q_descale=q_descale,
        k_descale=k_descale,
        q_descale_scales=q_descale_scales,
        k_descale_scales=k_descale_scales,
    )


def pki_output_stubs(prep):
    """is_get_output_task 场景(输出规格推导): 返回正确 shape/dtype 的空 tensor。"""
    query = prep["query"]
    layout_q = prep["layout_q"]
    topk = prep["topk"]
    pool_size = prep["pool_size"]
    out_len = topk + pool_size - 1
    sparse = topk // pool_size
    if layout_q == "BSND":
        idx_shape = [query.shape[0], query.shape[1], out_len]
        val_shape = [query.shape[0], query.shape[1], sparse]
    else:
        idx_shape = [query.shape[0], out_len]
        val_shape = [query.shape[0], sparse]
    indices = torch.empty(idx_shape, dtype=torch.int32)
    if prep["return_value"]:
        values = torch.empty(val_shape, dtype=torch.float32)
    else:
        values = torch.empty(0, dtype=torch.float32)
    return indices, values


def _estimate_npu_golden_load(prep):
    """估算 NPU 组合标杆稠密化后的峰值元素数。

    _reference_npu 按 (B, S1m, Pm) 稠密化: scores 一份 + sort 工作区 ~2x +
    展开输出 (B, S1m, topk+ps-1) 多份 —— 极端 shape(zero_q/unequal 使 S1m
    翻倍、大 topk)下峰值数十 GB, NPU 64GB HBM 放不下: 实测 OOM 或挂死
    (AdvancedIndex/NonZero aicpu timeout ~10min 且进程级不可捕获, 会级联
    饿死 worker 内后续用例)。返回 (scores元素数, 展开输出元素数) 供上限判断。
    """
    q = prep["query"]
    ps = prep["pool_size"]
    topk = prep["topk"]
    if prep["layout_q"] == "BSND":
        b, s1m = int(q.shape[0]), int(q.shape[1])
    else:
        aq = prep["actual_seq_q"]
        if aq is not None and len(aq) > 0:
            lens = [int(aq[0])] + [int(aq[i] - aq[i - 1]) for i in range(1, len(aq))]
            b, s1m = len(lens), (max(lens) if lens else 0)
        else:
            b, s1m = 1, int(q.shape[0])
    ak = prep["actual_seq_k"]
    key = prep["pool_key"]
    if prep["layout_k"] == "BSND":
        pm = int(key.shape[1])
    elif ak is not None and len(ak) > 0:
        if prep["layout_k"] == "TND":
            lensk = [int(ak[0])] + [int(ak[i] - ak[i - 1]) for i in range(1, len(ak))]
            pm = max(lensk) if lensk else 0
        else:  # PA: 每 batch 池数(非前缀和)
            pm = int(max(int(v) for v in ak))
    else:
        pm = int(key.shape[0])
    return b * s1m * pm, b * s1m * (topk + ps - 1)


# NPU 组合标杆稠密化元素上限(超过走分块 streaming 路径 _reference_npu_blocked,
# 数值语义一致): 原稠密实现一次性物化 (B,S1m,Pm) scores, 极端 shape(ps=1 +
# 池数 11~17 万)下单矩阵 50~500GB 放不下; 中大 shape 放得下但 worker 长跑后
# caching allocator 碎片化, 数 GB 级分配劣化直至 aicpu 看门狗超时(进程级
# 不可捕获, 级联饿死后续用例)。blocked 实现峰值仅 S1m*blockW + S1m*k_sel*2,
# 实测极端 case(ps=1/11~17 万池)11~39s 完成(稠密版 CPU 降级路径 30~90 分钟
# 仍超时)。CPU 向量化仅作 blocked 亦 OOM 的兜底。
_NPU_GOLDEN_MAX_SCORE_ELEMS = 50_000_000
_NPU_GOLDEN_MAX_TOKEN_ELEMS = 250_000_000


@register("pool_key_indexer")
class PKIApi(BaseApi):
    """PoolKeyIndexer 统一标杆执行器(CPU 金标 / NPU 组合标杆, 按输入设备路由)。"""

    def __init__(self, task_result):
        super().__init__(task_result)
        self.run_modes = []

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if not with_output:
            return None

        prep = prepare_pki_inputs(input_data)

        if getattr(self.task_result, "is_get_output_task", False):
            return pki_output_stubs(prep)

        # 量化场景: NPU 组合标杆暂不支持 descale(pool_key_indexer_reference
        # 约定), 统一搬 CPU 走金标 —— 输入已在 CPU 时无额外搬运
        quant = prep["quant_mode"] in (0, 1)
        if quant and prep["query"].device.type != "cpu":
            prep = {
                k: (v.cpu() if hasattr(v, "cpu") and torch.is_tensor(v) else v)
                for k, v in prep.items()
            }

        # 统一标杆(与 pytest 一致):
        #   输入在 CPU(真值任务升精度后) -> CPU 金标; 输入在 NPU -> NPU 组合标杆
        #   极端 shape(稠密化超限) -> 双侧均走分块 streaming 实现(数值语义
        #   一致, 峰值内存 S1m*blockW):
        #     NPU 标杆: 原稠密版放不下/碎片化挂死, blocked 11~39s 完成;
        #     CPU 真值: _reference_cpu 逐行 Python 循环在 11 万行规格下
        #     30~90 分钟超时, blocked 向量化(CPU)数分钟内完成。
        #   NPU 侧 blocked 仍 OOM 时兜底 CPU 向量化。
        scores_elems, token_elems = _estimate_npu_golden_load(prep)
        oversized = (
            scores_elems > _NPU_GOLDEN_MAX_SCORE_ELEMS
            or token_elems > _NPU_GOLDEN_MAX_TOKEN_ELEMS
        )
        if oversized and not quant:
            common = dict(
                actual_seq_q=prep["actual_seq_q"],
                actual_seq_k=prep["actual_seq_k"],
                block_table=prep["block_table"],
                layout_q=prep["layout_q"],
                layout_k=prep["layout_k"],
                topk=prep["topk"],
                pool_size=prep["pool_size"],
                mask_mode=prep["mask_mode"],
                return_value=prep["return_value"],
            )
            if prep["query"].device.type == "npu":
                try:
                    indices, values = _reference_npu_blocked(
                        prep["query"],
                        prep["pool_key"],
                        prep["weights"],
                        prep["pool_tail_k"],
                        **common,
                    )
                    return indices, values
                except torch.OutOfMemoryError:
                    torch.npu.empty_cache()
                    prep = {
                        k: (v.cpu() if torch.is_tensor(v) else v)
                        for k, v in prep.items()
                    }
            indices, values = _reference_npu_blocked(
                prep["query"],
                prep["pool_key"],
                prep["weights"],
                prep["pool_tail_k"],
                **common,
            )
            return indices, values
        indices, values = pool_key_indexer_reference(
            prep["query"],
            prep["pool_key"],
            prep["weights"],
            prep["pool_tail_k"],
            actual_seq_q=prep["actual_seq_q"],
            actual_seq_k=prep["actual_seq_k"],
            block_table=prep["block_table"],
            layout_q=prep["layout_q"],
            layout_k=prep["layout_k"],
            topk=prep["topk"],
            pool_size=prep["pool_size"],
            mask_mode=prep["mask_mode"],
            return_value=prep["return_value"],
            q_descale=prep["q_descale_scales"] if quant else None,
            k_descale=prep["k_descale_scales"] if quant else None,
            quant_mode=prep["quant_mode"] if quant else -1,
        )
        return indices, values
