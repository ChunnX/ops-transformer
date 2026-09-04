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

from pool_key_indexer_reference import pool_key_indexer_reference

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


def prepare_pki_inputs(input_data: InputDataset):
    """解析并规整 ATK 输入, 算子/标杆/真值三方一致消费。

    返回 dict:
      query/pool_key/weights/pool_tail_k/actual_seq_q/actual_seq_k/block_table
      layout_q/layout_k/topk/pool_size/mask_mode/quant_mode/return_value
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

    # TND: actualSeq 随机值 -> 合法前缀和
    if layout_q == "TND" and act_seq_q is not None:
        act_seq_q = _normalize_prefix_seq(act_seq_q, query.shape[0])
    if layout_k == "TND" and act_seq_k is not None:
        act_seq_k = _normalize_prefix_seq(act_seq_k, key.shape[0])

    # PA_BBND: actualSeqK 为每 batch 池数(非前缀和), 构造合法 block_table
    if layout_k == "PA_BBND":
        if act_seq_k is None:
            b = query.shape[0] if layout_q == "BSND" else query.shape[0]
            act_seq_k = torch.full((b,), key.shape[1], dtype=torch.int64)
        block_table = _construct_block_table(
            len(act_seq_k), act_seq_k, key.shape[0], key.shape[1]
        )

    if pool_tail_k is None:
        b = query.shape[0] if layout_q == "BSND" else len(act_seq_q)
        pool_tail_k = torch.zeros((b,), dtype=torch.int64)

    # 规整产生的 tensor 归一到 query 所在设备(CPU 真值 / NPU 标杆与算子),
    # 保证 reference/算子侧输入设备一致
    if hasattr(query, "device"):
        dev = query.device
        act_seq_q = act_seq_q.to(dev) if act_seq_q is not None else None
        act_seq_k = act_seq_k.to(dev) if act_seq_k is not None else None
        block_table = block_table.to(dev) if block_table is not None else None
        pool_tail_k = pool_tail_k.to(dev)

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

        # 统一标杆(与 pytest 一致):
        #   输入在 CPU(真值任务升精度后) -> CPU 金标; 输入在 NPU -> NPU 组合标杆
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
        )
        return indices, values
