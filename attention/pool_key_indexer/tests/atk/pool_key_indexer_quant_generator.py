#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PoolKeyIndexer 量化场景 ATK 用例参数约束生成器(quant_mode=0/1, 仅 Ascend950)。

注册名: "pki_quant_gen" (对应 pool_key_indexer_quant.yaml 的 generate 字段)

维度覆盖(与非量化生成器对齐, 不遗漏):
  - quant_mode: 0(per-token FP8, descale=FLOAT) / 1(mxFP8, descale=E8M0)
  - 布局: 全部 4 组合(BSND/BSND, BSND/PA_BBND, TND/TND, TND/PA_BBND)
  - weights dtype: FP16/BF16(量化规格允许, 与 fp8 q/k 独立)
  - mask_mode 0/3, pool_size 1~128 全档, topk 全分桶(小/中/大)
  - return_value, pool_tail_k 尾块, 池数<sparseCount 截断
  - 不等长 batch / 0 长度 query batch / 0 池 batch / PA key 0 轴非连续
  - n1 极值 1 与 64(N1_LIMIT), batch 1~16, S1 含极值 1

规模刻意收敛(CPU 金标逐行 Python 循环, 且 FP8+scale 分数近并列概率高于
fp16 —— 用中小规模控制 TopK 区分度, 与 pytest 量化套件口径一致):
  S1 <= 256, batch <= 16, 池数 <= 2048, 注意力元素 <= 1M。

descale 形状(与 op_host tiling 校验一致):
  mode=0: q (B,S1,N1)/(T1,N1); k (B,S2,N2)/(T2,N2)/(blockNum,blockSize,N2), FLOAT
  mode=1: 尾维追加 [headDim/64, 2] = [2, 2], uint8 生成(E8M0 raw, 适配层视图)
"""

import math
import random

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig

# 复用非量化生成器的共享常量与工具(不修改其行为)
from pool_key_indexer_generator import (
    HEAD_DIM,
    N2,
    IDX_QUERY,
    IDX_POOL_KEY,
    IDX_WEIGHTS,
    IDX_POOL_TAIL_K,
    IDX_ACT_SEQ_Q,
    IDX_ACT_SEQ_K,
    IDX_BLOCK_TABLE,
    IDX_Q_DESCALE,
    IDX_K_DESCALE,
    IDX_LAYOUT_Q,
    IDX_LAYOUT_K,
    IDX_TOPK,
    IDX_POOL_SIZE,
    IDX_MASK_MODE,
    IDX_QUANT_MODE,
    IDX_RETURN_VALUE,
    IDX_TND_UNEQUAL,
    IDX_ZERO_Q_BATCH,
    IDX_ZERO_POOL_BATCH,
    IDX_KEY_NON_CONTIG,
    LAYOUT_COMBOS,
    POOL_SIZE_CHOICES,
    MASK_MODE_CHOICES,
    _adjust_topk_pool,
    _gen_block_size,
)

# ---- 量化专属规模约束(CPU 金标速度 + FP8 近并列控制) ----
MAX_ELEMENTS_QUANT = 4_000_000
MAX_ATTENTION_ELEMENTS_QUANT = 1_000_000
SEQ_MAX_QUANT = 2048  # 池数上限
BATCH_MAX_QUANT = 16
S1_MAX_QUANT = 256

S1_CHOICES_QUANT = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
N1_CHOICES_QUANT = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]  # 64 = N1_LIMIT 极值
TOPK_LARGE_QUANT = [3072, 4096, 8192]

# descale 取值范围(与 pytest 量化套件一致)
DESCALE_FP32_RANGE = [0.25, 3.25]  # mode=0 正 scale
E8M0_RAW_RANGE = [120, 136]  # mode=1 raw 编码 -> 2^(raw-127)
# mxFP8 尾维 [headDim/64, 2](headDim=128)
MX_SCALE_TAIL = [HEAD_DIM // 64, 2]

# 场景标志触发概率(与非量化一致)
P_UNEQUAL = 0.25
P_ZERO_Q = 0.2
P_ZERO_POOL = 0.2
P_NON_CONTIG = 0.25


def _gen_topk_quant():
    # 10% 概率小 topk(ps=1, topk<=16, 覆盖 sparse 极小分桶, 与非量化对齐)
    if random.random() < 0.1:
        return random.choice([1, 2, 4, 8, 16])
    if random.random() < 0.5:
        return random.randint(1, 2048)
    return random.choice(TOPK_LARGE_QUANT)


def _gen_s1():
    return random.choice(S1_CHOICES_QUANT)


def _clamp_n1_quant(s1, head_dim):
    max_n1 = max(1, MAX_ELEMENTS_QUANT // (s1 * head_dim))
    valid = [n for n in N1_CHOICES_QUANT if n <= max_n1]
    return random.choice(valid) if valid else 1


def _clamp_b_quant(s1, n1, head_dim):
    max_b = max(1, min(BATCH_MAX_QUANT, MAX_ELEMENTS_QUANT // (s1 * n1 * head_dim)))
    return random.randint(1, max_b)


def _clamp_s2_quant(s1, b, n2, head_dim, pool_size):
    """池数: S2 >= ceil(S1/pool_size) 且受量化规模约束."""
    lower = max(1, math.ceil(s1 / pool_size))
    upper = min(
        SEQ_MAX_QUANT, max(lower, MAX_ELEMENTS_QUANT // max(1, b * n2 * head_dim))
    )
    attn_upper = max(lower, MAX_ATTENTION_ELEMENTS_QUANT // max(1, s1 * pool_size))
    upper = min(upper, attn_upper)
    return random.randint(lower, upper)


@GENERATOR_REGISTRY.register("pki_quant_gen")
class PoolKeyIndexerQuantGenerator(CaseGenerator):
    """PoolKeyIndexer 量化场景参数约束生成器。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        inputs = case_config.inputs

        # 1. quant_mode 随机(0/1), descale dtype/range 随之确定
        quant_mode = random.choice([0, 1])
        descale_dtype = "fp32" if quant_mode == 0 else "uint8"
        descale_range = (
            list(DESCALE_FP32_RANGE) if quant_mode == 0 else list(E8M0_RAW_RANGE)
        )

        # 2. query/poolKey 固定 fp8e4m3; weights 独立 FP16/BF16
        inputs[IDX_QUERY].dtype = "fp8e4m3"
        inputs[IDX_POOL_KEY].dtype = "fp8e4m3"
        inputs[IDX_WEIGHTS].dtype = random.choice(["fp16", "bf16"])

        # 3. 核心维度(量化规模收敛); 小 topk 分桶强制 ps=1(topk%ps==0 且 topk>=ps)
        s1 = _gen_s1()
        topk_raw = _gen_topk_quant()
        if topk_raw <= 16:
            pool_size = 1
        else:
            pool_size = random.choice(POOL_SIZE_CHOICES)
        topk = _adjust_topk_pool(topk_raw, pool_size)
        n1 = _clamp_n1_quant(s1, HEAD_DIM)
        b = _clamp_b_quant(s1, n1, HEAD_DIM)
        s2 = _clamp_s2_quant(s1, b, N2, HEAD_DIM, pool_size)

        # 4. 布局(全 4 组合)
        layout_query, layout_key = random.choice(LAYOUT_COMBOS)

        # 5. 属性
        inputs[IDX_LAYOUT_Q].range_values = [layout_query]
        inputs[IDX_LAYOUT_K].range_values = [layout_key]
        inputs[IDX_TOPK].range_values = [topk]
        inputs[IDX_POOL_SIZE].range_values = [pool_size]
        inputs[IDX_MASK_MODE].range_values = [random.choice(MASK_MODE_CHOICES)]
        inputs[IDX_QUANT_MODE].range_values = [quant_mode]
        inputs[IDX_RETURN_VALUE].range_values = [random.choice([True, False])]

        # 5.1 场景标志(与非量化同款约束)
        tnd_unequal = (
            b >= 2
            and (layout_query == "TND" or layout_key in ("TND", "PA_BBND"))
            and random.random() < P_UNEQUAL
        )
        zero_q_batch = b >= 2 and layout_query == "TND" and random.random() < P_ZERO_Q
        zero_pool_batch = (
            b >= 2
            and layout_key in ("TND", "PA_BBND")
            and random.random() < P_ZERO_POOL
        )
        key_non_contig = layout_key == "PA_BBND" and random.random() < P_NON_CONTIG
        inputs[IDX_TND_UNEQUAL].range_values = [tnd_unequal]
        inputs[IDX_ZERO_Q_BATCH].range_values = [zero_q_batch]
        inputs[IDX_ZERO_POOL_BATCH].range_values = [zero_pool_batch]
        inputs[IDX_KEY_NON_CONTIG].range_values = [key_non_contig]

        # 6. query / weights shape
        if layout_query == "BSND":
            query_shape = [b, s1, n1, HEAD_DIM]
            weights_shape = [b, s1, n1]
        else:
            t1 = b * s1
            query_shape = [t1, n1, HEAD_DIM]
            weights_shape = [t1, n1]
        inputs[IDX_QUERY].shape = query_shape
        inputs[IDX_WEIGHTS].shape = weights_shape

        # 7. poolKey shape + PA 块参数
        if layout_key == "BSND":
            key_shape = [b, s2, N2, HEAD_DIM]
            block_size = 0
            max_block_num_per_seq = 0
            block_num = 0
        elif layout_key == "TND":
            t2 = b * s2
            key_shape = [t2, N2, HEAD_DIM]
            block_size = 0
            max_block_num_per_seq = 0
            block_num = 0
        else:  # PA_BBND
            block_size = _gen_block_size()
            max_block_num_per_seq = math.ceil(s2 / block_size)
            block_num = b * max_block_num_per_seq
            key_shape = [block_num, block_size, N2, HEAD_DIM]
        inputs[IDX_POOL_KEY].shape = key_shape

        # 8. descale shape/dtype/range(与 tiling 校验一致)
        inputs[IDX_Q_DESCALE].dtype = descale_dtype
        inputs[IDX_K_DESCALE].dtype = descale_dtype
        inputs[IDX_Q_DESCALE].range_values = list(descale_range)
        inputs[IDX_K_DESCALE].range_values = list(descale_range)
        if quant_mode == 0:
            q_descale_shape = list(weights_shape)  # (B,S1,N1)/(T1,N1)
        else:
            q_descale_shape = list(weights_shape) + MX_SCALE_TAIL
        if layout_key == "BSND":
            k_base = [b, s2, N2]
        elif layout_key == "TND":
            k_base = [b * s2, N2]
        else:  # PA: 物理块布局, 与 pool_key 同构
            k_base = [block_num, block_size, N2]
        k_descale_shape = k_base if quant_mode == 0 else k_base + MX_SCALE_TAIL
        inputs[IDX_Q_DESCALE].shape = q_descale_shape
        inputs[IDX_K_DESCALE].shape = k_descale_shape

        # 9. poolTailK
        inputs[IDX_POOL_TAIL_K].shape = [b]
        inputs[IDX_POOL_TAIL_K].range_values = [0, max(0, pool_size - 1)]

        # 10. actualSeqQ / actualSeqK / blockTable(与非量化同款)
        if layout_query == "TND":
            inputs[IDX_ACT_SEQ_Q].shape = [b]
            inputs[IDX_ACT_SEQ_Q].range_values = [s1, b * s1]
        else:
            inputs[IDX_ACT_SEQ_Q].shape = [0]
            inputs[IDX_ACT_SEQ_Q].range_values = []

        if layout_key in ("TND", "PA_BBND"):
            inputs[IDX_ACT_SEQ_K].shape = [b]
            if layout_key == "TND":
                inputs[IDX_ACT_SEQ_K].range_values = [s2, b * s2]
            else:
                inputs[IDX_ACT_SEQ_K].range_values = [s2, s2]
        else:
            inputs[IDX_ACT_SEQ_K].shape = [0]
            inputs[IDX_ACT_SEQ_K].range_values = []

        if layout_key == "PA_BBND":
            inputs[IDX_BLOCK_TABLE].shape = [b, max_block_num_per_seq]
            inputs[IDX_BLOCK_TABLE].range_values = [0, max(0, block_num - 1)]
        else:
            inputs[IDX_BLOCK_TABLE].shape = [0]
            inputs[IDX_BLOCK_TABLE].range_values = []

        return case_config
