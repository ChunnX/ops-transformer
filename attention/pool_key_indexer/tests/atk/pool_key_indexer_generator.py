#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PoolKeyIndexer ATK 用例参数约束生成器。

职责:
  1. 从 ATK 框架生成的 query 一维 shape 中读取 S1
  2. 基于 S1 生成 S2 (pool 数), 保证 S2 >= S1 / poolSize
  3. 根据 layout 组合与 dtype 约束, 动态生成 query/poolKey/weights 的正确多维 shape
  4. 设置 actualSeqQ/actualSeqK/blockTable/poolTailK 的 shape
  5. 随机生成 B、poolSize、topk 等参数, 保证 topk % poolSize == 0

约束来源:
  - op_host/pool_key_indexer_tiling.cpp
  - op_host/pool_key_indexer_def.cpp
  - docs/aclnnKpoolIndexer.md
"""

import random
import math

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


HEAD_DIM = 128
N2 = 1
MAX_ELEMENTS = 20_000_000
MAX_ATTENTION_ELEMENTS = 5_000_000
SEQ_MAX = 8192

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

LAYOUT_COMBOS = [
    ("BSND", "BSND"),
    ("BSND", "PA_BBND"),
    ("TND", "TND"),
]

N1_CHOICES = [1, 2, 4, 8, 16, 24, 32]

TOPK_LARGE = [3072, 4096, 5120, 6144, 7168, 8192]

POOL_SIZE_CHOICES = [1, 2, 4, 8, 16, 32, 64, 128]

MASK_MODE_CHOICES = [0, 3]


def _gen_topk():
    if random.random() < 0.5:
        return random.randint(1, 2048)
    return random.choice(TOPK_LARGE)


def _gen_pool_size():
    return random.choice(POOL_SIZE_CHOICES)


def _adjust_topk_pool(topk, pool_size):
    """保证 topk % pool_size == 0, 向下取整."""
    topk = (topk // pool_size) * pool_size
    if topk == 0:
        topk = pool_size
    return topk


def _gen_block_size():
    return random.randint(1, 64) * 16


def _clamp_n1(s1, head_dim):
    max_n1 = max(1, MAX_ELEMENTS // (s1 * head_dim))
    valid = [n for n in N1_CHOICES if n <= max_n1]
    return random.choice(valid) if valid else 1


def _clamp_b(s1, n1, head_dim, max_b=64):
    max_b_by_mem = max(1, MAX_ELEMENTS // (s1 * n1 * head_dim))
    return random.randint(1, min(max_b, max_b_by_mem))


def _clamp_s2(s1, b, n2, head_dim, pool_size):
    """生成 S2 (pool 数): S2 >= ceil(S1 / pool_size) 且受内存约束."""
    lower = max(1, math.ceil(s1 / pool_size))
    upper = min(SEQ_MAX, max(lower, MAX_ELEMENTS // max(1, b * n2 * head_dim)))
    attn_upper = max(lower, MAX_ATTENTION_ELEMENTS // max(1, s1 * pool_size))
    upper = min(upper, attn_upper)
    return random.randint(lower, upper)


@GENERATOR_REGISTRY.register("pki_gen")
class PoolKeyIndexerGenerator(CaseGenerator):
    """PoolKeyIndexer 参数约束生成器。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        inputs = case_config.inputs

        # 1. dtype 约束: query/poolKey/weights 同 dtype
        query_dtype = inputs[IDX_QUERY].dtype
        inputs[IDX_POOL_KEY].dtype = query_dtype
        inputs[IDX_WEIGHTS].dtype = query_dtype

        # 2. 读取 S1
        s1 = max(1, inputs[IDX_QUERY].shape[0])

        # 3. 随机生成核心维度参数
        pool_size = _gen_pool_size()
        topk = _adjust_topk_pool(_gen_topk(), pool_size)
        n1 = _clamp_n1(s1, HEAD_DIM)
        b = _clamp_b(s1, n1, HEAD_DIM, max_b=64)
        s2 = _clamp_s2(s1, b, N2, HEAD_DIM, pool_size)

        # 4. 随机选取 layout 组合
        layout_query, layout_key = random.choice(LAYOUT_COMBOS)

        # 5. 设置属性
        inputs[IDX_LAYOUT_Q].range_values = [layout_query]
        inputs[IDX_LAYOUT_K].range_values = [layout_key]
        inputs[IDX_TOPK].range_values = [topk]
        inputs[IDX_POOL_SIZE].range_values = [pool_size]

        mask_mode = random.choice(MASK_MODE_CHOICES)
        inputs[IDX_MASK_MODE].range_values = [mask_mode]

        inputs[IDX_QUANT_MODE].range_values = [-1]

        if layout_key == "PA_BBND":
            return_value = False
        else:
            return_value = random.choice([True, False])
        inputs[IDX_RETURN_VALUE].range_values = [return_value]

        # 6. 根据 layout 生成 query/poolKey/weights shape
        if layout_query == "BSND":
            query_shape = [b, s1, n1, HEAD_DIM]
            weights_shape = [b, s1, n1]
        else:
            t1 = b * s1
            query_shape = [t1, n1, HEAD_DIM]
            weights_shape = [t1, n1]

        inputs[IDX_QUERY].shape = query_shape
        inputs[IDX_WEIGHTS].shape = weights_shape

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
            max_block_num_per_seq = math.ceil(s2 * pool_size / block_size)
            block_num = b * max_block_num_per_seq
            key_shape = [block_num, block_size, N2, HEAD_DIM]

        inputs[IDX_POOL_KEY].shape = key_shape

        # 7. poolTailK: shape=[B], range=[0, poolSize-1]
        inputs[IDX_POOL_TAIL_K].shape = [b]
        inputs[IDX_POOL_TAIL_K].range_values = [0, max(0, pool_size - 1)]

        # 8. actualSeqQ
        if layout_query == "TND":
            inputs[IDX_ACT_SEQ_Q].shape = [b]
            t1 = b * s1
            inputs[IDX_ACT_SEQ_Q].range_values = [s1, t1]
        else:
            inputs[IDX_ACT_SEQ_Q].shape = [0]
            inputs[IDX_ACT_SEQ_Q].range_values = []

        # 9. actualSeqK (值是 pool 数)
        if layout_key in ("TND", "PA_BBND"):
            inputs[IDX_ACT_SEQ_K].shape = [b]
            if layout_key == "TND":
                t2 = b * s2
                inputs[IDX_ACT_SEQ_K].range_values = [s2, t2]
            else:
                inputs[IDX_ACT_SEQ_K].range_values = [s2, s2]
        else:
            inputs[IDX_ACT_SEQ_K].shape = [0]
            inputs[IDX_ACT_SEQ_K].range_values = []

        # 10. blockTable (仅 PA_BBND)
        if layout_key == "PA_BBND":
            inputs[IDX_BLOCK_TABLE].shape = [b, max_block_num_per_seq]
            inputs[IDX_BLOCK_TABLE].range_values = [0, max(0, block_num - 1)]
        else:
            inputs[IDX_BLOCK_TABLE].shape = [0]
            inputs[IDX_BLOCK_TABLE].range_values = []

        # 11. qDescale / kDescale: 非量化场景为空
        inputs[IDX_Q_DESCALE].shape = [0]
        inputs[IDX_Q_DESCALE].range_values = []
        inputs[IDX_K_DESCALE].shape = [0]
        inputs[IDX_K_DESCALE].range_values = []

        return case_config
