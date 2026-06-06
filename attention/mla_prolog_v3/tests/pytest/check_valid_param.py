#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import math
import random
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)


def validate_config(params):
    batch_size, He, Hcq, Hckv, q_head_num, kv_head_num, head_dim, rope_head_dim, \
        q_seq, kv_seq, block_size, input_layout, cache_mode, cq_epsilon, ckv_epsilon, dtype, \
        weight_quant_mode, kv_cache_quant_mode, query_quant_mode, \
        ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, query_norm_flag = params

    if batch_size > 65536 or batch_size < 1:
        raise ValueError("batch_size must be <= 65536 and >= 1")
    if q_seq > 16 or q_seq < 1:
        raise ValueError("q_seq must be <= 16 and >= 1")
    if He != 7168:
        raise ValueError("He must = 7168")
    if Hcq != 1536:
        raise ValueError("Hcq must = 1536")
    if q_head_num > 256:
        raise ValueError("q_head_num must be <= 256")
    if kv_head_num > 256:
        raise ValueError("kv_head_num must be <= 256")
    if q_head_num % kv_head_num != 0:
        raise ValueError("q_head_num must be integral multiple of kv_head_num")
    if Hckv != 512:
        raise ValueError("Hckv must = 512")
    if head_dim != 128:
        raise ValueError("head_dim must = 128")
    if rope_head_dim != 64:
        raise ValueError("rope_head_dim must = 64")
    if kv_head_num != 1:
        raise ValueError("kv_head_num must = 1")
    if block_size not in (16, 128):
        raise ValueError("block_size must = 16 or 128")
    if dtype not in [torch.bfloat16, torch.float16, torch.int8]:
        raise ValueError("dtype should be: float16/bfloat16/int8")
    if input_layout not in ["BSH", "BSND", "BNSD"]:
        raise ValueError("input_layout should be: BSH/BSND/BNSD")
    if weight_quant_mode not in range(6):
        raise ValueError("weight_quant_mode must be 0-5")
    if kv_cache_quant_mode not in range(4):
        raise ValueError("kv_cache_quant_mode must be 0-3")
    if query_quant_mode not in range(2):
        raise ValueError("query_quant_mode must be 0 or 1")
    if ckvkr_repo_mode not in range(2):
        raise ValueError("ckvkr_repo_mode must be 0 or 1")
    if quant_scale_repo_mode not in range(2):
        raise ValueError("quant_scale_repo_mode must be 0 or 1")
    if tile_size not in (128,):
        raise ValueError("tile_size must = 128")

    if weight_quant_mode == 0 and kv_cache_quant_mode != 0:
        raise ValueError("weight_quant_mode=0 requires kv_cache_quant_mode=0")
    if weight_quant_mode == 1 and kv_cache_quant_mode not in (0, 2, 3):
        raise ValueError("weight_quant_mode=1 requires kv_cache_quant_mode in (0,2,3)")
    if weight_quant_mode == 2 and kv_cache_quant_mode not in (0, 1, 3):
        raise ValueError("weight_quant_mode=2 requires kv_cache_quant_mode in (0,1,3)")
    if weight_quant_mode == 3 and kv_cache_quant_mode not in (0, 1, 3):
        raise ValueError("weight_quant_mode=3 requires kv_cache_quant_mode in (0,1,3)")
    if weight_quant_mode in (4, 5) and kv_cache_quant_mode not in (0, 1, 3):
        raise ValueError("weight_quant_mode=4/5 requires kv_cache_quant_mode in (0,1,3)")


def create_random_block_table(batch_size, act_seq_len_kv, block_size):
    max_kv_seq_length = max(act_seq_len_kv)
    valid_block_num = [math.ceil(act_seq_len_kv[i] / block_size) for i in range(batch_size)]
    valid_block_num_sum = sum(valid_block_num)
    max_valid_block_num = math.ceil(max_kv_seq_length / block_size)
    block_table = -torch.ones(batch_size, max_valid_block_num, dtype=torch.int32)
    block_idx = list(range(valid_block_num_sum))
    random.shuffle(block_idx)
    block_idx_start = 0
    for i in range(batch_size):
        block_table[i][:valid_block_num[i]] = torch.tensor(
            block_idx[block_idx_start: block_idx_start + valid_block_num[i]],
            dtype=torch.int32)
        block_idx_start += valid_block_num[i]
    return block_table, valid_block_num_sum


def check_result(expect, result):
    thres = 0.005
    num_outputs = min(len(expect), len(result))
    for i in range(num_outputs):
        if expect[i] is None or result[i] is None:
            if expect[i] is None and result[i] is None:
                continue
            elif expect[i] is None:
                logger.info("output[%d] golden is None, NPU has shape=%s", i, result[i].shape if hasattr(result[i], 'shape') else 'tensor')
            else:
                logger.info("output[%d] NPU is None, golden has shape=%s", i, expect[i].shape)
            continue
        r = result[i].cpu().float()
        e = expect[i].cpu().float()
        diff = (r.reshape(-1) - e.reshape(-1)).abs()
        idx = torch.nonzero(diff > thres).squeeze()
        if idx.numel() > 0:
            logger.info("output[%d] >%s indices: %s", i, thres, idx[:10])
            logger.info("output[%d] >%s diff values: %s", i, thres, diff[idx][:10])
        pass_num = torch.sum(diff <= thres)
        total_num = diff.numel()
        acc = pass_num / total_num if total_num > 0 else 0.0
        logger.info("output[%d] pass: %d/%d accuracy: %.4f", i, pass_num, total_num, acc)
        assert torch.allclose(r.flatten(), e.flatten(), rtol=0.05, atol=0.05), \
            "output[{}] mismatch".format(i)
