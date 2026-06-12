#!/usr/bin/python3
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
非量化 PA 非连续 kvcache 测试

验证 fused_infer_attention_score 算子对传入非连续 KV Cache 时的 stride 传递正确性。
构造方式：stack k/v 在 Bn 维上拼接后 slice 出非连续 tensor，与连续 tensor 的 NPU 结果做对比。
"""

import logging
import math
import random

import torch
import torch_npu

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)

# ==============================================================================
# 配置区
# ==============================================================================
B = 1
N_q = 2
N_kv = 2
D = 128
Q_SEQ = 4
KV_SEQ = 512
BLOCK_SIZE = 512
IN_LAYOUT = "BNSD"
CACHE_LAYOUT = "BNBD"   # [BlockNum, N_kv, BlockSize, D]
DTYPE = torch.float16
IS_CONTIGUOUS = False   # False: 构造非连续kvcache做验证
DEVICE_ID = 0


def create_random_block_table(batch_size, act_seq_len_kv, block_size):
    max_kv_seq = max(act_seq_len_kv)
    valid_block_nums = [math.ceil(s / block_size) for s in act_seq_len_kv]
    total_blocks = sum(valid_block_nums)
    max_blocks = max(valid_block_nums)

    block_idx_list = list(range(total_blocks))
    random.shuffle(block_idx_list)
    block_table = -torch.ones(batch_size, max_blocks, dtype=torch.int32)
    idx = 0
    for b in range(batch_size):
        for j in range(valid_block_nums[b]):
            block_table[b, j] = block_idx_list[idx]
            idx += 1
    return block_table, total_blocks


def cpu_pa_golden(query, k_cache, v_cache, block_table, act_seq_q, act_seq_kv, scale):
    """
    CPU PA golden: 按 block_table 从 KV Cache 取 block，逐 token 计算 attention.
    K/V Cache: [total_blocks, N_kv, block_size, D] (BNBD)
    Q:         [B, N_q, Sq, D] (BNSD)
    """
    group_size = N_q // N_kv
    Sq = query.shape[2]
    D_val = v_cache.shape[-1]

    out = torch.zeros(B, N_q, Sq, D_val, dtype=torch.float32)

    for b in range(B):
        q_len = act_seq_q[b]
        kv_len = act_seq_kv[b]
        bt = block_table[b]
        valid_blocks = bt[bt != -1]

        for q_head in range(N_q):
            kv_head = q_head // group_size

            # 收集当前 sample 所有有效 block 的 k/v
            all_keys = []
            all_vals = []
            for blk_id in valid_blocks:
                all_keys.append(k_cache[int(blk_id), kv_head])  # [block_size, D]
                all_vals.append(v_cache[int(blk_id), kv_head])  # [block_size, D]

            big_k = torch.cat(all_keys, dim=0)[:kv_len]  # [kv_len, D]
            big_v = torch.cat(all_vals, dim=0)[:kv_len]  # [kv_len, D]

            for t in range(q_len):
                q_vec = query[b, q_head, t]  # [D]
                scores = torch.matmul(q_vec.float(), big_k.float().T) * scale
                attn = torch.softmax(scores, dim=-1)
                out[b, q_head, t] = torch.matmul(attn, big_v.float())

    return out.to(DTYPE)


def run_npu(query, k_cache, v_cache, block_table, act_seq_q, act_seq_kv, scale):
    """调用 NPU 算子"""
    torch_npu.npu.set_device(DEVICE_ID)
    torch.npu.synchronize()
    result, _ = torch_npu.npu_fused_infer_attention_score(
        query, k_cache, v_cache,
        actual_seq_lengths=act_seq_q,
        actual_seq_lengths_kv=act_seq_kv,
        num_heads=N_q,
        num_key_value_heads=N_kv,
        input_layout=IN_LAYOUT,
        scale=scale,
        pre_tokens=65535,
        next_tokens=65535,
        block_table=block_table,
        block_size=BLOCK_SIZE,
    )
    torch.npu.synchronize()
    return result


def check_result(expect, result, tag=""):
    diff = (result.cpu().float() - expect.cpu().float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    thres = 0.05
    pass_count = (diff <= thres).sum().item()
    total = diff.numel()
    logger.info(f"[{tag}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
                f"pass={pass_count}/{total}, ratio={pass_count/total:.4f}")
    assert torch.allclose(result.cpu().float(), expect.cpu().float(), rtol=0.05, atol=0.05), \
        f"[{tag}] FAILED: max_diff={max_diff}"


def test_pa_contiguous():
    """基线：连续 KV Cache"""
    act_seq_q = [Q_SEQ] * B
    act_seq_kv = [KV_SEQ] * B

    block_table_torch, total_blocks = create_random_block_table(B, act_seq_kv, BLOCK_SIZE)
    block_table_npu = block_table_torch.npu()

    scale = 1.0 / math.sqrt(D)

    torch.manual_seed(42)
    query = torch.randn(B, N_q, Q_SEQ, D, dtype=DTYPE).npu()

    k_cache = torch.randn(total_blocks, N_kv, BLOCK_SIZE, D, dtype=DTYPE).npu()
    v_cache = torch.randn(total_blocks, N_kv, BLOCK_SIZE, D, dtype=DTYPE).npu()

    cpu_out = cpu_pa_golden(
        query.cpu(), k_cache.cpu(), v_cache.cpu(), block_table_torch,
        act_seq_q, act_seq_kv, scale
    )

    npu_out = run_npu(query, k_cache, v_cache, block_table_npu, act_seq_q, act_seq_kv, scale)

    check_result(cpu_out, npu_out, "PA-contiguous")


def test_pa_noncontiguous():
    """验证：非连续 KV Cache（stack k/v 后 slice）"""
    if not IS_CONTIGUOUS:
        logger.info("IS_CONTIGUOUS=False, 构造非连续 kvcache")
    act_seq_q = [Q_SEQ] * B
    act_seq_kv = [KV_SEQ] * B

    block_table_torch, total_blocks = create_random_block_table(B, act_seq_kv, BLOCK_SIZE)
    block_table_npu = block_table_torch.npu()

    scale = 1.0 / math.sqrt(D)

    torch.manual_seed(42)
    query = torch.randn(B, N_q, Q_SEQ, D, dtype=DTYPE).npu()

    # ==========================
    # 连续版本：直接创建连续tensor转NPU
    # ==========================
    k_cache_cpu = torch.randn(total_blocks, N_kv, BLOCK_SIZE, D, dtype=DTYPE)
    v_cache_cpu = torch.randn(total_blocks, N_kv, BLOCK_SIZE, D, dtype=DTYPE)

    k_cont_npu = k_cache_cpu.npu()
    v_cont_npu = v_cache_cpu.npu()
    npu_out_cont = run_npu(query, k_cont_npu, v_cont_npu, block_table_npu, act_seq_q, act_seq_kv, scale)

    if IS_CONTIGUOUS:
        logger.info("[INFO] IS_CONTIGUOUS=True, skip non-contiguous test")
        return

    # ==========================
    # 非连续版本：在dim=1上垫一行padding后slice，保证dim0 stride错误从而强制非连续
    # k_padded: [Bn, N_kv+1, Bs, D], dim0 stride = (N_kv+1)*Bs*D
    # slice [:,:N_kv]: [Bn, N_kv, Bs, D], dim0 stride 仍为 (N_kv+1)*Bs*D
    # 但连续[Bn, N_kv, Bs, D]的 dim0 stride 应为 N_kv*Bs*D
    # (N_kv+1)*Bs*D ≠ N_kv*Bs*D → 必定非连续
    # ==========================
    k_padded = torch.randn(total_blocks, N_kv + 1, BLOCK_SIZE, D, dtype=DTYPE)
    v_padded = torch.randn(total_blocks, N_kv + 1, BLOCK_SIZE, D, dtype=DTYPE)
    k_padded[:, :N_kv] = k_cache_cpu
    v_padded[:, :N_kv] = v_cache_cpu

    k_noncont_cpu = k_padded[:, :N_kv, :, :]
    v_noncont_cpu = v_padded[:, :N_kv, :, :]

    logger.info(f"[CPU] k is_contiguous: {k_noncont_cpu.is_contiguous()}")
    logger.info(f"[CPU] v is_contiguous: {v_noncont_cpu.is_contiguous()}")
    logger.info(f"[CPU] k shape: {k_noncont_cpu.shape}, stride: {k_noncont_cpu.stride()}")
    logger.info(f"[CPU] v shape: {v_noncont_cpu.shape}, stride: {v_noncont_cpu.stride()}")

    assert not k_noncont_cpu.is_contiguous(), "k should be non-contiguous on CPU!"
    assert not v_noncont_cpu.is_contiguous(), "v should be non-contiguous on CPU!"

    # 非连续CPU tensor转NPU，stride应被保留
    k_noncont_npu = k_noncont_cpu.npu()
    v_noncont_npu = v_noncont_cpu.npu()

    logger.info(f"[NPU] k is_contiguous: {k_noncont_npu.is_contiguous()}")
    logger.info(f"[NPU] v is_contiguous: {v_noncont_npu.is_contiguous()}")

    npu_out_noncont = run_npu(query, k_noncont_npu, v_noncont_npu, block_table_npu,
                              act_seq_q, act_seq_kv, scale)

    # 非连续版本与连续版本结果应一致（物理数据相同，仅stride不同）
    check_result(npu_out_cont, npu_out_noncont, "PA-noncontiguous-vs-contiguous")


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("非量化 PA KVCache 非连续测试")
    logger.info(f"B={B}, N_q={N_q}, N_kv={N_kv}, D={D}, QS={Q_SEQ}, KVS={KV_SEQ}")
    logger.info(f"LAYOUT={IN_LAYOUT}, CACHE={CACHE_LAYOUT}, BLOCK_SIZE={BLOCK_SIZE}")
    logger.info(f"IS_CONTIGUOUS={IS_CONTIGUOUS}")
    logger.info("=" * 60)

    logger.info("\n[Test 1] PA 连续 — 基线")
    test_pa_contiguous()

    logger.info("\n[Test 2] PA 非连续 — stride 传递验证")
    test_pa_noncontiguous()

    logger.info("\n全部测试通过!")
