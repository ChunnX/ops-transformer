# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 小算子拼接实现：将 recurrent_gated_delta_rule 递归计算拆解为 matmul / mul / add / sub 等
# 基础算子，在 head 维度向量化（bmm），仅在时间步上循环。
# 参考 vllm-ascend 中 GDN 算子的分解方式：
#   1. 门控计算: alpha = exp(g), beta_out = sigmoid(b)
#   2. 状态衰减: S = S * alpha
#   3. 键门控:   S = S * alphak  (可选)
#   4. 状态-键乘: x = S @ k
#   5. Delta更新: delta = (v - x) * beta
#   6. 状态更新:  S = S + outer(delta, k)
#   7. 输出计算:  o = S @ q
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
import numpy as np
import logging
import os

from recurrent_gated_delta_rule_golden import (
    rand_range, adjust_range, check_result, print_log,
)

DEVICE_ID = 0
logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)


def small_ops_recurrent_gated_delta_rule(
    q, k, v, state, beta, scale_value, act_seq_len, ssm_state_indices,
    num_accepted_tokens=None, g=None, gk=None):
    """小算子拼接参考实现：用 bmm/mul/add/sub 等基础算子拼接 GDN 递归。

    与 cpu_recurrent_gated_delta_rule 算法等价，但在 head 维度向量化，
    每个时间步处理所有 head，仅按时间步顺序循环。

    Args:
        q:    [T, Nk, Dk]
        k:    [T, Nk, Dk]
        v:    [T, Nv, Dv]
        state:[block_num, Nv, Dv, Dk]  (原地更新)
        beta: [T, Nv]
        scale_value: float
        act_seq_len:  [B] int32
        ssm_state_indices: [T] int32
        num_accepted_tokens: [B] int32 or None
        g:  [T, Nv] float32 (log space, 需 exp) or None
        gk: [T, Nv, Dk] float32 (log space, 需 exp) or None

    Returns:
        output: [T, Nv, Dv]
        state:  更新后的 state
    """
    T, n_heads_qk, Dk = q.shape
    _, n_heads_v, Dv = v.shape
    B = act_seq_len.shape[0]

    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    beta = beta.to(torch.float32)
    state_work = state.to(torch.float32).clone()
    output = torch.empty(T, n_heads_v, Dv, dtype=torch.float32, device=v.device)

    # 门控预处理：alpha = exp(g), beta 保持原值
    if g is None:
        alpha = torch.ones(T, n_heads_v, dtype=torch.float32, device=v.device)
    else:
        alpha = g.to(torch.float32).exp()

    if gk is None:
        alphak = torch.ones(T, n_heads_v, Dk, dtype=torch.float32, device=v.device)
    else:
        alphak = gk.to(torch.float32).exp()

    # scale
    q = q * scale_value

    # GQA: 将 q/k 从 [T, Nk, Dk] 广播到 [T, Nv, Dk]
    if n_heads_v > n_heads_qk:
        rep = n_heads_v // n_heads_qk
        q = q.repeat_interleave(rep, dim=1)
        k = k.repeat_interleave(rep, dim=1)

    seq_start = 0
    for i in range(B):
        seq_len_i = int(act_seq_len[i])
        if seq_len_i <= 0:
            continue

        # 确定 init state 的索引
        if num_accepted_tokens is None:
            init_idx = int(ssm_state_indices[seq_start])
        else:
            accepted = int(num_accepted_tokens[i])
            init_idx = int(ssm_state_indices[seq_start + accepted - 1])

        # S: [Nv, Dv, Dk]
        S = state_work[init_idx].clone()

        for slot in range(seq_len_i):
            t = seq_start + slot
            # 取当前 token 的数据，shape 已对齐到 Nv
            q_t = q[t]       # [Nv, Dk]
            k_t = k[t]       # [Nv, Dk]
            v_t = v[t]       # [Nv, Dv]
            beta_t = beta[t]  # [Nv]
            alpha_t = alpha[t]  # [Nv]
            alphak_t = alphak[t]  # [Nv, Dk]

            # 1. 状态衰减: S = S * alpha  (alpha: [Nv] -> [Nv, 1, 1])
            S = S * alpha_t[:, None, None]

            # 2. 键门控: S = S * alphak  (alphak: [Nv, Dk] -> [Nv, 1, Dk])
            S = S * alphak_t[:, None, :]

            # 3. 状态-键乘: x = S @ k^T  -> [Nv, Dv]
            #    bmm([Nv, Dv, Dk], [Nv, Dk, 1]) -> [Nv, Dv, 1] -> squeeze
            x = torch.bmm(S, k_t.unsqueeze(-1)).squeeze(-1)

            # 4. Delta 更新: delta = (v - x) * beta
            delta = (v_t - x) * beta_t[:, None]

            # 5. 状态更新: S = S + outer(delta, k)
            #    outer: [Nv, Dv, 1] @ [Nv, 1, Dk] -> [Nv, Dv, Dk]
            S = S + torch.bmm(delta.unsqueeze(-1), k_t.unsqueeze(-2))

            # 6. 输出计算: o = S @ q^T  -> [Nv, Dv]
            o_t = torch.bmm(S, q_t.unsqueeze(-1)).squeeze(-1)

            # 写回 state 和 output
            state_idx = int(ssm_state_indices[t])
            state_work[state_idx] = S
            output[t] = o_t

        seq_start += seq_len_i

    output_golden = output.to(state.dtype)
    state_golden = state_work.to(state.dtype)

    return output_golden, state_golden


def run_recurrent_gated_delta_rule_small_ops(
    B, mtp, nk, nv, dk, dv, actual_seq_lengths=None,
    ssm_state_indices=None, has_gamma='False', has_gamma_k='False',
    has_num_accepted_tokens='False', scale_value=None, num_accepted_tokens=None,
    block_num=None, data_type=torch.bfloat16, query_datarange=[-10, 10], key_datarange=[-10, 10],
    value_datarange=[-10, 10], gamma_datarange=[0, 1], gamma_k_datarange=[0, 1],
    beta_datarange=[0, 1], state_datarange=[-10, 10]):
    """小算子拼接参考实现测试入口：生成数据 -> 小算子参考 -> NPU 算子 -> 精度对比。"""
    torch_npu.npu.set_device(int(DEVICE_ID))

    block_num = B * mtp if block_num is None else block_num
    if scale_value is None:
        scale_value = dk ** -0.5
    if actual_seq_lengths is None:
        actual_seq_lengths = [mtp] * B
    if has_num_accepted_tokens == True and num_accepted_tokens is None:
        num_accepted_tokens = torch.tensor([torch.randint(0, h, (1,)) for h in actual_seq_lengths]) + 1
    T = int(sum(actual_seq_lengths))
    actual_seq_lengths = torch.tensor(actual_seq_lengths)
    if ssm_state_indices is None:
        ssm_state_indices = torch.arange(T, dtype=torch.int32)

    # ======================== gen input data start =============================
    query = rand_range((T, nk, dk), query_datarange, data_type)
    key = rand_range((T, nk, dk), key_datarange, data_type)
    value = rand_range((T, nv, dv), value_datarange, data_type)
    g = rand_range((T, nv), adjust_range(gamma_datarange), dtype=torch.float32) if has_gamma == True else None
    gk = rand_range((T, nv, dk), adjust_range(gamma_k_datarange), dtype=torch.float32) if has_gamma_k == True else None
    beta = rand_range((T, nv), beta_datarange, data_type)
    num_accepted_tokens = torch.tensor(num_accepted_tokens, dtype=torch.int32) if has_num_accepted_tokens == True else None
    state = rand_range((block_num, nv, dv, dk), state_datarange, data_type)
    act_seq_len = torch.tensor(actual_seq_lengths, dtype=torch.int32)
    ssm_state_indices = torch.tensor(ssm_state_indices, dtype=torch.int32)

    # 对于 query/key 数据范围大于 [-1, 1] 的情况加入归一化处理
    if query_datarange[0] < -1 or query_datarange[1] > 1:
        query = torch.nn.functional.normalize(query, p=2, dim=-1)
    if key_datarange[0] < -1 or key_datarange[1] > 1:
        key = torch.nn.functional.normalize(key, p=2, dim=-1)
    # ======================== gen input data finish =============================

    # ======================== execute small-ops reference (CPU) ==================
    cpu_out, cpu_state_output = small_ops_recurrent_gated_delta_rule(
        query, key, value, state, beta, scale_value, act_seq_len, ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens, g=g, gk=gk)
    # ======================== execute small-ops reference finish ================

    # ======================== execute npu start ================================
    query_npu = query.to("npu:%s" % DEVICE_ID)
    key_npu = key.to("npu:%s" % DEVICE_ID)
    value_npu = value.to("npu:%s" % DEVICE_ID)
    state_npu = state.to("npu:%s" % DEVICE_ID)
    beta_npu = beta.to("npu:%s" % DEVICE_ID)
    act_seq_len_npu = act_seq_len.to("npu:%s" % DEVICE_ID)
    ssm_state_indices_npu = ssm_state_indices.to("npu:%s" % DEVICE_ID)
    num_accepted_tokens_npu = num_accepted_tokens.to("npu:%s" % DEVICE_ID) if has_num_accepted_tokens == True else None
    g_npu = g.to("npu:%s" % DEVICE_ID) if has_gamma == True else None
    gk_npu = gk.to("npu:%s" % DEVICE_ID) if has_gamma_k == True else None

    init_state = state_npu.clone()
    npu_out = torch_npu.npu_recurrent_gated_delta_rule(
        query_npu, key_npu, value_npu, init_state, beta=beta_npu, scale=scale_value,
        actual_seq_lengths=act_seq_len_npu, ssm_state_indices=ssm_state_indices_npu,
        num_accepted_tokens=num_accepted_tokens_npu, g=g_npu, gk=gk_npu)
    npu_state_out = init_state
    # ======================== execute npu finish ===============================

    print(f"[small_ops] query: {query.shape}, key: {key.shape}, value: {value.shape}, "
          f"state: {state.shape}, beta: {beta.shape}")
    print(f"[small_ops] act_seq_len: {act_seq_len.shape[0]}, ssm_state_indices: {ssm_state_indices.shape[0]}")

    # ======================== check result =====================================
    data_type_str = str(npu_out.dtype)
    print("-------------------------------------- check output (small_ops vs npu) --------------------------------------")
    check_result(cpu_out.to(torch.float32), npu_out.cpu().to(torch.float32), data_type_str)
    print("-------------------------------------- check state output (small_ops vs npu) ---------------------------------")
    check_result(cpu_state_output.to(torch.float32), npu_state_out.cpu().to(torch.float32), data_type_str)
