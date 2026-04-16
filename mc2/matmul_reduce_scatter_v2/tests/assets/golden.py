# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
__golden__ = {
        "kernel": {
            "matmul_reduce_scatter_v2": "matmul_reduce_scatter_v2_golden"
        }
}

import copy

import numpy as np
import torch
import torch.distributed as dist


# --- 以下与 MC2_test/script/func.py、common.py 对齐，供 CPU 参考复现 get_cpu ---


def _scale_generate(fp32_deq_scale):
    uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32)
    uint32_deq_scale &= 0xFFFFE000
    return np.frombuffer(uint32_deq_scale, np.float32)


def _fetch_batch_broadcast(batch_x1, batch_x2):
    batch_out = copy.deepcopy(batch_x1) if len(batch_x1) > len(batch_x2) else copy.deepcopy(batch_x2)
    min_len, max_len = 0, 0
    if batch_x2 != batch_x1 and batch_x1 and batch_x2:
        min_len = min(len(batch_x1), len(batch_x2))
        max_len = max(len(batch_x1), len(batch_x2))
    for idx in range(min_len):
        batch_out[-(idx + 1)] = max(batch_x1[-(idx + 1)], batch_x2[-(idx + 1)])
    if len(batch_x1) > len(batch_x2):
        for idx in range(min_len, max_len):
            batch_out[-(idx + 1)] = batch_x1[-(idx + 1)]
    else:
        for idx in range(min_len, max_len):
            batch_out[-(idx + 1)] = batch_x2[-(idx + 1)]
    return batch_out


def _unpack_group_size(group_size):
    if group_size == -1:
        return 0, 0, 0
    group_size_m = (group_size >> 32) & 0xFFFF
    group_size_n = (group_size >> 16) & 0xFFFF
    group_size_k = group_size & 0xFFFF
    return group_size_m, group_size_n, group_size_k


def _value_batch_broadcast(x, batch):
    new_x_shape = batch + list(x.shape[-2:])
    return torch.broadcast_to(x, new_x_shape)


def _per_block_cpu_compute(group_size, x1, x2, x1_scale, x2_scale):
    batch_x1 = np.array(x1.shape[:-2]).astype(int).tolist()
    batch_x2 = np.array(x2.shape[:-2]).astype(int).tolist()
    batch_out = _fetch_batch_broadcast(batch_x1, batch_x2)
    if batch_x1 != batch_out:
        x1 = _value_batch_broadcast(x1, batch_out)
        x1_scale = _value_batch_broadcast(x1_scale, batch_out)
    if batch_x2 != batch_out:
        x2 = _value_batch_broadcast(x2, batch_out)
        x2_scale = _value_batch_broadcast(x2_scale, batch_out)

    batch_all = 1
    is2dim = True
    if batch_out:
        is2dim = False
        batch_all = int(np.prod(batch_out))
        x1 = x1.reshape([batch_all] + list(x1.shape[-2:]))
        x2 = x2.reshape([batch_all] + list(x2.shape[-2:]))
        x1_scale = x1_scale.reshape([batch_all] + list(x1_scale.shape[-2:]))
        x2_scale = x2_scale.reshape([batch_all] + list(x2_scale.shape[-2:]))
    m = x1.shape[-2]
    k = x1.shape[-1]
    n = x2.shape[-1]
    out = torch.zeros(m, n)
    if x2_scale.dim() > 2 or x1_scale.dim() > 2:
        out = torch.zeros(batch_all, m, n)
    group_size_m, group_size_n, group_size_k = _unpack_group_size(group_size)
    for i in range(batch_all):
        for m_idx in range((m + group_size_m - 1) // group_size_m):
            m_start = m_idx * group_size_m
            m_end = min((m_idx + 1) * group_size_m, m)
            for n_idx in range((n + group_size_n - 1) // group_size_n):
                n_start = n_idx * group_size_n
                n_end = min((n_idx + 1) * group_size_n, n)
                for k_idx in range((k + group_size_k - 1) // group_size_k):
                    k_start = k_idx * group_size_k
                    k_end = min((k_idx + 1) * group_size_k, k)
                    if batch_all == 1 and is2dim:
                        block_output = torch.matmul(
                            x1[m_start:m_end, k_start:k_end],
                            x2[k_start:k_end, n_start:n_end],
                        ) * x1_scale[m_idx, k_idx] * x2_scale[k_idx, n_idx]
                        out[m_start:m_end, n_start:n_end] += block_output
                    else:
                        out[i, m_start:m_end, n_start:n_end] += torch.matmul(
                            x1[i, m_start:m_end, k_start:k_end],
                            x2[i, k_start:k_end, n_start:n_end],
                        ) * x1_scale[i, m_idx, k_idx] * x2_scale[i, k_idx, n_idx]
    if x2_scale.dim() > 2 or x1_scale.dim() > 2:
        out_shape = batch_out + [m, n]
        out = torch.reshape(out, out_shape)
    else:
        out = torch.reshape(out, [m, n])
    return out


def _mxfp_cpu_compute(x1, x2, x1scale, x2scale):
    """对齐 MC2_test MC2_OP.mxfp_cpu_compute（numpy 2D 输入）。"""
    for arr in [x1, x2, x1scale, x2scale]:
        if arr.ndim != 2:
            raise ValueError("mxfp_cpu_compute expects 2-D numpy arrays.")
    repeated_x1scale = np.repeat(x1scale, 32, axis=-1)
    repeated_x2scale = np.repeat(x2scale, 32, axis=-2)
    x1_pad_length = repeated_x1scale.shape[-1] - x1.shape[-1]
    x2_pad_len = repeated_x2scale.shape[-2] - x2.shape[-2]
    x1_pad_tuple = [(0, 0)] * (len(x1.shape) - 1) + [(0, x1_pad_length)]
    x2_pad_tuple = [(0, 0)] * (len(x2.shape) - 2) + [(0, x2_pad_len)] + [(0, 0)]
    padded_x1 = np.pad(x1, x1_pad_tuple, mode='constant', constant_values=0)
    padded_x2 = np.pad(x2, x2_pad_tuple, mode='constant', constant_values=0)

    padded_x1 = torch.from_numpy(padded_x1)
    padded_x2 = torch.from_numpy(padded_x2)
    repeated_x1scale = torch.from_numpy(repeated_x1scale)
    repeated_x2scale = torch.from_numpy(repeated_x2scale)
    return torch.matmul(padded_x1 * repeated_x1scale, padded_x2 * repeated_x2scale)


def matmul_reduce_scatter_v2_golden(x1, x2, bias = None, x1_scale = None, x2_scale = None, quant_scale = None, group = None, reduce_op = 'sum', is_trans_a = False, is_trans_b = False, comm_turn = 0, rank_size = 0, block_size = 0, group_size = 0, is_amax_out = False, y_dtype = 28, comm_mode = 'aicpu', **kwargs):
    """
    ReduceScatter(MatMul/量化还原 + bias) 的 CPU float32 参考，对齐 aclnnMatmulReduceScatterV2.get_cpu。

    - 情形1：无有效 x1_scale/x2_scale → fp matmul + bias。
    - 情形2：per-tensor 等 → matmul + bias 后按 MC2 scale_generate 施加双 scale。
    - 情形3：per-block → _per_block_cpu_compute（group_size 为 0 且判定为 per-block 时使用 128³ 打包默认值）。
    - 情形4：mxfp → _mxfp_cpu_compute（scale 可为 3D，内部压成 2D 再算）。
    通信段：all_reduce(SUM) 后按 rank 取行块，与 MatmulReduceScatter.get_cpu 一致。
    """
    del group
    del comm_mode
    del block_size  # tiling 参数，数学参考与 get_cpu 一致时不单独使用

    if reduce_op.lower() != "sum":
        raise ValueError("matmul_reduce_scatter_v2 currently only supports reduce_op='sum'.")
    if comm_turn != 0:
        raise ValueError("matmul_reduce_scatter_v2 currently only supports comm_turn=0.")
    if is_trans_a:
        raise ValueError("aclnnMatmulReduceScatterV2 requires non-transposed x1 (is_trans_a must be False).")

    x1_t = torch.as_tensor(x1)
    x2_t = torch.as_tensor(x2)
    if x1_t.shape == torch.Size([]):
        empty = torch.tensor([], dtype=torch.float32)
        if is_amax_out:
            return empty, torch.tensor(0.0, dtype=torch.float32)
        return empty

    if x1_t.dim() != 2 or x2_t.dim() != 2:
        raise ValueError("matmul_reduce_scatter_v2_golden expects 2-D x1 and x2.")

    x1_f = x1_t.to(torch.float32)
    x2_f = x2_t.to(torch.float32)
    if is_trans_b:
        x2_f = x2_f.t()

    m, k = x1_f.shape
    n = x2_f.shape[1]

    use_scale = (
        x1_scale is not None and x2_scale is not None
        and torch.as_tensor(x1_scale).numel() > 0 and torch.as_tensor(x2_scale).numel() > 0
    )

    if not use_scale:
        out = torch.matmul(x1_f, x2_f)
        if bias is not None:
            b = torch.as_tensor(bias, dtype=torch.float32)
            if b.numel() > 0:
                out = out + b.reshape(-1)
    else:
        s1_t = torch.as_tensor(x1_scale, dtype=torch.float32)
        s2_t = torch.as_tensor(x2_scale, dtype=torch.float32)

        per_block_flag = kwargs.get("per_block_flag", None)
        is_mxfp = kwargs.get("is_mxfp", None)

        gm, gk = (m + 127) // 128, (k + 127) // 128
        gkn, gn = (k + 127) // 128, (n + 127) // 128
        if per_block_flag is None:
            per_block_flag = (
                s1_t.dim() == 2 and s2_t.dim() == 2
                and tuple(s1_t.shape) == (gm, gk) and tuple(s2_t.shape) == (gkn, gn)
            )

        if per_block_flag:
            gs = int(group_size) if group_size != 0 else ((128 << 32) | (128 << 16) | 128)
            out = _per_block_cpu_compute(gs, x1_f, x2_f, s1_t, s2_t)
        else:
            if is_mxfp is None:
                is_mxfp = s1_t.dim() == 3 and s2_t.dim() == 3

            if is_mxfp:
                x1_np = x1_f.numpy()
                x2_np = x2_f.numpy()
                if s1_t.dim() == 3:
                    s1_np = s1_t.numpy().reshape(s1_t.shape[0], -1)
                else:
                    s1_np = s1_t.numpy()
                if s2_t.dim() == 3:
                    s2_np = s2_t.numpy().reshape(s2_t.shape[0], -1)
                else:
                    s2_np = s2_t.numpy()
                out = _mxfp_cpu_compute(x1_np, x2_np, s1_np, s2_np)
                if bias is not None:
                    b = torch.as_tensor(bias, dtype=torch.float32)
                    if b.numel() > 0:
                        out = out + b.reshape(-1)
            else:
                out = torch.matmul(x1_f, x2_f)
                if bias is not None:
                    b = torch.as_tensor(bias, dtype=torch.float32)
                    if b.numel() > 0:
                        out = out + b.reshape(-1)
                prod = s1_t.numpy() * s2_t.numpy()
                double_scale = _scale_generate(prod)
                double_scale_tensor = torch.unsqueeze(torch.from_numpy(double_scale), dim=1).to(torch.float32)
                out *= double_scale_tensor

    if quant_scale is not None:
        qs = torch.as_tensor(quant_scale, dtype=torch.float32)
        if qs.numel() > 0:
            out = out * qs.reshape(-1)[0]

    if rank_size > 0:
        world_size = int(rank_size)
    else:
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    if world_size < 1:
        raise ValueError("rank_size / world_size must be positive.")

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else int(kwargs.get("rank", 0))

    if out.shape[0] % world_size != 0:
        raise ValueError("output row count (m) must be divisible by rank_size/world_size.")

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.all_reduce(out, op=dist.ReduceOp.SUM)

    scatter_m = out.shape[0] // world_size
    out = out.narrow(0, rank * scatter_m, scatter_m)

    if is_amax_out:
        amax_out = torch.max(torch.abs(out)).to(torch.float32)
        return out, amax_out

    return out
