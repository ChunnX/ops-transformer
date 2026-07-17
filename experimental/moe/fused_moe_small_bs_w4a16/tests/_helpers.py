# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""fused_moe_small_bs_w4a16 python 测试的公共工具。

包含 kernel 原生权重布局（[E, K, N//8]，int4 二进制补码、低位在前）的打包函数，
以及纯 PyTorch 的 W4A16 参考实现（FP32 累加）。
"""

import numpy as np
import torch
import torch.nn.functional as F

GROUP_SIZE = 32  # kernel 常量（见 fused_moe_small_bs_w4a16_kernel.h）


def pack_int4(raw_signed: np.ndarray) -> np.ndarray:
    """将 [E, K, N] 的有符号 int4（-8..7）沿 N 维打包为 int32 [E, K, N//8]。

    每个 int32 打包 8 个连续 N 方向的 int4，低位在前（N[0] 占最低 4bit），
    存储值为 int4 二进制补码（value & 0xF），与 kernel 侧 int4b_t 解释一致。
    """
    assert raw_signed.shape[-1] % 8 == 0, "last dim N must be divisible by 8"
    nibbles = (raw_signed & 0xF).astype(np.uint32)
    nibbles = nibbles.reshape(*raw_signed.shape[:-1], raw_signed.shape[-1] // 8, 8)
    packed = np.zeros(nibbles.shape[:-1], dtype=np.uint32)
    for j in range(8):
        packed |= nibbles[..., j] << (4 * j)
    return packed.astype(np.int32)


def w4a16_matvec_ref(w_signed: torch.Tensor, scale: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """纯 PyTorch W4A16 matvec 参考（FP32 累加）。

    Args:
        w_signed: [K, N] 有符号 int4 原始权重（未打包）。
        scale:    [K // GROUP_SIZE, N] 分组 scale。
        x:        [K] 激活。

    Returns:
        [N] FP32 结果。
    """
    k, n = w_signed.shape
    assert scale.shape == (k // GROUP_SIZE, n), f"scale {scale.shape} != ({k // GROUP_SIZE}, {n})"
    scale_expanded = scale.repeat_interleave(GROUP_SIZE, dim=0).float()  # [K, N]
    ws = w_signed.float() * scale_expanded  # [K, N]
    return x.float() @ ws  # [N]


def fused_moe_w4a16_ref(
    x: torch.Tensor,
    raw_w13: torch.Tensor,
    scale13: torch.Tensor,
    raw_w2: torch.Tensor,
    scale2: torch.Tensor,
    expert_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """纯 PyTorch 的融合 W4A16 MoE 参考：W13 -> SwiGLU -> W2 -> topk 加权（FP32）。

    Args:
        x:            [BS, Hidden]
        raw_w13:      [E, Hidden, 2*Inter] 有符号 int4
        scale13:      [E, Hidden//32, 2*Inter]
        raw_w2:       [E, Inter, Hidden] 有符号 int4
        scale2:       [E, Inter//32, Hidden]
        expert_ids:   [BS, TopK] int32
        topk_weights: [BS, TopK] float32

    Returns:
        [BS, Hidden] fp32 输出。
    """
    bs, _ = x.shape
    top_k = expert_ids.shape[1]
    inter = raw_w2.shape[1]

    y = torch.zeros(bs, raw_w2.shape[2], dtype=torch.float32, device=x.device)
    for b in range(bs):
        for t in range(top_k):
            e = int(expert_ids[b, t].item())
            if e < 0:
                continue
            # W13: [2*Inter]
            h13 = w4a16_matvec_ref(raw_w13[e], scale13[e], x[b])
            # SwiGLU
            gate, val = h13[:inter], h13[inter:]
            h_act = F.silu(gate) * val
            # W2: [Hidden]
            h2 = w4a16_matvec_ref(raw_w2[e], scale2[e], h_act.to(x.dtype))
            y[b] += float(topk_weights[b, t].item()) * h2
    return y
