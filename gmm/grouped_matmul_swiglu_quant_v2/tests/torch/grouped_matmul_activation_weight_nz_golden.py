# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""Reference implementation for GroupedMatmulActivationWeightNz torch tests.

The caller supplies already dequantized logical MXFP4 values. Packing to FRACTAL_NZ
and E8M0/FP4 decoding are input-construction concerns; this reference deliberately
keeps the mathematical contract visible and checks the grouped GEMM and SwiGLU path.
"""

import torch


def grouped_matmul_activation_weight_nz_golden(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 0,
    swiglu_mode: int = 0,
    clamp_limit: float = 0.0,
    glu_alpha: float = 1.0,
    glu_bias: float = 0.0,
) -> torch.Tensor:
    """Computes the BF16-output GMM+SwiGLU reference from logical FP32 tensors."""
    if x.ndim != 2 or weight.ndim != 3:
        raise ValueError("x must be [M, K] and weight must be [E, K, N].")
    if weight.shape[1] != x.shape[1] or weight.shape[2] % 2 != 0:
        raise ValueError("K must match and N must be even.")
    if group_list.numel() != weight.shape[0]:
        raise ValueError("group_list length must equal the expert count.")
    if group_list_type not in (0, 1):
        raise ValueError("group_list_type must be 0 or 1.")
    if swiglu_mode not in (0, 2):
        raise ValueError("swiglu_mode must be 0 or 2.")

    counts = group_list.to(torch.int64)
    if group_list_type == 0:
        counts = torch.diff(torch.cat((torch.zeros(1, dtype=torch.int64), counts)))
    if torch.any(counts < 0) or int(counts.sum()) > x.shape[0]:
        raise ValueError("group_list does not describe a valid token partition.")

    output = torch.empty((x.shape[0], weight.shape[2] // 2), dtype=torch.float32)
    token_begin = 0
    for expert, count in enumerate(counts.tolist()):
        token_end = token_begin + count
        gmm = x[token_begin:token_end].float().matmul(weight[expert].float())
        act, gate = torch.chunk(gmm, 2, dim=-1)
        if swiglu_mode == 2:
            if clamp_limit <= 0:
                raise ValueError("clamp_limit must be positive when swiglu_mode is 2.")
            act = torch.minimum(act, torch.tensor(clamp_limit, dtype=act.dtype))
            gate = torch.clamp(gate, -clamp_limit, clamp_limit) + glu_bias
            output[token_begin:token_end] = act * torch.sigmoid(glu_alpha * act) * gate
        else:
            output[token_begin:token_end] = torch.nn.functional.silu(act) * gate
        token_begin = token_end
    return output.to(torch.bfloat16)
