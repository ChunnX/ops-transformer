# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AttnResFwd L0 golden reference (apply_attn_res_cust)."""

from __future__ import annotations

import os
import sys
from typing import Tuple

# CPU golden does not require torch_npu; avoid backend autoload failures in minimal env.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch

# 精度比对：同目录 precision_compare.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from precision_compare import data_compare  # noqa: E402

BF16_MERE_THRESHOLD = 2 ** (-7)  # 0.0078125
DEFAULT_NORM_EPS = 1e-5


def apply_attn_res_cust(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float = DEFAULT_NORM_EPS,
) -> torch.Tensor:
    """Golden from code/analyse/attenres.py:59-72."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm_eps)
    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = (probs.unsqueeze(-1) * v_float.unsqueeze(1)).sum(dim=2).squeeze(1)
    return hidden_states.to(v.dtype)


def prepare_tensors(
    num_tokens: int = 544,
    num_blocks: int = 12,
    hidden_size: int = 7168,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    torch.manual_seed(seed)
    prefix_sum = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    block_residual = torch.randn(num_tokens, num_blocks, hidden_size, device=device, dtype=dtype)
    norm_weight = torch.randn(hidden_size, device=device, dtype=dtype)
    proj_weight = torch.randn(1, hidden_size, device=device, dtype=dtype)
    return prefix_sum, block_residual, proj_weight, norm_weight, DEFAULT_NORM_EPS


def compute_mere_mare(npu_output: torch.Tensor, golden_output: torch.Tensor) -> Tuple[float, float]:
    npu_fp32 = npu_output.detach().float().cpu().numpy().reshape(-1)
    golden_fp32 = golden_output.detach().float().cpu().numpy().reshape(-1)
    abs_err = np.abs(npu_fp32 - golden_fp32)
    denom = np.maximum(np.abs(golden_fp32), 1e-10)
    rel_err = abs_err / denom
    return float(np.max(rel_err)), float(np.max(abs_err))


def compare_with_data_compare(npu_output: torch.Tensor, golden_output: torch.Tensor, diff_thd: float = BF16_MERE_THRESHOLD):
    result, pct, max_error = data_compare(
        npu_output.detach().float().cpu().numpy(),
        golden_output.detach().float().cpu().numpy(),
        diff_thd=diff_thd,
        pct_thd=diff_thd,
        max_diff_hd=diff_thd,
    )
    mere, mare = compute_mere_mare(npu_output, golden_output)
    passed = result == "success" and mere < diff_thd
    return {
        "result": "pass" if passed else "fail",
        "data_compare": result,
        "pct": pct,
        "max_error": max_error,
        "mere": mere,
        "mare": mare,
        "threshold": diff_thd,
    }


def run_cpu_golden_self_test(
    num_tokens: int = 544,
    num_blocks: int = 12,
    hidden_size: int = 7168,
    seed: int = 0,
) -> dict:
    prefix_sum, block_residual, proj_weight, norm_weight, norm_eps = prepare_tensors(
        num_tokens, num_blocks, hidden_size, seed=seed, device="cpu"
    )
    golden = apply_attn_res_cust(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps)
    golden_ref = apply_attn_res_cust(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps)
    stats = compare_with_data_compare(golden, golden_ref)
    stats["mode"] = "cpu_golden_self"
    return stats
