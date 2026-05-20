#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""CausalConv1d golden function."""

__golden__ = {
    "kernel": {
        "causal_conv1d": "CausalConv1d"
    }
}

import numpy as np
import torch
import torch.nn.functional as F


def _to_torch(arr):
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr
    try:
        return torch.from_numpy(arr)
    except TypeError:
        return torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16)


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _split_sequences(x: torch.Tensor, query_start_loc: torch.Tensor | None
                     ) -> tuple[list[torch.Tensor], int]:
    if x.ndim == 3:
        B = x.shape[0]
        return [x[i] for i in range(B)], B
    qsl = query_start_loc.to(torch.int64)
    B = qsl.shape[0] - 1
    return [x[qsl[i] : qsl[i + 1]] for i in range(B)], B


def CausalConv1d(
    x,                  # [T, D] varlen or [B, S, D] batch
    weight,             # [W, D]
    bias,               # [D] or None
    conv_states,        # [num_cache_lines, state_len, D]
    query_start_loc=None,            # [batch+1] int64
    cache_indices=None,              # [batch] int64
    initial_state_mode=None,         # [batch] int64
    num_accepted_tokens=None,        # [batch] int64
    *,
    activation_mode=0,   # attr
    pad_slot_id=-1,      # attr
    run_mode=0,          # attr
    **extra,
):
    """
    Golden function for CausalConv1d operator.

    Args:
        x: input tensor (REQUIRED) - [T, D] or [B, S, D]
        weight: weight tensor (REQUIRED) - [W, D]
        bias: bias tensor (OPTIONAL) - [D]
        conv_states: conv states tensor (REQUIRED) - [num_cache_lines, state_len, D]
        query_start_loc: query start locations (OPTIONAL) - [batch+1] int64
        cache_indices: cache indices (OPTIONAL) - [batch] int64
        initial_state_mode: initial state mode (OPTIONAL) - [batch] int64
        num_accepted_tokens: number of accepted tokens (OPTIONAL) - [batch] int64
        activation_mode: activation mode, 0=Identity, 1=SiLU - Attr
        pad_slot_id: pad slot id - Attr
        run_mode: run mode, 0=forward, 1=update - Attr
        **extra: extended parameters
    """
    # Step 1: convert inputs to torch tensors and cast to float32
    x_t = _to_torch(x).to(torch.float32)
    w_t = _to_torch(weight).to(torch.float32)
    b_t = _to_torch(bias)
    if b_t is not None:
        b_t = b_t.to(torch.float32)
    cs_t = _to_torch(conv_states)
    qsl_t = _to_torch(query_start_loc)
    ci_t = _to_torch(cache_indices)
    ism_t = _to_torch(initial_state_mode)

    activation = (activation_mode == 1)
    W = w_t.shape[0]
    dtype = x_t.dtype
    device = x_t.device

    # Step 2: split input into per-sequence tensors
    if x_t.ndim == 3:
        B = x_t.shape[0]
        xs = [x_t[i] for i in range(B)]
        is_3d = True
    else:
        qsl = qsl_t.to(torch.int64)
        B = qsl.shape[0] - 1
        xs = [x_t[qsl[i] : qsl[i + 1]] for i in range(B)]
        is_3d = False

    cache_idx = ci_t.tolist() if ci_t is not None else list(range(B))
    ism = ism_t.tolist() if ism_t is not None else [0] * B
    state_len = cs_t.shape[1]
    cs_out = cs_t.clone()
    ys = []

    # Step 3: compute per sequence
    for i in range(B):
        ci = int(cache_idx[i])
        if ci == pad_slot_id:
            seq_len = x_t.shape[1] if is_3d else xs[i].shape[0]
            ys.append(torch.zeros(seq_len, x_t.shape[-1], dtype=torch.float32, device=device))
            continue

        seq_x = xs[i]
        has_init = (run_mode == 1) or bool(ism[i])
        if has_init:
            history = cs_t[ci][:W - 1].to(torch.float32)
        else:
            history = torch.zeros(W - 1, seq_x.shape[-1], dtype=torch.float32, device=device)

        # Step 4: causal conv1d via F.conv1d(groups=dim)
        padded = torch.cat([history, seq_x], dim=0)
        w = w_t.T.unsqueeze(1)
        b = b_t
        result = F.conv1d(padded.T.unsqueeze(0), w, bias=b,
                           stride=1, padding=0, groups=seq_x.shape[-1]).squeeze(0).T

        # Step 5: zero first W-1 outputs when no prior history
        if not has_init:
            result[:W - 1] = 0

        # Step 6: activation
        if activation:
            result = _silu(result)

        ys.append(result.to(dtype))

        # Step 7: write back conv_states
        write_history = history if has_init else torch.zeros(W - 1, seq_x.shape[-1], dtype=torch.float32, device=device)
        write_padded = torch.cat([write_history, seq_x], dim=0)
        cs_out[ci] = write_padded[-state_len:].to(dtype)

    # Step 8: assemble output
    y = torch.stack(ys, dim=0) if is_3d else torch.cat(ys, dim=0)
    return cs_out.detach().cpu().numpy(), y.detach().cpu().numpy()


# -----------------------------------------------------------
# Quick smoke test
# -----------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, S, D, W = 2, 8, 128, 4
    x = torch.randn(B, S, D, dtype=torch.float16)
    weight = torch.randn(W, D, dtype=torch.float16)
    bias = torch.randn(D, dtype=torch.float16)
    cs_fwd = torch.zeros(B, W - 1, D, dtype=torch.float16)

    cs_out, y = CausalConv1d(x.numpy(), weight.numpy(), bias.numpy(), cs_fwd.numpy(),
                              None, None, None, None, activation_mode=1, run_mode=0)
    print(f"fwd:    y shape={y.shape}  cs_out shape={cs_out.shape}  dtype={y.dtype}")

    x_upd = x[:, -1:, :]
    cs_out, y = CausalConv1d(x_upd.numpy(), weight.numpy(), bias.numpy(), cs_fwd.numpy(),
                              None, None, None, None, activation_mode=1, run_mode=1)
    print(f"update: y shape={y.shape}  cs_out shape={cs_out.shape}  dtype={y.dtype}")

    print("All passed.")
