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

__golden__ = {"kernel": {"causal_conv1d": "CausalConv1d"}}

__input__ = {"kernel": {"causal_conv1d": "CausalConv1d_input"}}

import numpy as np
import torch
import torch.nn.functional as F
import ml_dtypes


def CausalConv1d_input(*input_arrays, **kwargs):
    """Custom input generator for causal_conv1d.
    
    Generates proper query_start_loc values (monotonically increasing sequence)
    instead of random values.
    """
    input_arrays = list(input_arrays)

    # Find query_start_loc in inputs (5th input, index 4)
    # Input order: x, weight, conv_states, bias, query_start_loc, ...
    if len(input_arrays) > 4 and input_arrays[4] is not None:
        qsl = input_arrays[4]
        if qsl.ndim == 1 and qsl.shape[0] > 1:
            # Generate monotonically increasing sequence
            batch_size = qsl.shape[0] - 1

            # Get x shape to determine total tokens
            x = input_arrays[0]
            if x.ndim == 2:
                # Varlen mode: x shape is (total_tokens, dim)
                total_tokens = x.shape[0]
            else:
                # Batch mode: x shape is (batch, seq_len, dim)
                total_tokens = x.shape[0] * x.shape[1]

            # Calculate sequence length from test case name if available
            testcase_name = kwargs.get('testcase_name', '')
            seq_len = None
            if '_s' in testcase_name:
                # Extract seq_len from testcase name like "fn_varlen_b16_s16_d4096_w4_bias_act0_qsl"
                import re
                match = re.search(r'_s(\d+)_', testcase_name)
                if match:
                    seq_len = int(match.group(1))

            # If seq_len not found, calculate from total tokens
            if seq_len is None:
                seq_len = total_tokens // batch_size

            # Generate query_start_loc: [0, seq_len, 2*seq_len, ..., total_tokens]
            qsl_values = np.arange(batch_size + 1, dtype=np.int32) * seq_len
            input_arrays[4] = qsl_values

    return input_arrays


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



def CausalConv1d(
    x,  # [T, D] varlen or [B, S, D] batch
    weight,  # [W, D]
    conv_states,  # [num_cache_lines, state_len, D]
    bias=None,  # [D] or None
    query_start_loc=None,  # [batch+1] int32
    cache_indices=None,  # [batch] int32
    initial_state_mode=None,  # [batch] int32
    num_accepted_tokens=None,  # [batch] int32
    *,
    activation_mode="silu",  # attr: "silu" or "none"
    null_block_id=0,  # attr
    **extra,
):
    """
    Golden function for CausalConv1d operator.

    Args:
        x: input tensor (REQUIRED) - [T, D] or [B, S, D]
        weight: weight tensor (REQUIRED) - [W, D]
        conv_states: conv states tensor (REQUIRED) - [num_cache_lines, state_len, D]
        bias: bias tensor (OPTIONAL) - [D]
        query_start_loc: query start locations (OPTIONAL) - [batch+1] int32
        cache_indices: cache indices (OPTIONAL) - [batch] int32
        initial_state_mode: initial state mode (OPTIONAL) - [batch] int32
        num_accepted_tokens: number of accepted tokens (OPTIONAL) - [batch] int32
        activation_mode: activation mode, "silu"=SiLU, "none"=Identity - Attr
        null_block_id: null block id - Attr
        **extra: extended parameters
    """
    # Infer run mode from shapes (same logic as InferIsFnMode in tiling)
    x_t = _to_torch(x)
    is_fn_mode = False
    if num_accepted_tokens is not None:
        is_fn_mode = False  # update mode
    elif x_t.ndim == 3:
        seq_len = x_t.shape[1]
        is_fn_mode = seq_len > 1
    elif x_t.ndim == 2:
        if query_start_loc is None:
            is_fn_mode = False
        else:
            is_fn_mode = True

    # Step 1: convert inputs to torch tensors and cast to float32
    orig_dtype = x_t.dtype
    x_t = x_t.to(torch.float32)
    weight_t = _to_torch(weight).to(torch.float32)
    bias_t = _to_torch(bias)
    if bias_t is not None:
        bias_t = bias_t.to(torch.float32)
    conv_states_t = _to_torch(conv_states)
    query_start_loc_t = _to_torch(query_start_loc)
    cache_indices_t = _to_torch(cache_indices)
    init_state_mode_t = _to_torch(initial_state_mode)

    activation = (activation_mode == "silu")
    kernel_size = weight_t.shape[0]
    dtype = orig_dtype
    device = x_t.device

    # Step 2: split input into per-sequence tensors
    if x_t.ndim == 3:
        batch_size = x_t.shape[0]
        seq_list = [x_t[i] for i in range(batch_size)]
        is_3d = True
    else:
        query_starts = query_start_loc_t.to(torch.int64)
        batch_size = query_starts.shape[0] - 1
        seq_list = [x_t[query_starts[i]:query_starts[i + 1]] for i in range(batch_size)]
        is_3d = False

    cache_indices_list = cache_indices_t.tolist() if cache_indices_t is not None else list(range(batch_size))
    init_state_modes = init_state_mode_t.tolist() if init_state_mode_t is not None else [0] * batch_size
    accepted_tokens_t = _to_torch(num_accepted_tokens)
    state_len = conv_states_t.shape[1]
    conv_states_out = conv_states_t.clone()
    output_list = []

    # Step 3: compute per sequence
    for i in range(batch_size):
        cache_idx = int(cache_indices_list[i])
        if cache_indices_t is not None and cache_idx == null_block_id:
            cur_seq_len = x_t.shape[1] if is_3d else seq_list[i].shape[0]
            output_list.append(
                torch.zeros(cur_seq_len,
                            x_t.shape[-1],
                            dtype=torch.float32,
                            device=device))
            continue

        seq_input = seq_list[i]
        has_init = (not is_fn_mode) or bool(init_state_modes[i])
        if has_init:
            state_offset = 0
            if accepted_tokens_t is not None:
                state_offset = max(int(accepted_tokens_t[i]) - 1, 0)
            history = conv_states_t[cache_idx][state_offset:state_offset + kernel_size - 1].to(torch.float32)
        else:
            history = torch.zeros(kernel_size - 1,
                                  seq_input.shape[-1],
                                  dtype=torch.float32,
                                  device=device)

        # Step 4: causal conv1d via F.conv1d(groups=dim)
        padded = torch.cat([history, seq_input], dim=0)
        weight_view = weight_t.T.unsqueeze(1)
        result = F.conv1d(padded.T.unsqueeze(0),
                          weight_view,
                          bias=bias_t,
                          stride=1,
                          padding=0,
                          groups=seq_input.shape[-1]).squeeze(0).T

        # Step 5: activation
        if activation:
            result = _silu(result)

        output_list.append(result.to(dtype))

        # Step 7: write back conv_states
        write_history = history if has_init else torch.zeros(
            kernel_size - 1, seq_input.shape[-1], dtype=torch.float32, device=device)
        write_padded = torch.cat([write_history, seq_input], dim=0)
        conv_states_out[cache_idx][0:state_len] = write_padded[-state_len:].to(dtype)

    # Step 8: assemble output
    output = torch.stack(output_list, dim=0) if is_3d else torch.cat(output_list, dim=0)
    conv_states_out = conv_states_out.detach().cpu()
    output = output.detach().cpu()
    if conv_states_out.dtype == torch.bfloat16:
        conv_states_out = conv_states_out.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    else:
        conv_states_out = conv_states_out.numpy()
    if output.dtype == torch.bfloat16:
        output = output.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    else:
        output = output.numpy()
    return conv_states_out, output

