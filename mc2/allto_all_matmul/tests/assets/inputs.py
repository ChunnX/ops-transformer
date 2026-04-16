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
__input__ = {
        "kernel": {
            "allto_all_matmul": "allto_all_matmul_inputs"
        }
}

import torch

# README：各产品支持的卡数并集；可选 kwargs['soc'] 为 'a2'|'a3'|'950' 时收紧
_WORLD_SIZE_BY_SOC = {
    "a2": {2, 4, 8},
    "atlas_a2": {2, 4, 8},
    "a3": {2, 4, 8, 16},
    "atlas_a3": {2, 4, 8, 16},
    "950": {2, 4, 8, 16},
    "ascend950": {2, 4, 8, 16},
}
_WORLD_SIZE_ALL = {2, 4, 8, 16}


def _effective_all2all_axes(all2all_axes):
    """README：空或缺省按 [-2,-1]；须为长度 2 的一维语义。"""
    if all2all_axes is None or (hasattr(all2all_axes, "__len__") and len(all2all_axes) == 0):
        return [-2, -1]
    ax = list(all2all_axes)
    if len(ax) != 2:
        raise ValueError("all2all_axes must be a 1-D array of shape (2) (README).")
    if sorted(ax) != [-2, -1]:
        raise ValueError(
            "all2all_axes currently only supports [-2, -1] (README), "
            f"got {all2all_axes!r}."
        )
    return ax


def allto_all_matmul_inputs(x1, x2, bias = None, x1_scale = None, x2_scale = None, comm_scale = None, x1_offset = None, x2_offset = None, group = None, world_size = None, all2all_axes = {-2,-1}, y_dtype = 28, x1_quant_mode = 0, x2_quant_mode = 0, comm_quant_mode = 0, x1_quant_dtype = 28, comm_quant_dtype = 28, transpose_x1 = False, transpose_x2 = False, group_size = 0, alltoall_out_flag = True, **kwargs):  # y_dtype: ge::DT_UNDEFINED
    """
    依据 README「约束说明」做静态校验（与量化/芯片相关的细项可通过 kwargs['soc'] 收紧 world_size）。

    - group：字符串长度 (0, 128)；
    - world_size：正整数，默认允许 {2,4,8,16}；若 kwargs['soc'] 为 a2/a3/950 等则按 README 子集校验；
    - transpose_x1 须为 False；comm_quant_mode 须为 0；comm_quant_dtype 当前须为 -1（UNDEFINED）或与占位 28；
    - x1 为 2 维 (BS,H)，BS 整除 world_size；x2 为 2 维，与 transpose_x2、H*rankSize 关系见 README；
    - bias 非空则 1 维 (N,)；all2all_axes 空或 [-2,-1]；
    - 量化 mode 枚举 0~7；若给出 x1_scale/x2_scale，按 x1_quant_mode/x2_quant_mode 做形状抽查（mx/K-C 动态等）。
    """
    if not isinstance(group, str):
        raise TypeError("group must be str (README).")
    if not (0 < len(group) < 128):
        raise ValueError("group string length must be in (0, 128) (README).")

    ws = int(world_size)
    if ws <= 0:
        raise ValueError("world_size (rankSize) must be positive (README).")

    soc = str(kwargs.get("soc", "")).lower().replace(" ", "")
    if soc and soc in _WORLD_SIZE_BY_SOC:
        if ws not in _WORLD_SIZE_BY_SOC[soc]:
            raise ValueError(
                f"world_size {ws} not in allowed set for soc={soc!r} (README): {_WORLD_SIZE_BY_SOC[soc]}."
            )
    elif ws not in _WORLD_SIZE_ALL:
        raise ValueError(
            f"world_size must be one of {_WORLD_SIZE_ALL} (README union), got {ws}."
        )

    if transpose_x1:
        raise ValueError("transpose_x1 must be False — x1 does not support transpose (README).")

    if comm_quant_mode != 0:
        raise ValueError("comm_quant_mode must be 0 (no comm quant) (README).")
    if comm_quant_dtype not in (-1, 28):
        raise ValueError(
            "comm_quant_dtype must be -1 (ACL_DT_UNDEFINED) or placeholder 28 (README)."
        )

    if x1_quant_mode not in range(0, 8) or x2_quant_mode not in range(0, 8):
        raise ValueError("x1_quant_mode / x2_quant_mode must be in [0, 7] (README enum).")

    if group_size < 0:
        raise ValueError("group_size must be >= 0 (README).")

    _effective_all2all_axes(all2all_axes)

    x1_t = torch.as_tensor(x1)
    x2_t = torch.as_tensor(x2)
    if x1_t.dim() != 2 or x2_t.dim() != 2:
        raise ValueError("x1 and x2 must be 2-D (README).")

    bs, h = int(x1_t.shape[0]), int(x1_t.shape[1])
    if bs % ws != 0:
        raise ValueError(f"x1.shape[0] (BS) must be divisible by world_size (README), got {bs} % {ws} != 0.")

    h_rs = h * ws

    if not transpose_x2:
        # x2: (H*rankSize, N)
        if int(x2_t.shape[0]) != h_rs:
            raise ValueError(
                "transpose_x2=False: x2.shape[0] must be H*rankSize = x1.shape[1]*world_size "
                f"(README), expected {h_rs}, got {x2_t.shape[0]}."
            )
        n_out = int(x2_t.shape[1])
    else:
        # x2: (N, H*rankSize) — mx 等场景
        if int(x2_t.shape[1]) != h_rs:
            raise ValueError(
                "transpose_x2=True: x2.shape[1] must be H*rankSize (README), "
                f"expected {h_rs}, got {x2_t.shape[1]}."
            )
        n_out = int(x2_t.shape[0])

    if x1_quant_mode == 6 and not transpose_x2:
        raise ValueError("mx quantization (mode 6) requires transpose_x2=True (README).")

    if bias is not None:
        b_t = torch.as_tensor(bias)
        if b_t.numel() > 0:
            if b_t.dim() != 1:
                raise ValueError("bias must be 1-D when non-empty (README).")
            if int(b_t.shape[0]) != n_out:
                raise ValueError(
                    f"bias shape must be (N,) with N={n_out} (README), got length {b_t.shape[0]}."
                )

    # README：scale 形状与量化模式
    if x1_scale is not None and torch.as_tensor(x1_scale).numel() > 0:
        s1 = torch.as_tensor(x1_scale)
        if x1_quant_mode == 6:
            ce = (h + 63) // 64
            if tuple(s1.shape) != (bs, ce, 2):
                raise ValueError(
                    "mx: x1_scale shape must be (BS, ceil(H/64), 2) (README), "
                    f"expected ({bs}, {ce}, 2), got {tuple(s1.shape)}."
                )
        elif x1_quant_mode == 7:
            if s1.dim() != 1 or int(s1.shape[0]) != h_rs:
                raise ValueError(
                    "K-C dynamic: x1_scale must be 1-D (H*rankSize) (README), "
                    f"expected ({h_rs},), got {tuple(s1.shape)}."
                )
        elif s1.dim() == 1:
            if int(s1.shape[0]) not in (bs, h_rs):
                raise ValueError(
                    "K-C / other: x1_scale 1-D 时长度应为 BS 或 H*rankSize（README），"
                    f"got length {s1.shape[0]}, BS={bs}, H*rankSize={h_rs}."
                )

    if x2_scale is not None and torch.as_tensor(x2_scale).numel() > 0:
        s2 = torch.as_tensor(x2_scale)
        if x2_quant_mode == 6:
            ce2 = (h_rs + 63) // 64
            if tuple(s2.shape) != (n_out, ce2, 2):
                raise ValueError(
                    "mx: x2_scale shape must be (N, ceil(H*rankSize/64), 2) (README), "
                    f"expected ({n_out}, {ce2}, 2), got {tuple(s2.shape)}."
                )
        elif s2.dim() == 1:
            if int(s2.shape[0]) != n_out:
                raise ValueError(
                    "x2_scale 非 mx 时须为 1-D (N,)（README），"
                    f"expected ({n_out},), got {tuple(s2.shape)}."
                )
        else:
            raise ValueError(
                "x2_scale: mx 场景须为 3 维，其余场景须为 1 维 (N,)（README）。"
                f" mode={x2_quant_mode}, shape={tuple(s2.shape)}."
            )

    return (x1, x2, bias, x1_scale, x2_scale, comm_scale, x1_offset, x2_offset,
            group, world_size, all2all_axes,
            y_dtype, x1_quant_mode,
            x2_quant_mode, comm_quant_mode, x1_quant_dtype,
            comm_quant_dtype, transpose_x1, transpose_x2,
            group_size, alltoall_out_flag, kwargs)
