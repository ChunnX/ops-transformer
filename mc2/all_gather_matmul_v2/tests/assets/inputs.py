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
            "all_gather_matmul_v2": "all_gather_matmul_v2_inputs"
        }
}

import torch

# README（Ascend 950）：当 groupSize 取该值时 bias 必须为空
_GROUP_SIZE_BIAS_MUST_EMPTY = 549764202624


def _dtype_hint_for_int4(kwargs):
    """从 kwargs 中尝试读取 x1 类型名（如测试用例传入 input_tensor_dtype / x1_dtype）。"""
    for key in ("x1_dtype", "input_tensor_dtype", "x_dtype"):
        v = kwargs.get(key)
        if v is None:
            continue
        if isinstance(v, (list, tuple)) and len(v) > 0:
            v = v[0]
        return str(v).lower()
    return None


def _dtype_name(tensor, meta_name):
    if meta_name:
        return str(meta_name).lower()
    return str(torch.as_tensor(tensor).dtype).lower()


def _is_float8_or_hifloat8(dtype_name):
    return any(x in dtype_name for x in ("float8_e4m3fn", "float8_e5m2", "hifloat8"))


def all_gather_matmul_v2_inputs(x1, x2, bias = None, x1_scale = None, x2_scale = None, quant_scale = None, group = None, is_trans_a = False, is_trans_b = False, gather_index = 0, comm_turn = 0, rank_size = 0, block_size = 0, group_size = 0, is_gather_out = True, is_amax_out = False, y_dtype = 28, comm_mode = 'aicpu', **kwargs):  # y_dtype: ge::DT_UNDEFINED
    """
    依据 README「约束说明」与 all_gather_matmul_v2 golden 约定做静态校验。

    - gather_index 与当前实现一致须为 0；
    - x1 仅不转置（is_trans_a=False）；x2 可转置；
    - 非空 tensor 时 x1/x2 为 2 维，k 一致且 k∈[256,65535)；
    - Ascend 950：bias 非空时为 1 维且长度为 n；groupSize 为 549764202624 时 bias 必须为空；
    - A2/A3：README 要求 bias 仅 nullptr——若 kwargs['soc'] 为 'a2'/'a3'（可选）则强制 bias 为空；
    - INT4：若 kwargs 中类型提示为 int4，则 k、n 须为偶数。

    comm_turn / block_size / y_dtype / comm_mode / quant_scale / is_amax_out 由调用侧传入，此处不做数值约束，原样返回。
    """
    if gather_index != 0:
        raise ValueError("all_gather_matmul_v2: gather_index 当前实现仅支持 0（与 golden / tiling 一致）。")
    if is_trans_a:
        raise ValueError(
            "all_gather_matmul_v2: x1 仅支持不转置，is_trans_a 须为 False（README）。")
    if not isinstance(group, str) or not (0 < len(group) < 128):
        raise ValueError("all_gather_matmul_v2: group 为必填字符串，长度需在 (0, 128)。")

    if rank_size < 0:
        raise ValueError("all_gather_matmul_v2: rank_size 不得为负数。")
    if rank_size != 0 and int(rank_size) != rank_size:
        raise ValueError("all_gather_matmul_v2: rank_size 须为整数。")

    x1_t = torch.as_tensor(x1)
    x2_t = torch.as_tensor(x2)
    x1_dtype_name = _dtype_name(x1_t, kwargs.get("x1_dtype"))
    x2_dtype_name = _dtype_name(x2_t, kwargs.get("x2_dtype"))
    soc = str(kwargs.get("soc", "")).lower()

    # A2/A3 明确不支持空 tensor；950 沿用 golden 的空 tensor 兼容路径
    if x1_t.shape == torch.Size([]):
        if soc in ("a2", "a3", "atlas_a2", "atlas_a3"):
            raise ValueError("all_gather_matmul_v2: A2/A3 场景不支持空 tensor（README）。")
        return (x1, x2, bias, x1_scale, x2_scale,
                quant_scale, group,
                is_trans_a, is_trans_b,
                gather_index, comm_turn,
                rank_size, block_size,
                group_size, is_gather_out,
                is_amax_out, y_dtype, comm_mode, kwargs)

    if x1_t.dim() != 2 or x2_t.dim() != 2:
        raise ValueError(
            "all_gather_matmul_v2: 输入 x1、x2 须为 2 维（README：x1 为 (m,k)，x2 为 (k,n) 或转置存储）。")

    m, k = int(x1_t.shape[0]), int(x1_t.shape[1])
    if is_trans_b:
        n, k2 = int(x2_t.shape[0]), int(x2_t.shape[1])
    else:
        k2, n = int(x2_t.shape[0]), int(x2_t.shape[1])

    if k == 0:
        raise ValueError("all_gather_matmul_v2: k 轴不可为空（README）。")
    if k != k2:
        raise ValueError(
            f"all_gather_matmul_v2: matmul 要求 k 轴一致，x1 的 k={k}，x2 侧 k={k2}。")

    if not (256 <= k < 65535):
        raise ValueError(
            f"all_gather_matmul_v2: k 轴须在 [256, 65535) 内（README），当前 k={k}。")

    if int(group_size) == _GROUP_SIZE_BIAS_MUST_EMPTY:
        if bias is not None and torch.as_tensor(bias).numel() > 0:
            raise ValueError(
                "all_gather_matmul_v2: groupSize 为 549764202624 时 bias 必须为空（README Ascend 950）。")

    if soc in ("a2", "a3", "atlas_a2", "atlas_a3"):
        if bias is not None and torch.as_tensor(bias).numel() > 0:
            raise ValueError(
                "all_gather_matmul_v2: Atlas A2/A3 场景 bias 仅支持 nullptr（README）。")

    if bias is not None:
        b_t = torch.as_tensor(bias)
        if b_t.numel() > 0:
            if b_t.dim() != 1:
                raise ValueError("all_gather_matmul_v2: bias 非空时须为 1 维，shape 为 (n,)（README Ascend 950）。")
            if int(b_t.shape[0]) != n:
                raise ValueError(
                    f"all_gather_matmul_v2: bias 长度须等于 n={n}，当前 bias.shape[0]={b_t.shape[0]}。")

    hint = _dtype_hint_for_int4(kwargs)
    if hint and "int4" in hint:
        if k % 2 != 0 or n % 2 != 0:
            raise ValueError(
                "all_gather_matmul_v2: x1/x2 为 INT4 时 k 与 n 须均为偶数（README Atlas A2/A3）。")
    if soc in ("a2", "a3", "atlas_a2", "atlas_a3"):
        allowed = ("float16", "bfloat16", "int8", "int4")
        if not any(a in x1_dtype_name for a in allowed):
            raise ValueError("all_gather_matmul_v2: A2/A3 仅支持 x1 为 float16/bfloat16/int8/int4。")
        if not any(a in x2_dtype_name for a in allowed):
            raise ValueError("all_gather_matmul_v2: A2/A3 仅支持 x2 为 float16/bfloat16/int8/int4。")
        if x1_dtype_name != x2_dtype_name:
            raise ValueError("all_gather_matmul_v2: A2/A3 场景要求 x1 与 x2 dtype 完全一致。")
    elif soc in ("950", "ascend950", "950pr", "950dt"):
        allowed = ("float16", "bfloat16", "hifloat8", "float8_e4m3fn", "float8_e5m2")
        if not any(a in x1_dtype_name for a in allowed):
            raise ValueError("all_gather_matmul_v2: 950 仅支持 x1 为 float16/bfloat16/hifloat8/float8。")
        if not any(a in x2_dtype_name for a in allowed):
            raise ValueError("all_gather_matmul_v2: 950 仅支持 x2 为 float16/bfloat16/hifloat8/float8。")
        # README：hifloat8 需同型；e4m3/e5m2 可混合
        if ("hifloat8" in x1_dtype_name or "hifloat8" in x2_dtype_name) and x1_dtype_name != x2_dtype_name:
            raise ValueError("all_gather_matmul_v2: 950 上 hifloat8 场景要求 x1/x2 dtype 一致。")
    # 低比特路径通常需要 scale 输入，不在 fp16/bf16 纯浮点路径放行
    if _is_float8_or_hifloat8(x1_dtype_name) or _is_float8_or_hifloat8(x2_dtype_name):
        if x1_scale is None or torch.as_tensor(x1_scale).numel() == 0:
            raise ValueError("all_gather_matmul_v2: float8/hifloat8 场景要求 x1_scale 非空。")
        if x2_scale is None or torch.as_tensor(x2_scale).numel() == 0:
            raise ValueError("all_gather_matmul_v2: float8/hifloat8 场景要求 x2_scale 非空。")

    return (x1, x2, bias, x1_scale, x2_scale,
            quant_scale, group,
            is_trans_a, is_trans_b,
            gather_index, comm_turn,
            rank_size, block_size,
            group_size, is_gather_out,
            is_amax_out, y_dtype, comm_mode, kwargs)
