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
            "allto_allv_quant_grouped_mat_mul": "allto_allv_quant_grouped_mat_mul_inputs"
        }
}

import torch

def _dtype_name(tensor, meta_name):
    if meta_name:
        return str(meta_name).lower()
    return str(torch.as_tensor(tensor).dtype).lower()


def _is_lowbit(name):
    return any(x in name for x in ("hifloat8", "float8_e4m3fn", "float8_e5m2"))


def allto_allv_quant_grouped_mat_mul_inputs(gmm_x, gmm_weight, send_counts_tensor = None, recv_counts_tensor = None, mm_x = None, mm_weight = None, gmm_x_scale = None, gmm_weight_scale = None, mm_x_scale = None, mm_weight_scale = None, group = None, ep_world_size = None, send_counts = None, recv_counts = None, trans_gmm_weight = False, trans_mm_weight = False, permute_out_flag = False, gmm_x_quant_mode = None, gmm_weight_quant_mode = None, mm_x_quant_mode = 0, mm_weight_quant_mode = 0, group_size = 0, y_dtype = 28, mm_dtype = 28, **kwargs):
    # 与 README 对齐：send/recv_counts_tensor 预留参数，仅支持 nullptr
    if send_counts_tensor is not None or recv_counts_tensor is not None:
        raise ValueError("send_counts_tensor/recv_counts_tensor are reserved and must be None.")

    if not isinstance(group, str):
        raise TypeError("group must be a string.")
    if not (0 < len(group) < 128):
        raise ValueError("group string length must be in (0, 128).")
    if gmm_x_quant_mode is None or gmm_weight_quant_mode is None:
        raise ValueError("gmm_x_quant_mode and gmm_weight_quant_mode are required.")

    if ep_world_size is None or send_counts is None or recv_counts is None:
        raise ValueError("ep_world_size/send_counts/recv_counts are required.")
    if int(ep_world_size) != ep_world_size or ep_world_size <= 0:
        raise ValueError("ep_world_size must be a positive integer.")

    x = torch.as_tensor(gmm_x)
    w_gmm = torch.as_tensor(gmm_weight)
    if x.dim() != 2:
        raise ValueError("gmm_x must be 2-D.")
    if w_gmm.dim() != 3:
        raise ValueError("gmm_weight must be 3-D.")

    # 变量约束（README）
    e_local = int(w_gmm.shape[0])
    if not (0 < e_local <= 32):
        raise ValueError("local expert num e must satisfy (0, 32].")
    if e_local * int(ep_world_size) > 256:
        raise ValueError("e * ep_world_size must be <= 256.")

    h1 = int(x.shape[1])
    if not (0 < h1 < 65536):
        raise ValueError("H1 must be in (0, 65536).")

    if trans_gmm_weight:
        # gmm_weight: (e, n1, h1)
        n1 = int(w_gmm.shape[1])
        if int(w_gmm.shape[2]) != h1:
            raise ValueError("trans_gmm_weight=True requires gmm_weight.shape[2] == gmm_x.shape[1].")
    else:
        # gmm_weight: (e, h1, n1)
        n1 = int(w_gmm.shape[2])
        if int(w_gmm.shape[1]) != h1:
            raise ValueError("trans_gmm_weight=False requires gmm_weight.shape[1] == gmm_x.shape[1].")
    if not (0 < n1 < 65536):
        raise ValueError("N1 must be in (0, 65536).")

    send_t = torch.as_tensor(send_counts, dtype=torch.int64).reshape(-1)
    recv_t = torch.as_tensor(recv_counts, dtype=torch.int64).reshape(-1)
    expected_len = int(ep_world_size) * e_local
    if send_t.numel() != expected_len or recv_t.numel() != expected_len:
        raise ValueError(
            f"send_counts/recv_counts length must be ep_world_size * e = {expected_len}."
        )
    if torch.any(send_t < 0) or torch.any(recv_t < 0):
        raise ValueError("send_counts/recv_counts must be non-negative.")

    bsk = int(send_t.sum().item())
    if bsk != int(x.shape[0]):
        raise ValueError("sum(send_counts) must equal gmm_x.shape[0].")
    if not (0 < bsk < 52428800):
        raise ValueError("BSK=sum(send_counts) must be in (0, 52428800).")

    # 共享专家输入：要么同时为空，要么同时给定
    if (mm_x is None) ^ (mm_weight is None):
        raise ValueError("mm_x and mm_weight must both be provided or both be None.")
    if (mm_x_scale is None) ^ (mm_weight_scale is None):
        raise ValueError("mm_x_scale and mm_weight_scale must both be provided or both be None.")

    if mm_x is not None and mm_weight is not None:
        mmx = torch.as_tensor(mm_x)
        mmw = torch.as_tensor(mm_weight)
        if mmx.dim() != 2 or mmw.dim() != 2:
            raise ValueError("mm_x and mm_weight must be 2-D when provided.")
        h2 = int(mmx.shape[1])
        if not (0 < h2 <= 12288):
            raise ValueError("H2 must be in (0, 12288].")

        if trans_mm_weight:
            n2 = int(mmw.shape[0])
            if int(mmw.shape[1]) != h2:
                raise ValueError("trans_mm_weight=True requires mm_weight.shape[1] == mm_x.shape[1].")
        else:
            n2 = int(mmw.shape[1])
            if int(mmw.shape[0]) != h2:
                raise ValueError("trans_mm_weight=False requires mm_weight.shape[0] == mm_x.shape[1].")
        if not (0 < n2 < 65536):
            raise ValueError("N2 must be in (0, 65536).")

    # dtype: 仅支持 950 低比特输入组合（HIFLOAT8/FLOAT8）
    gx_name = _dtype_name(x, kwargs.get("gmm_x_dtype"))
    gw_name = _dtype_name(w_gmm, kwargs.get("gmm_weight_dtype"))
    if not _is_lowbit(gx_name) or not _is_lowbit(gw_name):
        raise ValueError("gmm_x/gmm_weight must be low-bit (hifloat8/float8_e4m3fn/float8_e5m2).")
    if mm_x is not None and mm_weight is not None:
        mmx_name = _dtype_name(torch.as_tensor(mm_x), kwargs.get("mm_x_dtype"))
        mmw_name = _dtype_name(torch.as_tensor(mm_weight), kwargs.get("mm_weight_dtype"))
        if not _is_lowbit(mmx_name) or not _is_lowbit(mmw_name):
            raise ValueError("mm_x/mm_weight must be low-bit (hifloat8/float8_e4m3fn/float8_e5m2).")

    # 量化参数：gmm 分支必填；mm 分支仅在启用时检查
    valid_qmodes = {1, 6}
    for name, mode in [
        ("gmm_x_quant_mode", gmm_x_quant_mode),
        ("gmm_weight_quant_mode", gmm_weight_quant_mode),
    ]:
        if int(mode) != mode or int(mode) not in valid_qmodes:
            raise ValueError(f"{name} must be one of {sorted(valid_qmodes)} (pertensor/mx).")
    if mm_x is not None and mm_weight is not None:
        for name, mode in [("mm_x_quant_mode", mm_x_quant_mode), ("mm_weight_quant_mode", mm_weight_quant_mode)]:
            if int(mode) != mode or int(mode) not in valid_qmodes:
                raise ValueError(f"{name} must be one of {sorted(valid_qmodes)} (pertensor/mx) when mm branch enabled.")
    else:
        if int(mm_x_quant_mode) != 0 or int(mm_weight_quant_mode) != 0:
            raise ValueError("mm_x_quant_mode/mm_weight_quant_mode must be 0 when mm branch is disabled.")

    # scale 不能为空（量化算子）
    for name, s in [
        ("gmm_x_scale", gmm_x_scale),
        ("gmm_weight_scale", gmm_weight_scale),
    ]:
        if s is None:
            raise ValueError(f"{name} must be provided for quant grouped matmul.")
    if mm_x is not None and (mm_x_scale is None or mm_weight_scale is None):
        raise ValueError("mm_x/mm_weight provided => mm_x_scale/mm_weight_scale must be provided.")

    s_gx = torch.as_tensor(gmm_x_scale)
    s_gw = torch.as_tensor(gmm_weight_scale)
    if int(gmm_x_quant_mode) == 6:
        if s_gx.dim() < 2 or s_gw.dim() < 2:
            raise ValueError("mx mode requires gmm_x_scale and gmm_weight_scale to be at least 2-D.")
    else:
        if s_gx.dim() != 1 or s_gw.dim() != 1:
            raise ValueError("pertensor mode requires gmm_x_scale and gmm_weight_scale to be 1-D.")

    if mm_x is not None:
        s_mx = torch.as_tensor(mm_x_scale)
        s_mw = torch.as_tensor(mm_weight_scale)
        if int(mm_x_quant_mode) == 6:
            if s_mx.dim() < 2 or s_mw.dim() < 2:
                raise ValueError("mx mode requires mm_x_scale and mm_weight_scale to be at least 2-D.")
        else:
            if s_mx.dim() != 1 or s_mw.dim() != 1:
                raise ValueError("pertensor mode requires mm_x_scale and mm_weight_scale to be 1-D.")

    if int(group_size) < 0:
        raise ValueError("group_size must be >= 0.")
    soc = str(kwargs.get("soc", "")).lower()
    if soc and soc not in {"950", "ascend950", "950pr", "950dt"}:
        raise ValueError("allto_allv_quant_grouped_mat_mul is only supported on Ascend 950 series.")

    return (
        gmm_x, gmm_weight, send_counts_tensor,
        recv_counts_tensor, mm_x, mm_weight,
        gmm_x_scale, gmm_weight_scale, mm_x_scale,
        mm_weight_scale, group, ep_world_size,
        send_counts, recv_counts, trans_gmm_weight,
        trans_mm_weight, permute_out_flag, gmm_x_quant_mode,
        gmm_weight_quant_mode, mm_x_quant_mode, mm_weight_quant_mode, group_size,
        y_dtype, mm_dtype, kwargs
    )