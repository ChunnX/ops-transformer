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
            "allto_allv_grouped_mat_mul": "allto_allv_grouped_mat_mul_inputs"
        }
}

import torch

def allto_allv_grouped_mat_mul_inputs(gmm_x, gmm_weight, send_counts_tensor = None, recv_counts_tensor = None, mm_x = None, mm_weight = None, group = None, ep_world_size = None, send_counts = None, recv_counts = None, trans_gmm_weight = False, trans_mm_weight = False, permute_out_flag = False, **kwargs):
    if send_counts_tensor is not None or recv_counts_tensor is not None:
        raise ValueError("send_counts_tensor/recv_counts_tensor are reserved and must be None (README).")
    if not isinstance(group, str):
        raise TypeError("group must be a string.")
    if not (0 < len(group) < 128):
        raise ValueError("group string length must be in (0, 128).")
    if ep_world_size is None:
        raise ValueError("ep_world_size is required.")

    if int(ep_world_size) != ep_world_size or ep_world_size <= 0:
        raise ValueError("ep_world_size must be a positive integer.")

    x = torch.as_tensor(gmm_x)
    w_gmm = torch.as_tensor(gmm_weight)
    if x.dim() != 2:
        raise ValueError("gmm_x must be 2-D.")
    if w_gmm.dim() != 3:
        raise ValueError("gmm_weight must be 3-D.")

    e_local = int(w_gmm.shape[0])
    if e_local <= 0 or e_local > 32:
        raise ValueError("Local expert num e must satisfy 1 <= e <= 32.")
    if e_local * int(ep_world_size) > 256:
        raise ValueError("e * ep_world_size must be <= 256.")

    h1 = int(x.shape[1])
    if not (0 < h1 < 65536):
        raise ValueError("H1 (gmm_x hidden size) must be in (0, 65536).")

    if trans_gmm_weight:
        if int(w_gmm.shape[1]) <= 0 or int(w_gmm.shape[1]) >= 65536:
            raise ValueError("N1 must be in (0, 65536) when trans_gmm_weight=True.")
        if int(w_gmm.shape[2]) != h1:
            raise ValueError("When trans_gmm_weight=True, gmm_weight.shape[2] must equal H1.")
    else:
        if int(w_gmm.shape[2]) <= 0 or int(w_gmm.shape[2]) >= 65536:
            raise ValueError("N1 must be in (0, 65536) when trans_gmm_weight=False.")
        if int(w_gmm.shape[1]) != h1:
            raise ValueError("When trans_gmm_weight=False, gmm_weight.shape[1] must equal H1.")

    # sendCounts / recvCounts
    send_t = torch.as_tensor(send_counts, dtype=torch.int64).reshape(-1)
    recv_t = torch.as_tensor(recv_counts, dtype=torch.int64).reshape(-1)
    expected_len = int(ep_world_size) * e_local
    if send_t.numel() != expected_len:
        raise ValueError(
            f"send_counts length must be ep_world_size * e = {expected_len}, got {send_t.numel()}."
        )
    if recv_t.numel() != expected_len:
        raise ValueError(
            f"recv_counts length must be ep_world_size * e = {expected_len}, got {recv_t.numel()}."
        )
    if torch.any(send_t < 0) or torch.any(recv_t < 0):
        raise ValueError("send_counts and recv_counts must be non-negative.")

    bsk = int(send_t.sum().item())
    a = int(recv_t.sum().item())
    if bsk != int(x.shape[0]):
        raise ValueError(f"sum(send_counts) must equal gmm_x.shape[0], got {bsk} vs {x.shape[0]}.")
    if not (0 < bsk < 52428800):
        raise ValueError("BSK=sum(send_counts) must be in (0, 52428800).")

    # 共享专家 mm 输入可选，但两者要么同时给，要么同时空
    if (mm_x is None) ^ (mm_weight is None):
        raise ValueError("mm_x and mm_weight must be both provided or both None.")
    if mm_x is not None and mm_weight is not None:
        mmx = torch.as_tensor(mm_x)
        mmw = torch.as_tensor(mm_weight)
        if mmx.dim() != 2 or mmw.dim() != 2:
            raise ValueError("mm_x and mm_weight must be 2-D when provided.")
        if int(mmx.shape[1]) <= 0 or int(mmx.shape[1]) > 12288:
            raise ValueError("H2 (mm_x hidden size) must be in (0, 12288].")
        if trans_mm_weight:
            if int(mmw.shape[1]) != int(mmx.shape[1]):
                raise ValueError("When trans_mm_weight=True, mm_weight.shape[1] must equal mm_x.shape[1].")
            if not (0 < int(mmw.shape[0]) < 65536):
                raise ValueError("N2 must be in (0, 65536).")
        else:
            if int(mmw.shape[0]) != int(mmx.shape[1]):
                raise ValueError("When trans_mm_weight=False, mm_weight.shape[0] must equal mm_x.shape[1].")
            if not (0 < int(mmw.shape[1]) < 65536):
                raise ValueError("N2 must be in (0, 65536).")

    allowed = {torch.float16, torch.bfloat16}
    if x.dtype not in allowed or w_gmm.dtype not in allowed:
        raise ValueError("gmm_x and gmm_weight dtypes must be float16 or bfloat16.")
    if x.dtype != w_gmm.dtype:
        raise ValueError("gmm_x and gmm_weight dtypes must match.")
    if mm_x is not None and mm_weight is not None:
        mmx = torch.as_tensor(mm_x)
        mmw = torch.as_tensor(mm_weight)
        if mmx.dtype not in allowed or mmw.dtype not in allowed:
            raise ValueError("mm_x and mm_weight dtypes must be float16 or bfloat16.")
        if mmx.dtype != mmw.dtype:
            raise ValueError("mm_x and mm_weight dtypes must match.")

    return (
        gmm_x, gmm_weight, send_counts_tensor, recv_counts_tensor,
        mm_x, mm_weight, group, ep_world_size, send_counts, recv_counts,
        trans_gmm_weight, trans_mm_weight, permute_out_flag, kwargs
    )