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
            "grouped_mat_mul_allto_allv": "grouped_mat_mul_allto_allv_inputs"
        }
}

import torch

def grouped_mat_mul_allto_allv_inputs(gmm_x, gmm_weight, send_counts_tensor = None, recv_counts_tensor = None, mm_x = None, mm_weight = None, group = None, ep_world_size = None, send_counts = None, recv_counts = None, trans_gmm_weight = False, trans_mm_weight = False, **kwargs):
    # README: send/recv_counts_tensor 当前版本不支持，必须传空
    if send_counts_tensor is not None or recv_counts_tensor is not None:
        raise ValueError("send_counts_tensor and recv_counts_tensor are not supported currently, pass None.")

    if not isinstance(group, str):
        raise TypeError("group must be a string.")
    if not (0 < len(group) < 128):
        raise ValueError("group length must be in (0, 128).")

    if int(ep_world_size) != ep_world_size or int(ep_world_size) <= 0:
        raise ValueError("ep_world_size must be a positive integer.")

    # 不同产品支持范围不同；输入层采用并集，必要时可通过 kwargs['soc'] 收紧
    ep = int(ep_world_size)
    soc = str(kwargs.get("soc", "")).lower()
    if soc in ("a3", "atlas_a3"):
        valid_ep = {8, 16, 32, 64, 128}
    elif soc in ("950", "ascend950", "950pr", "950dt"):
        valid_ep = {2, 4, 8, 16, 32, 64}
    else:
        valid_ep = {2, 4, 8, 16, 32, 64, 128}
    if ep not in valid_ep:
        raise ValueError(f"ep_world_size={ep} not in supported set {sorted(valid_ep)}.")

    x = torch.as_tensor(gmm_x)
    w = torch.as_tensor(gmm_weight)
    if x.dim() != 2:
        raise ValueError("gmm_x must be 2-D with shape (A, H1).")
    if w.dim() != 3:
        raise ValueError("gmm_weight must be 3-D with shape (e, H1, N1).")

    a, h1 = int(x.shape[0]), int(x.shape[1])
    if not (0 < a < 52428800):
        raise ValueError("A=sum(send_counts) / gmm_x.shape[0] must be in (0, 52428800).")
    if not (0 < h1 <= 65536):
        raise ValueError("H1 must be in (0, 65536].")

    e = int(w.shape[0])
    if not (0 < e <= 32):
        raise ValueError("e must satisfy 0 < e <= 32.")
    if e * ep > 256:
        raise ValueError("e * ep_world_size must be <= 256.")

    # gmm_weight 维度对齐（考虑转置）
    if trans_gmm_weight:
        n1 = int(w.shape[1])
        if int(w.shape[2]) != h1:
            raise ValueError("trans_gmm_weight=True expects gmm_weight.shape[2] == gmm_x.shape[1].")
    else:
        n1 = int(w.shape[2])
        if int(w.shape[1]) != h1:
            raise ValueError("trans_gmm_weight=False expects gmm_weight.shape[1] == gmm_x.shape[1].")
    if not (0 < n1 <= 65536):
        raise ValueError("N1 must be in (0, 65536].")

    sc = torch.as_tensor(send_counts, dtype=torch.int64).reshape(-1)
    rc = torch.as_tensor(recv_counts, dtype=torch.int64).reshape(-1)
    expected = e * ep
    if sc.numel() != expected or rc.numel() != expected:
        raise ValueError(f"send_counts/recv_counts length must equal e * ep_world_size = {expected}.")
    if torch.any(sc < 0) or torch.any(rc < 0):
        raise ValueError("send_counts and recv_counts must be non-negative.")
    if int(sc.sum().item()) != a:
        raise ValueError("sum(send_counts) must equal gmm_x.shape[0].")
    bsk = int(rc.sum().item())
    if not (0 < bsk < 52428800):
        raise ValueError("BSK=sum(recv_counts) must be in (0, 52428800).")

    # 共享专家输入：同时传或同时不传
    if (mm_x is None) ^ (mm_weight is None):
        raise ValueError("mm_x and mm_weight must be both provided or both None.")
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
                raise ValueError("trans_mm_weight=True expects mm_weight.shape[1] == mm_x.shape[1].")
        else:
            n2 = int(mmw.shape[1])
            if int(mmw.shape[0]) != h2:
                raise ValueError("trans_mm_weight=False expects mm_weight.shape[0] == mm_x.shape[1].")
        if not (0 < n2 <= 65536):
            raise ValueError("N2 must be in (0, 65536].")

    # 数据类型：gmm_x/gmm_weight/mm_x/mm_weight（若有）需同属于 fp16/bf16
    allowed = {torch.float16, torch.bfloat16}
    if x.dtype not in allowed or w.dtype not in allowed:
        raise ValueError("gmm_x and gmm_weight dtype must be float16 or bfloat16.")
    if x.dtype != w.dtype:
        raise ValueError("gmm_x and gmm_weight dtype must be the same.")
    if mm_x is not None and mm_weight is not None:
        mmx = torch.as_tensor(mm_x)
        mmw = torch.as_tensor(mm_weight)
        if mmx.dtype not in allowed or mmw.dtype not in allowed:
            raise ValueError("mm_x and mm_weight dtype must be float16 or bfloat16.")
        if mmx.dtype != x.dtype or mmw.dtype != x.dtype:
            raise ValueError("mm_x/mm_weight dtype must match gmm_x dtype.")

    return (gmm_x,gmm_weight,send_counts_tensor,recv_counts_tensor,
            mm_x,mm_weight,group,ep_world_size,send_counts,recv_counts,
            trans_gmm_weight,trans_mm_weight,kwargs)