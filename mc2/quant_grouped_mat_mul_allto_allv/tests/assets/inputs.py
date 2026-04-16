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
            "quant_grouped_mat_mul_allto_allv": "quant_grouped_mat_mul_allto_allv_inputs"
        }
}

import torch

def quant_grouped_mat_mul_allto_allv_inputs(gmm_x, gmm_weight, send_counts_tensor = None, recv_counts_tensor = None, mm_x = None, mm_weight = None, gmmXScale = None, gmmWeightScale = None, mmXScaleOptional = None, mmWeightScaleOptional = None, commQuantScaleOptional = None, group = None, ep_world_size = None, send_counts = None, recv_counts = None, trans_gmm_weight = False, trans_mm_weight = False, gmm_x_quant_mode = 0, gmm_weight_quant_mode = 0, mm_x_quant_mode = 0, mm_weight_quant_mode = 0, comm_quant_mode = 0, group_size = 0, comm_quant_dtype = 0, y_dtype = 28, mm_dtype = 28, **kwargs):
    if not isinstance(group, str):
        raise TypeError("group must be a string.")
    if not (0 < len(group) < 128):
        raise ValueError("group length must be in [1, 128).")

    int_fields = {
        "ep_world_size": ep_world_size,
        "gmm_x_quant_mode": gmm_x_quant_mode,
        "gmm_weight_quant_mode": gmm_weight_quant_mode,
        "mm_x_quant_mode": mm_x_quant_mode,
        "mm_weight_quant_mode": mm_weight_quant_mode,
        "comm_quant_mode": comm_quant_mode,
        "group_size": group_size,
        "comm_quant_dtype": comm_quant_dtype,
        "y_dtype": y_dtype,
        "mm_dtype": mm_dtype,
    }
    for name, value in int_fields.items():
        if int(value) != value:
            raise ValueError(f"{name} must be an integer.")

    ep_ws = int(ep_world_size)
    if ep_ws not in {2, 4, 8, 16, 32, 64, 128, 256}:
        raise ValueError("ep_world_size must be one of {2,4,8,16,32,64,128,256}.")
    if int(group_size) != 0:
        raise ValueError("group_size currently only supports 0.")
    if int(comm_quant_mode) != 0:
        raise ValueError("comm_quant_mode currently only supports 0.")
    if int(comm_quant_dtype) not in (-1, 0):
        raise ValueError("comm_quant_dtype must be -1 (README) or 0 (def default).")
    if int(y_dtype) not in (1, 27, 28):
        raise ValueError("y_dtype must be one of {1, 27, 28}.")
    if int(mm_dtype) not in (1, 27, 28):
        raise ValueError("mm_dtype must be one of {1, 27, 28}.")

    # 量化模式：README 要求 1/6；同时兼容 def 默认值 0
    if int(gmm_x_quant_mode) not in (0, 1, 6) or int(gmm_weight_quant_mode) not in (0, 1, 6):
        raise ValueError("gmm_x_quant_mode and gmm_weight_quant_mode must be in {0,1,6}.")

    gx_t = torch.as_tensor(gmm_x)
    gw_t = torch.as_tensor(gmm_weight)
    if gx_t.dim() != 2:
        raise ValueError("gmm_x must be 2-D with shape (A, H1).")
    if gw_t.dim() != 3:
        raise ValueError("gmm_weight must be 3-D with shape (e, H1/N1, N1/H1).")

    e = int(gw_t.shape[0])
    if not (1 <= e <= 32):
        raise ValueError("expert count e must be in [1, 32].")
    if e * ep_ws > 256:
        raise ValueError("e * ep_world_size must be <= 256.")

    a = int(gx_t.shape[0])
    h1 = int(gx_t.shape[1])
    if not (0 < h1 < 65536):
        raise ValueError("H1 must be in (0, 65536).")

    if not trans_gmm_weight:
        if int(gw_t.shape[1]) != h1:
            raise ValueError("trans_gmm_weight=False expects gmm_weight.shape[1] == H1.")
        n1 = int(gw_t.shape[2])
    else:
        if int(gw_t.shape[2]) != h1:
            raise ValueError("trans_gmm_weight=True expects gmm_weight.shape[2] == H1.")
        n1 = int(gw_t.shape[1])
    if not (0 < n1 < 65536):
        raise ValueError("N1 must be in (0, 65536).")

    send_counts_l = [int(v) for v in list(send_counts)]
    recv_counts_l = [int(v) for v in list(recv_counts)]
    if len(send_counts_l) != e * ep_ws or len(recv_counts_l) != e * ep_ws:
        raise ValueError("send_counts/recv_counts length must be e * ep_world_size.")
    if any(v < 0 for v in send_counts_l + recv_counts_l):
        raise ValueError("send_counts/recv_counts values must be >= 0.")
    if sum(send_counts_l) != a:
        raise ValueError("sum(send_counts) must equal gmm_x.shape[0] (A).")
    recv_sum = sum(recv_counts_l)
    if not (0 < recv_sum < 52428800):
        raise ValueError("BSK=sum(recv_counts) must be in (0, 52428800).")

    # dtype 检查：float8/float4 通过 metadata 兼容
    lowbit_names = {"hifloat8", "float8_e4m3fn", "float8_e5m2", "float4_e2m1", "float4_e1m2"}
    def _is_allowed_input_dtype(t, meta_key):
        return t.dtype in (torch.float16, torch.bfloat16) or str(kwargs.get(meta_key, "")).lower() in lowbit_names

    if not _is_allowed_input_dtype(gx_t, "gmm_x_dtype"):
        raise ValueError("gmm_x dtype must be float16/bfloat16 or low-bit via gmm_x_dtype metadata.")
    if not _is_allowed_input_dtype(gw_t, "gmm_weight_dtype"):
        raise ValueError("gmm_weight dtype must be float16/bfloat16 or low-bit via gmm_weight_dtype metadata.")

    gxs_t = torch.as_tensor(gmmXScale)
    gws_t = torch.as_tensor(gmmWeightScale)
    gxs_is_mx = (str(kwargs.get("gmmXScale_dtype", "")).lower() == "float8_e8m0") or int(gmm_x_quant_mode) == 6
    gws_is_mx = (str(kwargs.get("gmmWeightScale_dtype", "")).lower() == "float8_e8m0") or int(gmm_weight_quant_mode) == 6
    if gxs_t.dtype != torch.float32 and not gxs_is_mx:
        raise ValueError("gmmXScale dtype must be float32 (or float8_e8m0 via metadata/mx mode).")
    if gws_t.dtype != torch.float32 and not gws_is_mx:
        raise ValueError("gmmWeightScale dtype must be float32 (or float8_e8m0 via metadata/mx mode).")

    if int(gmm_x_quant_mode) == 1:
        if gxs_t.numel() != 1:
            raise ValueError("pertensor gmmXScale must have one element.")
    elif int(gmm_x_quant_mode) == 6:
        if tuple(gxs_t.shape) != (a, (h1 + 63) // 64, 2):
            raise ValueError("mx gmmXScale shape must be (A, ceil(H1/64), 2).")

    if int(gmm_weight_quant_mode) == 1:
        if gws_t.numel() != 1:
            raise ValueError("pertensor gmmWeightScale must have one element.")
    elif int(gmm_weight_quant_mode) == 6:
        if not trans_gmm_weight:
            expect = (e, n1, (h1 + 63) // 64, 2)
        else:
            expect = (e, (h1 + 63) // 64, n1, 2)
        if tuple(gws_t.shape) != expect:
            raise ValueError(f"mx gmmWeightScale shape must be {expect}.")

    # 共享专家分支：mm_x/mm_weight 可以都为空；若启用则需成对出现并检查
    has_mm_x = mm_x is not None and torch.as_tensor(mm_x).numel() > 0
    has_mm_w = mm_weight is not None and torch.as_tensor(mm_weight).numel() > 0
    if has_mm_x != has_mm_w:
        raise ValueError("mm_x and mm_weight must be both provided or both empty.")

    if has_mm_x:
        mx_t = torch.as_tensor(mm_x)
        mw_t = torch.as_tensor(mm_weight)
        if mx_t.dim() != 2 or mw_t.dim() != 2:
            raise ValueError("mm_x and mm_weight must be 2-D when provided.")
        if not _is_allowed_input_dtype(mx_t, "mm_x_dtype"):
            raise ValueError("mm_x dtype must be float16/bfloat16 or low-bit via mm_x_dtype metadata.")
        if not _is_allowed_input_dtype(mw_t, "mm_weight_dtype"):
            raise ValueError("mm_weight dtype must be float16/bfloat16 or low-bit via mm_weight_dtype metadata.")

        h2 = int(mx_t.shape[1])
        if not (0 < h2 <= 12288):
            raise ValueError("H2 must be in (0, 12288].")
        if not trans_mm_weight:
            if int(mw_t.shape[0]) != h2:
                raise ValueError("trans_mm_weight=False expects mm_weight.shape[0] == H2.")
            n2 = int(mw_t.shape[1])
        else:
            if int(mw_t.shape[1]) != h2:
                raise ValueError("trans_mm_weight=True expects mm_weight.shape[1] == H2.")
            n2 = int(mw_t.shape[0])
        if not (0 < n2 < 65536):
            raise ValueError("N2 must be in (0, 65536).")

        if int(mm_x_quant_mode) not in (1, 6) or int(mm_weight_quant_mode) not in (1, 6):
            raise ValueError("mm_x_quant_mode and mm_weight_quant_mode must be 1 or 6 when mm branch is enabled.")

        mmxs_t = torch.as_tensor(mmXScaleOptional)
        mmws_t = torch.as_tensor(mmWeightScaleOptional)
        if mmxs_t.numel() == 0 or mmws_t.numel() == 0:
            raise ValueError("mmXScaleOptional and mmWeightScaleOptional must be provided when mm branch is enabled.")

        if int(mm_x_quant_mode) == 1:
            if mmxs_t.numel() != 1:
                raise ValueError("pertensor mmXScaleOptional must have one element.")
        else:
            expect = (int(mx_t.shape[0]), (h2 + 63) // 64, 2)
            if tuple(mmxs_t.shape) != expect:
                raise ValueError(f"mx mmXScaleOptional shape must be {expect}.")

        if int(mm_weight_quant_mode) == 1:
            if mmws_t.numel() != 1:
                raise ValueError("pertensor mmWeightScaleOptional must have one element.")
        else:
            if not trans_mm_weight:
                expect = (n2, (h2 + 63) // 64, 2)
            else:
                expect = ((h2 + 63) // 64, n2, 2)
            if tuple(mmws_t.shape) != expect:
                raise ValueError(f"mx mmWeightScaleOptional shape must be {expect}.")
    else:
        # mm 分支未启用时，量化 mode 建议维持默认 0
        if int(mm_x_quant_mode) != 0 or int(mm_weight_quant_mode) != 0:
            raise ValueError("mm_x_quant_mode/mm_weight_quant_mode must be 0 when mm branch is disabled.")

    if commQuantScaleOptional is not None and torch.as_tensor(commQuantScaleOptional).numel() > 0:
        raise ValueError("commQuantScaleOptional is reserved and must be empty when comm_quant_mode=0.")

    if send_counts_tensor is not None and torch.as_tensor(send_counts_tensor).numel() > 0:
        raise ValueError("send_counts_tensor is reserved and must be empty (README).")
    if recv_counts_tensor is not None and torch.as_tensor(recv_counts_tensor).numel() > 0:
        raise ValueError("recv_counts_tensor is reserved and must be empty (README).")

    soc = str(kwargs.get("soc", "")).lower()
    if soc and soc not in {"950", "ascend950", "950pr", "950dt"}:
        raise ValueError("quant_grouped_mat_mul_allto_allv is only supported on Ascend 950 series.")

    return (gmm_x,gmm_weight,send_counts_tensor,recv_counts_tensor,mm_x,mm_weight,gmmXScale,
            gmmWeightScale,mmXScaleOptional,mmWeightScaleOptional,commQuantScaleOptional,group,
            ep_world_size,send_counts,recv_counts,trans_gmm_weight,trans_mm_weight,
            gmm_x_quant_mode,gmm_weight_quant_mode,mm_x_quant_mode,mm_weight_quant_mode,
            comm_quant_mode,group_size,comm_quant_dtype,y_dtype,mm_dtype,kwargs)