#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""数据工具（flash_attn harness 的 make_noncontiguous 直接复用）。"""

import torch


def make_noncontiguous(tensor, nc_dims):
    """返回指定维非连续但底层数据相同的 tensor view（同 flash_attn harness）。"""
    device = tensor.device
    dtype = tensor.dtype
    shape = tensor.shape

    if nc_dims == 0:
        pad_dim = 1
        shp = list(shape)
        shp[pad_dim] += 1
        p = torch.zeros(shp, dtype=dtype, device=device)
        ws = [slice(None)] * len(shape)
        ws[pad_dim] = slice(None, shape[pad_dim])
        p[tuple(ws)] = tensor
        return p[tuple(ws)]
    elif nc_dims == (0, 1):
        pad_dim = 2
        shp = list(shape)
        shp[pad_dim] += 1
        p = torch.zeros(shp, dtype=dtype, device=device)
        ws = [slice(None)] * len(shape)
        ws[pad_dim] = slice(None, shape[pad_dim])
        p[tuple(ws)] = tensor
        return p[tuple(ws)]
    return tensor