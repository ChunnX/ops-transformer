#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import torch

__golden__ = {
    "kernel": {"mhc_post": "mhc_post_golden"},
    "aclnn": {"aclnnMhcPost": "aclnn_mhc_post_golden"}
}


def _mhc_post_golden_impl(x_f32, h_res_f32, h_out_f32, h_post_f32):
    h_post_term = h_post_f32.unsqueeze(-1) * h_out_f32.unsqueeze(-2)

    h_comb_term = h_post_term
    for i in range(h_res_f32.size(-2)):
        a = h_res_f32[..., i, :].unsqueeze(-1)
        b = x_f32[..., i, :].unsqueeze(-2)
        h_comb_term = torch.addcmul(h_comb_term, a, b, value=1.0)

    return h_comb_term


def mhc_post_golden(x, h_res, h_out, h_post, **kwargs):
    dtype = x.dtype

    x_f32 = torch.from_numpy(x.astype(np.float32))
    h_res_f32 = torch.from_numpy(h_res.astype(np.float32))
    h_out_f32 = torch.from_numpy(h_out.astype(np.float32))
    h_post_f32 = torch.from_numpy(h_post.astype(np.float32))

    y = _mhc_post_golden_impl(x_f32, h_res_f32, h_out_f32, h_post_f32)
    return y.numpy().astype(dtype)


def aclnn_mhc_post_golden(x, hRes, hOut, hPost, out, **kwargs):
    '''
    Aclnn golden for aclnnMhcPost.
    All the parameters (name & order) follow
    function `aclnnMhcPostGetWorkspaceSize` in @aclnn_mhc_post.h
    without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch,
    the Tensors in the parameters are all torch.Tensor.
    Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    '''

    def _to_numpy(t):
        if hasattr(t, 'numpy'):
            return t.numpy()
        return t

    x_np = _to_numpy(x)
    hRes_np = _to_numpy(hRes)
    hOut_np = _to_numpy(hOut)
    hPost_np = _to_numpy(hPost)

    result = mhc_post_golden(x_np, hRes_np, hOut_np, hPost_np, **kwargs)

    if hasattr(x, 'numpy'):
        return torch.from_numpy(result)
    return result
