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
import numpy as np

__input__ = {
    "kernel": {
        "stem_oam_prep_varlen_q": "stem_oam_prep_varlen_q_input"
    }
}


def stem_oam_prep_varlen_q_input(
        q,
        q_seq_lens,
        cu_seq_lens_q,
        q_scale=None,
        *,
        stemBlockSize=128,
        stemStride=16,
        **kwargs
):
    """
    Input generator for stem_oam_prep_varlen_q.

    Generates random q_scale values for FP8 input.
    """
    total_tokens = q.shape[0]
    H_q = q.shape[1]

    if q_scale is None and q.dtype == np.uint8:
        q_scale = np.random.uniform(0.5, 1.5, (total_tokens, H_q)).astype(np.float32)

    return [q, q_seq_lens, cu_seq_lens_q, q_scale]
