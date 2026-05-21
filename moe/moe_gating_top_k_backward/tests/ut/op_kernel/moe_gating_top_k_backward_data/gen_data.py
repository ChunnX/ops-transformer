#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
import random
import numpy
import torch


def gen_case(m, n, k, dtype=torch.bfloat16, seed=None):
    if seed:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None

    shape_mn = (m, n)
    shape_mk = (m, k)

    x_norm = torch.rand(shape_mn, dtype=torch.float32, generator=gen)
    grad_y_raw = torch.rand(shape_mk, dtype=dtype, generator=gen)
    grad_y = grad_y_raw * 2 - 1

    base_perm = torch.arange(n)
    matrix = base_perm.repeat(m, 1)

    random_noise = torch.rand(m, n, generator=gen)
    shuffle_idx = torch.argsort(random_noise, dim=1)
    shuffled_matrix = torch.gather(matrix, dim=1, index=shuffle_idx)
    expert_idx = shuffled_matrix[:, :k]

    return x_norm, grad_y, expert_idx


def gen_data(params):
    m = params["m"]
    n = params["n"]
    k = params["k"]
    dtype = params["dtype"]
    seed = params["seed"]

    x_norm, grad_y, expert_idx = gen_case(m, n, k, dtype, seed)

    x_norm.numpy().tofile("x_norm.bin")
    grad_y.numpy().tofile("grad_y.bin")
    expert_idx.numpy().tofile("expert_idx.bin")


params_info = {
    "test_moe_gating_top_k_backward_case_00": {
        "m": 1,
        "n": 8,
        "k": 4,
        "dtype": torch.bfloat16,
        "seed": 44,
    },
}


if __name__ == "__main__":
    params_list = params_info[sys.argv[1]]
    gen_data(params_list)
