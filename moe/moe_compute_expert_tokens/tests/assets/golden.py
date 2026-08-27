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

import numpy

__spec__ = {
    "moe_compute_expert_tokens": "MoeComputeExpertTokensTestSpec",
    "aclnnMoeComputeExpertTokens": "MoeComputeExpertTokensTestSpec",
}


class MoeComputeExpertTokensTestSpec:
    @staticmethod
    def golden(sorted_experts, num_experts=None, out=None, **kwargs):
        if num_experts is None:
            num_experts = kwargs.get("num_experts", kwargs.get("numExperts", 0))
        if hasattr(num_experts, "item"):
            num_experts = num_experts.item()
        if hasattr(sorted_experts, "numpy"):
            sorted_experts = (
                sorted_experts.detach().cpu().numpy()
                if hasattr(sorted_experts, "detach")
                else sorted_experts.numpy()
            )
        elif not isinstance(sorted_experts, numpy.ndarray):
            sorted_experts = numpy.asarray(sorted_experts)
        sorted_experts = sorted_experts.flatten()
        res = numpy.array(
            [numpy.count_nonzero(sorted_experts <= i) for i in range(num_experts)],
            dtype=numpy.int32,
        )
        return [res]

    @staticmethod
    def customize_inputs(sorted_experts, num_experts=None, out=None, **kwargs):
        if isinstance(sorted_experts, numpy.ndarray):
            flat = sorted_experts.ravel()
            flat.sort()
        elif hasattr(sorted_experts, "numpy"):
            np_arr = (
                sorted_experts.detach().cpu().numpy()
                if hasattr(sorted_experts, "detach")
                else sorted_experts.numpy()
            )
            np_arr = numpy.sort(np_arr.flatten())
            import torch

            sorted_experts.copy_(
                torch.from_numpy(np_arr.reshape(sorted_experts.shape)).to(
                    sorted_experts.dtype
                )
            )
        return (sorted_experts,)

    tolerance = {
        "int32": {"standard": "stat_rel_err"},
    }
