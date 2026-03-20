#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""GroupedMatmulSwigluQuantV2 Input Generator"""
import numpy as np

__input__ = {"kernel": {"grouped_matmul_swiglu_quant_v2": "grouped_matmul_swiglu_quant_v2_input"}}


def grouped_matmul_swiglu_quant_v2_input(inputs, attrs, **kwargs):
    input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7 = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7]
    x1, pertoken_scale, group_list_shape, x2, input_deq_scale, weight_assist_matrix, bias, smooth_scale = input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7

    if input_deq_scale.dtype == "float8_e8m0" and pertoken_scale.dtype == "float8_e8m0":
        x1_mx_gm = np.random.uniform(127, 130, size=pertoken_scale.shape).astype(np.uint8)
        x1_mx = 2**(x1_mx_gm.astype(np.float64) - 127)
        pertoken_scale = x1_mx.astype("float8_e8m0")

        x2_mx_gm = np.random.randint(127, 130, size=input_deq_scale.shape).astype(np.uint8)
        x2_mx = 2**(x2_mx_gm.astype(np.float64) - 127)
        input_deq_scale = x2_mx.astype("float8_e8m0")

    group_num = group_list_shape.shape[0]
    group_list = []
    group_list_type = kwargs.get("group_list_type", 0)
    group_list = kwargs.get("group_list_expect", [])

    group_list_tmp = group_list
    if group_list_type == 1:
        group_list_tmp = np.cumsum(group_list)
    if group_list_tmp and group_list_tmp[-1] > x1.shape[0]:
        raise Exception('sum of grouplist: ({}) can not be greater than x1[0]: ({})'.format(group_list_tmp[-1], x1.shape[0]))
    print('input_deq_scale ', input_deq_scale)
    print('pertoken_scale ', pertoken_scale)
    return [x1, pertoken_scale, np.array(group_list), x2, input_deq_scale, weight_assist_matrix, bias, smooth_scale]
