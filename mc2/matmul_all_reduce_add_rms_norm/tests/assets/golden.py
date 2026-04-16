# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
__golden__ = {
        "kernel": {
            "matmul_all_reduce_add_rms_norm": "matmul_all_reduce_add_rms_norm_golden"
        }
}


# TODO matmul_all_reduce_add_rms_norm_golden待实现
def matmul_all_reduce_add_rms_norm_golden(x1, x2, bias = None, residual = None, gamma = None, antiquant_scale = None, antiquant_offset = None, dequant_scale = None, group = None, reduce_op = 'sum', is_trans_a = False, is_trans_b = False, comm_turn = 0, antiquant_group_size = 0, epsilon = 1e-6, **kwargs):   # Attr("epsilon").AttrType(OPTIONAL).Float(1e-6f);
    pass