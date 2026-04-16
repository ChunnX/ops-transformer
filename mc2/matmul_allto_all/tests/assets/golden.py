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
            "matmul_allto_all": "matmul_allto_all_golden"
        }
}

# TODO matmul_allto_all_golden待实现
def matmul_allto_all_golden(x1, x2, bias = None, x1_scale = None, x2_scale = None, comm_scale = None, x1_offset = None, x2_offset = None, group = None, world_size = None, all2all_axes = {-1,-2}, y_dtype = 28, x1_quant_mode = 0, x2_quant_mode = 0, comm_quant_mode = 0, comm_quant_dtype = 28, transpose_x1 = False, transpose_x2 = False, group_size = 0, **kwargs):
    pass