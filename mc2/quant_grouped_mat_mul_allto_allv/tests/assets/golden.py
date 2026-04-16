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
            "quant_grouped_mat_mul_allto_allv": "quant_grouped_mat_mul_allto_allv_golden"
        }
}

# TODO quant_grouped_mat_mul_allto_allv_golden待实现
def quant_grouped_mat_mul_allto_allv_golden(gmm_x, gmm_weight, send_counts_tensor = None, recv_counts_tensor = None, mm_x = None, mm_weight = None, gmmXScale = None, gmmWeightScale = None, mmXScaleOptional = None, mmWeightScaleOptional = None, commQuantScaleOptional = None, group = None, ep_world_size = None, send_counts = None, recv_counts = None, trans_gmm_weight = False, trans_mm_weight = False, gmm_x_quant_mode = 0, gmm_weight_quant_mode = 0, mm_x_quant_mode = 0, mm_weight_quant_mode = 0, comm_quant_mode = 0, group_size = 0, comm_quant_dtype = 0, y_dtype = 28, mm_dtype = 28):       # int=28 : static_cast<int>(ge::DT_UNDEFINED)
    pass