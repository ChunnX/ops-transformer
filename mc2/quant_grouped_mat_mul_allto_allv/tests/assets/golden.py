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
def quant_grouped_mat_mul_allto_allv_golden(gmm_x,gmm_weight,send_counts_tensor,recv_counts_tensor,mm_x,mm_weight,gmmXScale,
                                            gmmWeightScale,mmXScaleOptional,mmWeightScaleOptional,commQuantScaleOptional,group,
                                            ep_world_size,send_counts,recv_counts,trans_gmm_weight:bool=False,trans_mm_weight:bool=False,
                                            gmm_x_quant_mode:int=0,gmm_weight_quant_mode:int=0,mm_x_quant_mode:int=0,mm_weight_quant_mode:int=0,
                                            comm_quant_mode:int=0,group_size:int=0,comm_quant_dtype:int=0,y_dtype:int=28,mm_dtype:int=28):       # int=28 : static_cast<int>(ge::DT_UNDEFINED)
    pass