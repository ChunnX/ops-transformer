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
            "allto_allv_quant_grouped_mat_mul": "allto_allv_quant_grouped_mat_mul_golden"
        }
}


# TODO allto_allv_quant_grouped_mat_mul_golden待实现
def allto_allv_quant_grouped_mat_mul_golden(gmm_x,gmm_weight,send_counts_tensor,
                                            recv_counts_tensor,mm_x,mm_weight,
                                            gmm_x_scale,gmm_weight_scale,mm_x_scale,
                                            mm_weight_scale,group,ep_world_size,
                                            send_counts,recv_counts,gmm_x_quant_mode,gmm_weight_quant_mode,mm_x_quant_mode,
                                            mm_weight_quant_mode,group_size,trans_gmm_weight:bool=False,
                                            trans_mm_weight:bool=False,permute_out_flag:bool=False,
                                            y_dtype:int=28,
                                            mm_dtype:int=28,**kwargs):          # 28 : static_cast<int>(ge::DT_UNDEFINED)
    pass