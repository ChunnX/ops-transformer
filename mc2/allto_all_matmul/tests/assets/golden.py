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
__golden__ = {
        "kernel": {
            "allto_all_matmul": "allto_all_matmul_golden"
        }
}


# TODO allto_all_matmul_golden待实现
def allto_all_matmul_golden(x1,x2,bias,x1_scale,x2_scale,comm_scale,x1_offset,x2_offset,
                            group,world_size,all2all_axes:list = {-2,-1},
                            y_dtype:int = 28,x1_quant_mode:int = 0,
                            x2_quant_mode:int =0,comm_quant_mode:int = 0,x1_quant_dtype:int = 28,
                            comm_quant_dtype:int = 28,transpose_x1:bool=False,transpose_x2:bool=False,
                            group_size:int=0,alltoall_out_flag:bool=True,**kwargs):     # 28 : static_cast<int>(ge::DT_UNDEFINED)
    pass    
