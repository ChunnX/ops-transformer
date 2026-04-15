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
def matmul_all_reduce_add_rms_norm_golden(x1,x2,bias,residual,gamma,antiquant_scale,
                                          antiquant_offset,dequant_scale,group,reduce_op:str='sum',
                                          is_trans_a:bool=False,is_trans_b:bool=False,comm_turn:int=0,
                                          antiquant_group_size:int=0,epsilon:float=1e-6,**kwargs):   # Attr("epsilon").AttrType(OPTIONAL).Float(1e-6f);
    pass