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
            "batch_mat_mul_reduce_scatter_allto_all": "batch_mat_mul_reduce_scatter_allto_all_golden"
        }
}

# TODO batch_mat_mul_reduce_scatter_allto_all_golden待实现
def batch_mat_mul_reduce_scatter_allto_all_golden(x, weight, bias = None, group_ep = None, group_tp = None, ep_world_size = None, tp_world_size = None, y_shard_type = 0, transpose_weight = False, **kwargs):
    pass