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
            "moe_distribute_combine_add_rms_norm": "moe_distribute_combine_add_rms_norm_golden"
        }
}

# TODO moe_distribute_combine_add_rms_norm_golden待实现
def moe_distribute_combine_add_rms_norm_golden(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales, residual_x, gamma, tp_send_counts = None, x_active_mask = None, activation_scale = None, weight_scale = None, group_list = None, expand_scales = None, shared_expert_x = None, elastic_info = None, ori_x = None, const_expert_alpha_1 = None, const_expert_alpha_2 = None, const_expert_v = None, group_ep = None, ep_world_size = None, ep_rank_id = None, moe_expert_num = None, group_tp = '', tp_world_size = 0, tp_rank_id = 0, expert_shard_type = 0, shared_expert_num = 0, shared_expert_rank_num = 0, global_bs = 0, out_dtype = 0, comm_quant_mode = 0, group_list_type = 0, comm_alg = '', norm_eps = 1e-6, zero_expert_num = 0, copy_expert_num = 0, const_expert_num = 0, **kwargs):
    pass