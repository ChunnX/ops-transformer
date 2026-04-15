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
def moe_distribute_combine_add_rms_norm_golden(expand_x,expert_ids,assist_info_for_combine,ep_send_counts,expert_scales,
                                  residual_x,gamma,tp_send_counts,x_active_mask,activation_scale,weight_scale,
                                  group_list,expand_scales,shared_expert_x,elastic_info,ori_x,const_expert_alpha_1,
                                  const_expert_alpha_2,const_expert_v,group_ep,ep_world_size,ep_rank_id,
                                  moe_expert_num,group_tp:str='',tp_world_size:int=0,tp_rank_id:int=0,expert_shard_type:int=0,
                                  shared_expert_num:int=0,shared_expert_rank_num:int=0,global_bs:int=0,out_dtype:int=0,
                                  comm_quant_mode:int=0,group_list_type:int=0,comm_alg:str='',norm_eps:float=1e-6,
                                  zero_expert_num:int=0,copy_expert_num:int=0,const_expert_num:int=0,**kwargs):
    pass