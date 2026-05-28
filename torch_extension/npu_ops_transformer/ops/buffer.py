# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY
from .dispatch_layout import npu_dispatch_layout
from .dispatch_prefill import npu_moe_dispatch_prefill
from .combine_prefill import npu_moe_combine_prefill
from .update_context import update_context
# from .update_context_by_hccl_channel import update_context_by_hccl_channel


class IntWrapper:
    def __init__(self, value=0):
        self.value = value


class Buffer:
    def __init__(self, group, ccl_buffer_size: int = 0):
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.world_size = torch.distributed.get_world_size(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank_id, init_comm=False)
        self.local_ranksize = 8     # TODO
        self.round = 1  # TODO
        self.per_round_tokens = 8192    # TODO
        self.quant_mode = 0 # TODO
        context_struct_size = (2 + 1024)
        context_tensor_size = ((context_struct_size * 8) + 3) // 4
        self.context = torch.zeros(context_tensor_size, dtype=torch.int32).npu()
        self.ccl_buffer_size = IntWrapper()
        res = update_context(self.group_name, self.world_size, self.ccl_buffer_size, self.context)    #TODO deepep
        # update_context_by_hccl_channel(self.group_name, self.ep_world_size, self.ccl_buffer_size, self.context) #TODO megamoe

    @staticmethod
    def get_normal_ccl_buffer_size(world_size: int, num_max_dispatch_tokens_per_rank: int, hidden: int,
                                        num_moe_expert: int, topk: int, num_shared_expert: int = 0,
                                        num_shared_expert_ranks: int = 0, comm_alg: str = "",
                                        ) -> int:
        def inline_align(value, base):
            return (value + base - 1) // base * base
        torch._check(((world_size >= 2) and (world_size <= 768)),   # world_size当前仅支持[2,768]
                     lambda: (f"world_size only support in [2, 768], but got {world_size=}."))
        torch._check(((hidden >= 1024) and (hidden <= 8192)),       # hidden当前仅支持[1024,8192]
                     lambda: (f"hidden only support in [1024, 8192], but got {hidden=}."))
        # num_max_dispatch_tokens_per_rank当前仅支持[1,512]
        torch._check(((num_max_dispatch_tokens_per_rank >= 1) and (num_max_dispatch_tokens_per_rank <= 512)),
                     lambda: (f"num_max_dispatch_tokens_per_rank only support in [1, 512], "
                              f"but got {num_max_dispatch_tokens_per_rank=}."))
        torch._check(((num_moe_expert >= 1) and (num_moe_expert <= 1024)),       # num_moe_expert当前仅支持[1,1024]
                     lambda: (f"num_moe_expert only support in [1, 1024], but got {num_moe_expert=}."))
        torch._check(((topk >= 1) and (topk <= 16)),                # topk当前仅支持[1,16]
                     lambda: (f"topk only support in [1, 16], but got {topk=}."))
        torch._check(((num_shared_expert >= 0) and (num_shared_expert <= 4)),   # num_shared_expert当前仅支持[0,4]
                     lambda: (f"num_shared_expert only support in [0, 4], but got {num_shared_expert=}."))
        torch._check((world_size - num_shared_expert_ranks > 0),                # 至少存在一张moe专家卡
                     lambda: (f"world_size - num_shared_expert_ranks must be greater than 0, "
                              f"but got {world_size=} {num_shared_expert_ranks=}."))
        # local_moe_expert_num*world_size仅支持(0，2048]
        local_moe_expert_num = num_moe_expert // (world_size - num_shared_expert_ranks)
        torch._check(((local_moe_expert_num * world_size > 0) and (local_moe_expert_num * world_size <= 2048)),
            lambda: (f"local_moe_expert_num * world_size only support in (0, 2048], "
                     f"but got {world_size=} {num_shared_expert_ranks=}, "
                     f"local_moe_expert_num = num_moe_expert // (world_size - num_shared_expert_ranks), "
                     f"{local_moe_expert_num=}"))

        max_out_dtype_size = 2 # sizeof(int32)
        mb_conversion = 1024 * 1024
        ub_align = 32 # 32B
        scale_expand_index_buffer = 44 # scale 32B + 3 * 4 expand_idx
        full_mesh_data_align = 480
        win_addr_align = 512

        comm_alg_support_list = ["fullmesh_v1", "fullmesh_v2", ""]
        torch._check(comm_alg in comm_alg_support_list,
                     lambda: (f"comm_alg only support {comm_alg_support_list=}, but got {comm_alg=}."))

        token_actual_len = inline_align(hidden * max_out_dtype_size, ub_align) + scale_expand_index_buffer
        if (comm_alg == "fullmesh_v2"):
            token_need_size_dispatch = inline_align(token_actual_len, full_mesh_data_align) // full_mesh_data_align * \
                win_addr_align
        else:
            token_need_size_dispatch = inline_align(token_actual_len, win_addr_align)
        token_need_size_combine = inline_align(hidden * max_out_dtype_size, win_addr_align)
        minimum_buffer_size = 2 * (
            (num_max_dispatch_tokens_per_rank * token_need_size_dispatch * world_size * local_moe_expert_num) + \
            (num_max_dispatch_tokens_per_rank * token_need_size_combine * (topk + num_shared_expert))) + mb_conversion
        ## hccl按照2*bufferSize大小开设
        ccl_buffer_size = inline_align(inline_align(minimum_buffer_size, mb_conversion) // mb_conversion, 2) // 2
        return ccl_buffer_size

    def update_ctx(self, new_group) -> bool:
        self.group = new_group
        self.rank_id = torch.distributed.get_rank(new_group)
        self.group_name = new_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank_id, init_comm=False)
        new_world_size = torch.distributed.get_world_size(new_group)
        torch._check(new_world_size == self.world_size,
                    lambda: (f"New world size should be the same as orginal world size, "
                            f"but got {new_world_size=}, orginial={self.world_size}"
                            f"{ops_error(ErrCode.VALUE)}."),)
        res = update_context(self.group_name, self.world_size, self.ccl_buffer_size, self.context)
        return res

    def npu_get_dispatch_layout(self, topk_idx, num_experts: int, previous_event=None, async_finish=False,
                                allocate_on_comm_stream=False):
        self.num_experts = num_experts
        self.num_tokens = topk_idx.size(0)  #TODO
        self.num_topk = topk_idx.size(1)    #TODO
        (num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, ext_handle_data) \
            = torch.ops.npu_ops_transformer.npu_dispatch_layout(
                                             topk_idx=topk_idx,
                                             num_ranks=self.world_size,
                                             num_experts=num_experts,
                                             local_ranksize=self.local_ranksize,
                                             per_round_tokens=self.per_round_tokens, 
                                             rank_id=self.rank_id)
        event = None
        (notify_send_data, send_token_idx_small) = ext_handle_data
        self.ext_handle_data = ext_handle_data
        self.notify_send_data = notify_send_data
        self.send_token_idx_small = send_token_idx_small
        self.notify_send_data_size = notify_send_data.size(0)
        return num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, EventOverlap(event)
    
    def npu_dispatch(self, x, handle=None, num_tokens_per_rank=None, num_tokens_per_rdma_rank=None, is_token_in_rank=None,
                     num_tokens_per_expert=None, topk_idx=None, topk_weights=None, expert_alignment=1, num_worst_tokens=0,
                     config=None, previous_event=None, async_finish=False, allocate_on_comm_stream=False,
                     dispatch_wait_recv_cost_stats=None):
        torch._check(handle != None,
            lambda: (f"Optional communication handle is not supported yet, "
                    f"{ops_error(ErrCode.VALUE)}."),)
        (recv_x, dynamic_scales, expand_idx, recv_count, ext_handle_data) \
            = torch.ops.npu_ops_transformer.npu_moe_dispatch_prefill(
                                            context=self.context,
                                            x=x,
                                            topk_idx=topk_idx,
                                            ext_handle_data=self.ext_handle_data,
                                            ep_world_size=self.world_size,
                                            ep_rank_id=self.rank_id,
                                            local_ranksize=self.local_ranksize,
                                            moe_expert_num=self.num_experts,
                                            ccl_buffer_size=self.ccl_buffer_size.value,
                                            x_scales=None,  # TODO
                                            topk_weights=topk_weights,
                                            num_tokens_per_rank=num_tokens_per_rank,
                                            is_token_in_rank=is_token_in_rank,
                                            num_tokens_per_expert=num_tokens_per_expert,
                                            dispatch_wait_recv_cost_stats=dispatch_wait_recv_cost_stats
                                            num_worst_tokens=num_worst_tokens,
                                            # config=config,
                                            quant_mode=self.quant_mode,   #TODO
                                            real_max_bs=real_max_bs,
                                            round=self.round,    #TODO
                                            per_round_tokens=self.per_round_tokens)     #TODO
        # handle = (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank,
        #           send_head, topk_idx, topk_weights, )
        if (quant_mode > 0):
            recv_x = (recv_x, dynamic_scales)
        num_recv_tokens_per_expert_list = []
        send_head = recv_count.sum(dim=0, keepdim=False)
        handle = (None, None, None, expand_idx, is_token_in_rank, send_head, topk_idx, topk_weights)
        event = None
        return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event
        # return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, EventOverlap(event)

    def npu_combine(self, x, handle, topk_weights=None, bias=None, config=None, previous_event=None,
                    async_finish=False, allocate_on_comm_stream=False, combine_send_cost_stats=None):
        (_, _, _, src_idx, is_recv_token_in_rank, send_head, topk_idx, topk_weights_ori) = handle
        recv_x = torch.ops.npu_ops_transformer.npu_moe_combine_prefill(
                                               context=self.context,
                                               recv_x=x,
                                               token_src_info=src_idx,
                                               ep_recv_counts=send_head,
                                               topk_weights=topk_weights_ori,
                                               token_idx=topk_idx,
                                               ep_world_size=self.world_size,
                                               ep_rank_id=self.rank_id,
                                               moe_expert_num=self.num_experts,
                                               ccl_buffer_size=self.ccl_buffer_size.value,
                                               combine_send_cost_stats=combine_send_cost_stats,
                                               real_max_bs=real_max_bs,
                                               round=self.round,    #TODO
                                               per_round_tokens=self.per_round_tokens)     #TODO
        recv_topk_weights = None
        event = None
        return recv_x, recv_topk_weights, event
        # return recv_x, recv_topk_weights, EventOverlap(event)

  