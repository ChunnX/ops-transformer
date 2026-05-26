import torch
import torch_npu
import torch.distributed as dist
from torch.multiprocessing import Process, Lock, Value, Queue
import argparse
import torchair as tng
import random

from mindspeed.ops.npu_bmm_reduce_scatter_all_to_all import npu_bmm_reducescatter_alltoall
from mindspeed.ops.npu_all_to_all_all_gather_bmm import npu_alltoall_allgather_bmm
import numpy as np


# 定义模型model
# add_result、mm1在默认stream，mm_result在流“1”，add2在流“2”
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, op1, op2, arg1, arg2):
        # if True:
        with tng.scope.npu_stream_switch('1'):
            if op1 == 'matmul':
                output1 = torch.matmul(arg1['input'], arg1['weight'])
            elif op1 == 'bmm_reducescatter_alltoall':
                output1 = npu_bmm_reducescatter_alltoall(arg1['x_npu'],
                                               arg1['weight_npu'],
                                               arg1['ep_group'],
                                               arg1['ep_size'],
                                               arg1['tp_group'],
                                               arg1['tp_size'],
                                               bias=arg1['bias_npu'],
                                               shard_type=arg1['y_shard_type'])
            elif op1 == 'alltoall_allgather_bmm':
                output1 = npu_alltoall_allgather_bmm(arg1['x_npu'],
                                           arg1['weight_npu'],
                                           arg1['ep_group'],
                                           arg1['ep_size'],
                                           arg1['tp_group'],
                                           arg1['tp_size'],
                                           bias=arg1['bias_npu'],
                                           shard_type=arg1['x_shard_type'],
                                           act_type=arg1['act_type'],
                                           need_allgather_out=arg1['out_y2_flag'],
                                           need_activation_feature=arg1['out_y3_flag'])
            elif op1 == 'alltoallv_gmm':
                output1 = torch_npu.npu_alltoallv_gmm(gmm_x=arg1['gmm_x'],
                                            gmm_weight=arg1['gmm_weight'],
                                            hcom=arg1['hcom'],
                                            ep_world_size=arg1['ep_world_size'],
                                            send_counts=list(arg1['send_counts']),
                                            recv_counts=list(arg1['recv_counts']),
                                            send_counts_tensor=None,  # send_counts_tensor,
                                            recv_counts_tensor=None,  # recv_counts_tensor,
                                            mm_x=arg1['mm_x'],
                                            mm_weight=arg1['mm_weight'],
                                            trans_gmm_weight=arg1['trans_gmm_weight'],
                                            trans_mm_weight=arg1['trans_mm_weight'],
                                            permute_out_flag=arg1['permute_out_flag'])
            elif op1 == 'gmm_alltoallv':
                output1 = torch_npu.npu_gmm_alltoallv(gmm_x=arg1['gmm_x'],
                                            gmm_weight=arg1['gmm_weight'],
                                            hcom=arg1['hcom'],
                                            ep_world_size=arg1['ep_world_size'],
                                            send_counts=list(arg1['send_counts']),
                                            recv_counts=list(arg1['recv_counts']),
                                            send_counts_tensor=None,  # send_counts_tensor,
                                            recv_counts_tensor=None,  # recv_counts_tensor,
                                            mm_x=arg1['mm_x'],
                                            mm_weight=arg1['mm_weight'],
                                            trans_gmm_weight=arg1['trans_gmm_weight'],
                                            trans_mm_weight=arg1['trans_mm_weight'])
            elif op1 == 'all_gather_base_mm':
                output1 = torch_npu.npu_all_gather_base_mm(arg1['input'], arg1['weight'], arg1['hcomm_info'], arg1['world_size'])
            elif op1 == 'mm_reduce_scatter_base':
                output1 = torch_npu.npu_mm_reduce_scatter_base(arg1['input'], arg1['weight'], arg1['hcomm_info'], arg1['world_size'])
            elif op1 == 'moe_distribute_combine':
                output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch(x=arg1['x'],
                                                                            expert_ids=arg1['expert_ids'],
                                                                            group_ep=arg1['group_ep'],
                                                                            group_tp=arg1['group_tp'],
                                                                            ep_world_size=arg1['ep_world_size'],
                                                                            tp_world_size=arg1['tp_world_size'],
                                                                            ep_rank_id=arg1['ep_rank_id'],
                                                                            tp_rank_id=arg1['tp_rank_id'],
                                                                            expert_shard_type=arg1['expert_shard_type'],
                                                                            shared_expert_rank_num=arg1['shared_expert_rank_num'],
                                                                            moe_expert_num=arg1['moe_expert_num'],
                                                                            scales=arg1['scales'],
                                                                            quant_mode=arg1['quant_mode'],
                                                                            global_bs=arg1['global_bs'])

                expand_x_npu, _, expand_idx_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
                input_dtype = torch.bfloat16
                if expand_x_npu.dtype == torch.int8:
                    expand_x_npu = expand_x_npu.to(input_dtype)

                output1 = torch_npu.npu_moe_distribute_combine(expand_x=expand_x_npu,
                                                                          expert_ids=arg1['expert_ids'],
                                                                          expand_idx=expand_idx_npu,
                                                                          ep_send_counts=ep_recv_counts_npu,
                                                                          tp_send_counts=tp_recv_counts_npu,
                                                                          expert_scales=arg1['expert_scales'],
                                                                          group_ep=arg1['group_ep'],
                                                                          group_tp=arg1['group_tp'],
                                                                          ep_world_size=arg1['ep_world_size'],
                                                                          tp_world_size=arg1['tp_world_size'],
                                                                          ep_rank_id=arg1['ep_rank_id'],
                                                                          tp_rank_id=arg1['tp_rank_id'],
                                                                          expert_shard_type=arg1['expert_shard_type'],
                                                                          shared_expert_rank_num=arg1['shared_expert_rank_num'],
                                                                          moe_expert_num=arg1['moe_expert_num'],
                                                                          global_bs=arg1['global_bs'])
            elif op1 == 'moe_distribute_combine_v2':
                output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(
                    x=arg1['x'],
                    expert_ids=arg1['expert_ids'],
                    group_ep=arg1['group_ep'],
                    group_tp=arg1['group_tp'],
                    ep_world_size=arg1['ep_world_size'],
                    tp_world_size=arg1['tp_world_size'],
                    ep_rank_id=arg1['ep_rank_id'],
                    tp_rank_id=arg1['tp_rank_id'],
                    expert_shard_type=arg1['expert_shard_type'],
                    shared_expert_rank_num=arg1['shared_expert_rank_num'],
                    moe_expert_num=arg1['moe_expert_num'],
                    scales=arg1['scales'],
                    quant_mode=arg1['quant_mode'],
                    global_bs=arg1['global_bs'],
                    elastic_info=arg1['elastic_info'],
                    zero_expert_num=arg1['zero_expert_num'],
                    copy_expert_num=arg1['copy_expert_num'],
                    const_expert_num=arg1['const_expert_num']
                )
                expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
                input_dtype = torch.bfloat16
                if expand_x_npu.dtype == torch.int8:
                    expand_x_npu = expand_x_npu.to(input_dtype)


                output1 = torch_npu.npu_moe_distribute_combine_v2(
                    expand_x=expand_x_npu,
                    expert_ids=arg1['expert_ids'],
                    assist_info_for_combine=assist_info_for_combine_npu,
                    ep_send_counts=ep_recv_counts_npu,
                    tp_send_counts=tp_recv_counts_npu,
                    expert_scales=arg1['expert_scales'],
                    group_ep=arg1['group_ep'],
                    group_tp=arg1['group_tp'],
                    ep_world_size=arg1['ep_world_size'],
                    tp_world_size=arg1['tp_world_size'],
                    ep_rank_id=arg1['ep_rank_id'],
                    tp_rank_id=arg1['tp_rank_id'],
                    expert_shard_type=arg1['expert_shard_type'],
                    shared_expert_rank_num=arg1['shared_expert_rank_num'],
                    moe_expert_num=arg1['moe_expert_num'],
                    global_bs=arg1['global_bs'],
                    elastic_info=arg1['elastic_info'],
                    ori_x=arg1['x'],
                    const_expert_alpha_1=arg1['const_expert_alpha_1'],
                    const_expert_alpha_2=arg1['const_expert_alpha_2'],
                    const_expert_v=arg1['const_expert_v'],
                    zero_expert_num=arg1['zero_expert_num'],
                    copy_expert_num=arg1['copy_expert_num'],
                    const_expert_num=arg1['const_expert_num']
                )
            elif op1 == 'moe_combine_add_rms_norm':
                output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(x=arg1['x'],
                                                                               expert_ids=arg1['expert_ids'],
                                                                               group_ep=arg1['group_ep'],
                                                                               group_tp=arg1['group_tp'],
                                                                               ep_world_size=arg1['ep_world_size'],
                                                                               tp_world_size=arg1['tp_world_size'],
                                                                               ep_rank_id=arg1['ep_rank_id'],
                                                                               tp_rank_id=arg1['tp_rank_id'],
                                                                               expert_shard_type=arg1['expert_shard_type'],
                                                                               shared_expert_rank_num=arg1['shared_expert_rank_num'],
                                                                               moe_expert_num=arg1['moe_expert_num'],
                                                                               scales=arg1['scales'],
                                                                               quant_mode=arg1['quant_mode'],
                                                                               global_bs=arg1['global_bs'],
                                                                               elastic_info=arg1['elastic_info'],
                                                                               zero_expert_num=arg1['zero_expert_num'],
                                                                               copy_expert_num=arg1['copy_expert_num'],
                                                                               const_expert_num=arg1['const_expert_num'])

                expand_x_npu, _, expand_idx_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
                input_dtype = torch.bfloat16
                if expand_x_npu.dtype == torch.int8:
                    expand_x_npu = expand_x_npu.to(input_dtype)
                output1 = torch_npu.npu_moe_distribute_combine_add_rms_norm(expand_x=expand_x_npu,
                                                                                   expert_ids=arg1['expert_ids'],
                                                                                   expand_idx=expand_idx_npu,
                                                                                   ep_send_counts=ep_recv_counts_npu,
                                                                                   tp_send_counts=tp_recv_counts_npu,
                                                                                   expert_scales=arg1['expert_scales'],
                                                                                   group_ep=arg1['group_ep'],
                                                                                   group_tp=arg1['group_tp'],
                                                                                   ep_world_size=arg1['ep_world_size'],
                                                                                   tp_world_size=arg1['tp_world_size'],
                                                                                   ep_rank_id=arg1['ep_rank_id'],
                                                                                   tp_rank_id=arg1['tp_rank_id'],
                                                                                   expert_shard_type=arg1['expert_shard_type'],
                                                                                   shared_expert_rank_num=arg1['shared_expert_rank_num'],
                                                                                   moe_expert_num=arg1['moe_expert_num'],
                                                                                   global_bs=arg1['global_bs'],
                                                                                   shared_expert_num=0,
                                                                                   residual_x=arg1['residual_x'],
                                                                                   gamma=arg1['gamma'],
                                                                                   norm_eps=arg1['norm_eps'],
                                                                                   elastic_info=arg1['elastic_info'],
                                                                                   ori_x=arg1['x'],
                                                                                   const_expert_alpha_1=arg1['const_expert_alpha_1'],
                                                                                   const_expert_alpha_2=arg1['const_expert_alpha_2'],
                                                                                   const_expert_v=arg1['const_expert_v'],
                                                                                   zero_expert_num=arg1['zero_expert_num'],
                                                                                   copy_expert_num=arg1['copy_expert_num'],
                                                                                   const_expert_num=arg1['const_expert_num'])
            elif op1 == 'distribute_barrier':
                output1 = torch_npu._npu_distribute_barrier(arg1['x'], arg1['hcomm_info'], arg1['world_size'])
            elif op1 == 'moe_update_expert':
                balanced_expert_ids, balanced_active_mask = torch_npu.npu_moe_update_expert(expert_ids=arg1['expert_ids'],
                                                                                            eplb_table=arg1['eplb_table'],
                                                                                            expert_scales=arg1['expert_scales'] if arg1['is_pruning'] else None,
                                                                                            pruning_threshold=arg1['pruning_threshold'],
                                                                                            active_mask=arg1['active_mask'],
                                                                                            local_rank_id=arg1['ep_rank_id'],
                                                                                            world_size=arg1['ep_world_size'],
                                                                                            balance_mode=arg1['balance_mode'])
                output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(
                    x=arg1['x'],
                    expert_ids=balanced_expert_ids,
                    group_ep=arg1['group_ep'],
                    group_tp=arg1['group_tp'],
                    ep_world_size=arg1['ep_world_size'],
                    tp_world_size=arg1['tp_world_size'],
                    ep_rank_id=arg1['ep_rank_id'],
                    tp_rank_id=arg1['tp_rank_id'],
                    expert_shard_type=arg1['expert_shard_type'],
                    shared_expert_rank_num=arg1['shared_expert_rank_num'],
                    moe_expert_num=arg1['moe_expert_num'],
                    scales=arg1['scales'],
                    quant_mode=arg1['quant_mode'],
                    global_bs=arg1['global_bs']
                )
                expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
                input_dtype = torch.bfloat16
                if expand_x_npu.dtype == torch.int8:
                    expand_x_npu = expand_x_npu.to(input_dtype)


                output1 = torch_npu.npu_moe_distribute_combine_v2(
                    expand_x=expand_x_npu,
                    expert_ids=arg1['expert_ids'],
                    assist_info_for_combine=assist_info_for_combine_npu,
                    ep_send_counts=ep_recv_counts_npu,
                    tp_send_counts=tp_recv_counts_npu,
                    expert_scales=arg1['expert_scales'],
                    group_ep=arg1['group_ep'],
                    group_tp=arg1['group_tp'],
                    ep_world_size=arg1['ep_world_size'],
                    tp_world_size=arg1['tp_world_size'],
                    ep_rank_id=arg1['ep_rank_id'],
                    tp_rank_id=arg1['tp_rank_id'],
                    expert_shard_type=arg1['expert_shard_type'],
                    shared_expert_rank_num=arg1['shared_expert_rank_num'],
                    moe_expert_num=arg1['moe_expert_num'],
                    global_bs=arg1['global_bs']
                )
        with tng.scope.npu_stream_switch('2'):
            if op2 == 'matmul':
                output2 = torch.matmul(arg2['input'], arg2['weight'])
            elif op2 == 'bmm_reducescatter_alltoall':
                output2 = npu_bmm_reducescatter_alltoall(arg2['x_npu'],
                                               arg2['weight_npu'],
                                               arg2['ep_group'],
                                               arg2['ep_size'],
                                               arg2['tp_group'],
                                               arg2['tp_size'],
                                               bias=arg2['bias_npu'],
                                               shard_type=arg2['y_shard_type'])
            elif op2 == 'alltoall_allgather_bmm':
                output2 = npu_alltoall_allgather_bmm(arg2['x_npu'],
                                           arg2['weight_npu'],
                                           arg2['ep_group'],
                                           arg2['ep_size'],
                                           arg2['tp_group'],
                                           arg2['tp_size'],
                                           bias=arg2['bias_npu'],
                                           shard_type=arg2['x_shard_type'],
                                           act_type=arg2['act_type'],
                                           need_allgather_out=arg2['out_y2_flag'],
                                           need_activation_feature=arg2['out_y3_flag'])
            elif op2 == 'alltoallv_gmm':
                output2 = torch_npu.npu_alltoallv_gmm(gmm_x=arg2['gmm_x'],
                                            gmm_weight=arg2['gmm_weight'],
                                            hcom=arg2['hcom'],
                                            ep_world_size=arg2['ep_world_size'],
                                            send_counts=list(arg2['send_counts']),
                                            recv_counts=list(arg2['recv_counts']),
                                            send_counts_tensor=None,  # send_counts_tensor,
                                            recv_counts_tensor=None,  # recv_counts_tensor,
                                            mm_x=arg2['mm_x'],
                                            mm_weight=arg2['mm_weight'],
                                            trans_gmm_weight=arg2['trans_gmm_weight'],
                                            trans_mm_weight=arg2['trans_mm_weight'],
                                            permute_out_flag=arg2['permute_out_flag'])
            elif op2 == 'gmm_alltoallv':
                output2 = torch_npu.npu_gmm_alltoallv(gmm_x=arg2['gmm_x'],
                                            gmm_weight=arg2['gmm_weight'],
                                            hcom=arg2['hcom'],
                                            ep_world_size=arg2['ep_world_size'],
                                            send_counts=list(arg2['send_counts']),
                                            recv_counts=list(arg2['recv_counts']),
                                            send_counts_tensor=None,  # send_counts_tensor,
                                            recv_counts_tensor=None,  # recv_counts_tensor,
                                            mm_x=arg2['mm_x'],
                                            mm_weight=arg2['mm_weight'],
                                            trans_gmm_weight=arg2['trans_gmm_weight'],
                                            trans_mm_weight=arg2['trans_mm_weight'])
            elif op2 == 'all_gather_base_mm':
                output2 = torch_npu.npu_all_gather_base_mm(arg2['input'], arg2['weight'], arg2['hcomm_info'], arg2['world_size'])
            elif op2 == 'mm_reduce_scatter_base':
                output2 = torch_npu.npu_mm_reduce_scatter_base(arg2['input'], arg2['weight'], arg2['hcomm_info'], arg2['world_size'])
            elif op2 == 'moe_distribute_combine':
                output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch(x=arg2['x'],
                                                                            expert_ids=arg2['expert_ids'],
                                                                            group_ep=arg2['group_ep'],
                                                                            group_tp=arg2['group_tp'],
                                                                            ep_world_size=arg2['ep_world_size'],
                                                                            tp_world_size=arg2['tp_world_size'],
                                                                            ep_rank_id=arg2['ep_rank_id'],
                                                                            tp_rank_id=arg2['tp_rank_id'],
                                                                            expert_shard_type=arg2['expert_shard_type'],
                                                                            shared_expert_rank_num=arg2['shared_expert_rank_num'],
                                                                            moe_expert_num=arg2['moe_expert_num'],
                                                                            scales=arg2['scales'],
                                                                            quant_mode=arg2['quant_mode'],
                                                                            global_bs=arg2['global_bs'])


                expand_x_npu, _, expand_idx_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
                input_dtype = torch.bfloat16
                if expand_x_npu.dtype == torch.int8:
                    expand_x_npu = expand_x_npu.to(input_dtype)

                output2 = torch_npu.npu_moe_distribute_combine(expand_x=expand_x_npu,
                                                                          expert_ids=arg2['expert_ids'],
                                                                          expand_idx=expand_idx_npu,
                                                                          ep_send_counts=ep_recv_counts_npu,
                                                                          tp_send_counts=tp_recv_counts_npu,
                                                                          expert_scales=arg2['expert_scales'],
                                                                          group_ep=arg2['group_ep'],
                                                                          group_tp=arg2['group_tp'],
                                                                          ep_world_size=arg2['ep_world_size'],
                                                                          tp_world_size=arg2['tp_world_size'],
                                                                          ep_rank_id=arg2['ep_rank_id'],
                                                                          tp_rank_id=arg2['tp_rank_id'],
                                                                          expert_shard_type=arg2['expert_shard_type'],
                                                                          shared_expert_rank_num=arg2['shared_expert_rank_num'],
                                                                          moe_expert_num=arg2['moe_expert_num'],
                                                                          global_bs=arg2['global_bs'])
            elif op2 == 'moe_distribute_combine_v2':
                output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(
                    x=arg2['x'],
                    expert_ids=arg2['expert_ids'],
                    group_ep=arg2['group_ep'],
                    group_tp=arg2['group_tp'],
                    ep_world_size=arg2['ep_world_size'],
                    tp_world_size=arg2['tp_world_size'],
                    ep_rank_id=arg2['ep_rank_id'],
                    tp_rank_id=arg2['tp_rank_id'],
                    expert_shard_type=arg2['expert_shard_type'],
                    shared_expert_rank_num=arg2['shared_expert_rank_num'],
                    moe_expert_num=arg2['moe_expert_num'],
                    scales=arg2['scales'],
                    quant_mode=arg2['quant_mode'],
                    global_bs=arg2['global_bs'],
                    elastic_info=arg2['elastic_info'],
                    zero_expert_num=arg2['zero_expert_num'],
                    copy_expert_num=arg2['copy_expert_num'],
                    const_expert_num=arg2['const_expert_num']
                )
                expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
                input_dtype = torch.bfloat16
                if expand_x_npu.dtype == torch.int8:
                    expand_x_npu = expand_x_npu.to(input_dtype)


                output2 = torch_npu.npu_moe_distribute_combine_v2(
                    expand_x=expand_x_npu,
                    expert_ids=arg2['expert_ids'],
                    assist_info_for_combine=assist_info_for_combine_npu,
                    ep_send_counts=ep_recv_counts_npu,
                    tp_send_counts=tp_recv_counts_npu,
                    expert_scales=arg2['expert_scales'],
                    group_ep=arg2['group_ep'],
                    group_tp=arg2['group_tp'],
                    ep_world_size=arg2['ep_world_size'],
                    tp_world_size=arg2['tp_world_size'],
                    ep_rank_id=arg2['ep_rank_id'],
                    tp_rank_id=arg2['tp_rank_id'],
                    expert_shard_type=arg2['expert_shard_type'],
                    shared_expert_rank_num=arg2['shared_expert_rank_num'],
                    moe_expert_num=arg2['moe_expert_num'],
                    global_bs=arg2['global_bs'],
                    elastic_info=arg2['elastic_info'],
                    ori_x=arg2['x'],
                    const_expert_alpha_1=arg2['const_expert_alpha_1'],
                    const_expert_alpha_2=arg2['const_expert_alpha_2'],
                    const_expert_v=arg2['const_expert_v'],
                    zero_expert_num=arg2['zero_expert_num'],
                    copy_expert_num=arg2['copy_expert_num'],
                    const_expert_num=arg2['const_expert_num']
                )
            elif op2 == 'moe_combine_add_rms_norm':
                output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(x=arg2['x'],
                                                                               expert_ids=arg2['expert_ids'],
                                                                               group_ep=arg2['group_ep'],
                                                                               group_tp=arg2['group_tp'],
                                                                               ep_world_size=arg2['ep_world_size'],
                                                                               tp_world_size=arg2['tp_world_size'],
                                                                               ep_rank_id=arg2['ep_rank_id'],
                                                                               tp_rank_id=arg2['tp_rank_id'],
                                                                               expert_shard_type=arg2['expert_shard_type'],
                                                                               shared_expert_rank_num=arg2['shared_expert_rank_num'],
                                                                               moe_expert_num=arg2['moe_expert_num'],
                                                                               scales=arg2['scales'],
                                                                               quant_mode=arg2['quant_mode'],
                                                                               global_bs=arg2['global_bs'],
                                                                               elastic_info=arg2['elastic_info'],
                                                                               zero_expert_num=arg2['zero_expert_num'],
                                                                               copy_expert_num=arg2['copy_expert_num'],
                                                                               const_expert_num=arg2['const_expert_num'])

                expand_x_npu, _, expand_idx_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
                input_dtype = torch.bfloat16
                if expand_x_npu.dtype == torch.int8:
                    expand_x_npu = expand_x_npu.to(input_dtype)
                output2 = torch_npu.npu_moe_distribute_combine_add_rms_norm(expand_x=expand_x_npu,
                                                                                   expert_ids=arg2['expert_ids'],
                                                                                   expand_idx=expand_idx_npu,
                                                                                   ep_send_counts=ep_recv_counts_npu,
                                                                                   tp_send_counts=tp_recv_counts_npu,
                                                                                   expert_scales=arg2['expert_scales'],
                                                                                   group_ep=arg2['group_ep'],
                                                                                   group_tp=arg2['group_tp'],
                                                                                   ep_world_size=arg2['ep_world_size'],
                                                                                   tp_world_size=arg2['tp_world_size'],
                                                                                   ep_rank_id=arg2['ep_rank_id'],
                                                                                   tp_rank_id=arg2['tp_rank_id'],
                                                                                   expert_shard_type=arg2['expert_shard_type'],
                                                                                   shared_expert_rank_num=arg2['shared_expert_rank_num'],
                                                                                   moe_expert_num=arg2['moe_expert_num'],
                                                                                   global_bs=arg2['global_bs'],
                                                                                   shared_expert_num=0,
                                                                                   residual_x=arg2['residual_x'],
                                                                                   gamma=arg2['gamma'],
                                                                                   norm_eps=arg2['norm_eps'],
                                                                                   elastic_info=arg2['elastic_info'],
                                                                                   ori_x=arg2['x'],
                                                                                   const_expert_alpha_1=arg2['const_expert_alpha_1'],
                                                                                   const_expert_alpha_2=arg2['const_expert_alpha_2'],
                                                                                   const_expert_v=arg2['const_expert_v'],
                                                                                   zero_expert_num=arg2['zero_expert_num'],
                                                                                   copy_expert_num=arg2['copy_expert_num'],
                                                                                   const_expert_num=arg2['const_expert_num'])
            elif op2 == 'distribute_barrier':
                output2 = torch_npu._npu_distribute_barrier(arg2['x'], arg2['hcomm_info'], arg2['world_size'])
            elif op2 == 'moe_update_expert':
                balanced_expert_ids, balanced_active_mask = torch_npu.npu_moe_update_expert(expert_ids=arg2['expert_ids'],
                                                                                            eplb_table=arg2['eplb_table'],
                                                                                            expert_scales=arg2['expert_scales'] if arg2['is_pruning'] else None,
                                                                                            pruning_threshold=arg2['pruning_threshold'],
                                                                                            active_mask=arg2['active_mask'],
                                                                                            local_rank_id=arg2['ep_rank_id'],
                                                                                            world_size=arg2['ep_world_size'],
                                                                                            balance_mode=arg2['balance_mode'])
                output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(x=arg2['x'],
                                                                               expert_ids=balanced_expert_ids,
                                                                               group_ep=arg2['group_ep'],
                                                                               group_tp=arg2['group_tp'],
                                                                               ep_world_size=arg2['ep_world_size'],
                                                                               tp_world_size=arg2['tp_world_size'],
                                                                               ep_rank_id=arg2['ep_rank_id'],
                                                                               tp_rank_id=arg2['tp_rank_id'],
                                                                               expert_shard_type=arg2['expert_shard_type'],
                                                                               shared_expert_rank_num=arg2['shared_expert_rank_num'],
                                                                               moe_expert_num=arg2['moe_expert_num'],
                                                                               scales=arg2['scales'],
                                                                               quant_mode=arg2['quant_mode'],
                                                                               global_bs=arg2['global_bs'])
                expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
                input_dtype = torch.bfloat16
                if expand_x_npu.dtype == torch.int8:
                    expand_x_npu = expand_x_npu.to(input_dtype)
                output2 = torch_npu.npu_moe_distribute_combine_v2(expand_x=expand_x_npu,
                                                                             expert_ids=arg2['expert_ids'],
                                                                             assist_info_for_combine=assist_info_for_combine_npu,
                                                                             ep_send_counts=ep_recv_counts_npu,
                                                                             tp_send_counts=tp_recv_counts_npu,
                                                                             expert_scales=arg2['expert_scales'],
                                                                             group_ep=arg2['group_ep'],
                                                                             group_tp=arg2['group_tp'],
                                                                             ep_world_size=arg2['ep_world_size'],
                                                                             tp_world_size=arg2['tp_world_size'],
                                                                             ep_rank_id=arg2['ep_rank_id'],
                                                                             tp_rank_id=arg2['tp_rank_id'],
                                                                             expert_shard_type=arg2['expert_shard_type'],
                                                                             shared_expert_rank_num=arg2['shared_expert_rank_num'],
                                                                             moe_expert_num=arg2['moe_expert_num'],
                                                                             global_bs=arg2['global_bs'])
        return output1, output2

def get_hcomm(p_rank):
    world_size = args.server_num * args.dev_num
    torch_npu.npu.set_device(p_rank)
    rank = p_rank + args.dev_num * args.server_index
    init_method = 'tcp://' + args.master_ip + ':' + '50001'
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    ep_ranks_list, ep_ranks_list1, tp_ranks_list, tp_ranks_list1 = get_rank_list(world_size)
    ep_group, ep_group1, tp_group, tp_group1 = init_hccl_comm(rank, ep_ranks_list, ep_ranks_list1,
                                                              tp_ranks_list, tp_ranks_list1)
    ep_hcomm_info = get_hcomm_info(ep_group, rank)
    ep_hcomm_info1 = get_hcomm_info(ep_group1, rank)
    tp_hcomm_info = get_hcomm_info(tp_group, rank)
    tp_hcomm_info1 = get_hcomm_info(tp_group1, rank)
    hcomm_info_dict = {"hcomm_info": hcomm_info, "ep_hcomm_info": ep_hcomm_info, "ep_hcomm_info1": ep_hcomm_info1,
                       "tp_hcomm_info": tp_hcomm_info, "tp_hcomm_info1": tp_hcomm_info1}
    return hcomm_info_dict


def get_rank_list(world_size):
    ep_ranks_list = []
    ep_ranks_list1 = []
    tp_ranks_list = []
    tp_ranks_list1 = []
    for i in range(tp_size):
        # [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
        ep_ranks = [x * tp_size + i for x in range(ep_size)]
        ep_ranks_list.append(ep_ranks)

    for i in range(tp_size):
        # [[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15]]
        ep_ranks = [x + ep_size * i for x in range(ep_size)]
        ep_ranks_list1.append(ep_ranks)

    for i in range(ep_size):
        # [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
        tp_ranks = [x + tp_size * i for x in range(tp_size)]
        tp_ranks_list.append(tp_ranks)

    # [[0, 15], [1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9], [7, 8]]
    # [[0, 15, 1, 14], [2, 13, 3, 12], [4, 11, 5, 10], [6, 9, 7, 8]]
    lst = [i for i in range(world_size)]
    while lst:
        if len(lst) >= tp_size:
            part = lst[:tp_size // 2] + lst[-tp_size // 2:]
            tp_ranks_list1.append(part)
            lst = lst[tp_size // 2: -tp_size // 2]  # 移除首尾各两个元素
        else:
            break
    return ep_ranks_list, ep_ranks_list1, tp_ranks_list, tp_ranks_list1


def init_hccl_comm(rank, ep_ranks_list, ep_ranks_list1, tp_ranks_list, tp_ranks_list1):
    # print(f"device {rank} 初始化EP域")
    for ep_ranks in ep_ranks_list:
        tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks)
        if rank in ep_ranks:
            ep_group = tmp_group

    # print(f"device {rank} 初始化EP域1")
    for ep_ranks1 in ep_ranks_list1:
        tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks1)
        if rank in ep_ranks1:
            ep_group1 = tmp_group

    # print(f"device {rank} 初始化TP域")
    for tp_ranks in tp_ranks_list:
        tmp_group = dist.new_group(backend="hccl", ranks=tp_ranks)
        if rank in tp_ranks:
            tp_group = tmp_group

    # print(f"device {rank} 初始化TP域1")
    for tp_ranks1 in tp_ranks_list1:
        tmp_group = dist.new_group(backend="hccl", ranks=tp_ranks1)
        if rank in tp_ranks1:
            tp_group1 = tmp_group

    return ep_group, ep_group1, tp_group, tp_group1


def get_hcomm_info(comm_group, rank):
    if torch.__version__ > '2.0.1':
        hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = comm_group.get_hccl_comm_name(rank)
    return hcomm_info


class MC2_OP:
    def __init__(self, args, rank, hcomm_info_dict):
        self.master_ip = args.master_ip
        self.dev_num = int(args.dev_num)
        self.server_num = int(args.server_num)
        self.server_index = int(args.server_index)
        self.stream_num = int(args.stream_num)
        self.loop = int(args.loop)
        self.world_size = self.server_num * self.dev_num
        self.p_rank = rank
        self.graph = args.graph
        self.dynamic = args.dynamic
        self.rank = rank + self.dev_num * self.server_index
        self.tp_size = tp_size
        self.ep_size = self.world_size // tp_size
        self.hcomm_info = hcomm_info_dict["hcomm_info"]
        self.ep_hcomm_info = hcomm_info_dict["ep_hcomm_info"]
        self.ep_hcomm_info1 = hcomm_info_dict["ep_hcomm_info1"]
        self.tp_hcomm_info = hcomm_info_dict["tp_hcomm_info"]
        self.tp_hcomm_info1 = hcomm_info_dict["tp_hcomm_info1"]

    @staticmethod
    def define_model(model, dynamic):
        npu_backend = torchair.get_npu_backend(compiler_config=None)
        model = torch.compile(model, backend=npu_backend, dynamic=dynamic)
        return model

    @staticmethod
    def get_profiling(path):
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            record_op_args=False, data_simplification=False
        )
        p = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
            ],
            schedule=torch_npu.profiler.schedule(
                wait=0,
                warmup=0,
                active=50,
                repeat=1,
                skip_first=0
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
        )
        return p


class AlltoallvGmm(MC2_OP):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)
        # BS = 512
        # K = 8
        self.gmm_x_shape = [4096, 8192]
        self.gmm_weight_shape = [self.dev_num, 8192, 8192]
        self.send_counts = [4096 // (self.dev_num * self.world_size)] * (self.dev_num * self.world_size)
        self.recv_counts = [4096 // (self.dev_num * self.world_size)] * (self.dev_num * self.world_size)
        self.gmm_x = torch.randn(self.gmm_x_shape, dtype=torch.float16).npu()
        self.gmm_weight = torch.randn(self.gmm_weight_shape, dtype=torch.float16).npu()

    def run_npu(self):
        print(torch_npu.npu_alltoallv_gmm(gmm_x=self.gmm_x,
                                          gmm_weight=self.gmm_weight,
                                          hcom=self.ep_hcomm_info,
                                          ep_world_size=self.world_size,
                                          send_counts=list(self.send_counts),
                                          recv_counts=list(self.recv_counts),
                                          send_counts_tensor=None,  # send_counts_tensor,
                                          recv_counts_tensor=None,  # recv_counts_tensor,
                                          mm_x=None,
                                          mm_weight=None,
                                          trans_gmm_weight=False,
                                          trans_mm_weight=False,
                                          permute_out_flag=False))


class GmmAlltoallv(AlltoallvGmm):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)

    def run_npu(self):
        print(torch_npu.npu_gmm_alltoallv(gmm_x=self.gmm_x,
                                          gmm_weight=self.gmm_weight,
                                          hcom=self.ep_hcomm_info1,
                                          ep_world_size=self.world_size,
                                          send_counts=list(self.send_counts),
                                          recv_counts=list(self.recv_counts),
                                          send_counts_tensor=None,  # send_counts_tensor,
                                          recv_counts_tensor=None,  # recv_counts_tensor,
                                          mm_x=None,
                                          mm_weight=None,
                                          trans_gmm_weight=False,
                                          trans_mm_weight=False))


class AllGatherCustom(MC2_OP):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)
        self.x1_shape = [4096, 8192]
        self.x1 = torch.randn(self.x1_shape, dtype=torch.float16).npu()
        self.tileM = 128

    def run_npu(self):
        model = self.AllGatherCustomModel()
        if self.graph:
            model = self.define_model(model, self.dynamic)
        out = model(x1=self.x1, hcom=self.hcomm_info, world_size=self.world_size, tileM=self.tileM,
                   strideCount=0, comm_turn=0, version=3, x1_dtype=None)
        print(out,'-----------------------------===================================')
        # else:
        #     out = torch_npu.npu_all_gather_custom(
        #         x1=self.x1,
        #         hcom=self.hcomm_info,
        #         world_size=self.world_size,
        #         tileM=self.tileM,
        #         strideCount=0,
        #         comm_turn=0,
        #         version=3,
        #         x1_dtype=None
        #     )
        #     print(out)

    class AllGatherCustomModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # super(AllGatherCustom.AllGatherCustomModel, self).__init__()

        def forward(self, x1, hcom, world_size, tileM, strideCount, comm_turn=0, version=3, x1_dtype=None):
            output = torch_npu.npu_all_gather_custom(
                x1=x1,
                hcom=hcom,
                world_size=world_size,
                tileM=tileM,
                strideCount=strideCount,
                comm_turn=comm_turn,
                version=version,
                x1_dtype=x1_dtype
            )
            return output


class AllReduceCustom(AllGatherCustom):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)
        self.reduceOp = 3

    def run_npu(self):
        output = torch_npu.npu_all_reduce_custom(
            x1=self.x1,
            hcom=self.hcomm_info,
            reduceOp=self.reduceOp,
            world_size=self.world_size,
            tileM=self.tileM,
            comm_turn=0,
            version=3
        )
        print(output)


class AllToAllVAllGatherCustom(AllGatherCustom):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)

    def run_npu(self):
        output, gather_output = torch_npu.npu_all_to_allv_all_gather_custom(
            x1=self.x1,
            ep_hcom=self.ep_hcomm_info,
            tp_hcom=self.tp_hcomm_info,
            ep_world_size=self.ep_size,
            tp_world_size=self.tp_size,
            is_gather_output=True,
            version=3
        )
        print(output)


class AllGatherAllToAllReduceScatterAllToAllCustom(AllGatherCustom):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)
        self.x1_shape = [100, 1024]
        self.x1 = torch.randn(self.x1_shape, dtype=torch.float16).npu()
        self.tileM = 40

    def run_npu(self):
        output = torch_npu.npu_allgather_alltoall_allreducescatter_alltoall_custom(self.x1, self.ep_hcomm_info,
                                                                                   self.tp_hcomm_info1,
                                                                                   self.ep_size,
                                                                                   self.tp_size,
                                                                                   self.tileM)
        print(output)


class ReduceScatterCustom(AllReduceCustom):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)

    def run_npu(self):
        output = torch_npu.npu_reduce_scatter_custom(
            x1=self.x1,
            hcom=self.hcomm_info,
            reduceOp=self.reduceOp,
            world_size=self.world_size,
            tileM=self.tileM,
            strideCount=0,
            comm_turn=0,
            version=3
        )
        print(output)


class AllToAllCustom(AllGatherCustom):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)

    def run_npu(self):
        output = torch_npu.npu_all_to_all_custom(
            x1=self.x1,
            hcom=self.hcomm_info,
            world_size=self.world_size,
            tile_m=self.tileM,
            version=3
        )
        print(output)


class AllGatherMatMul(AllGatherCustom):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)
        self.x1_shape = [4096, 8192]
        self.x2_shape = [8192, 8192]
        self.x1 = torch.randn(self.x1_shape, dtype=torch.float16).npu()
        self.x2 = torch.randn(self.x2_shape, dtype=torch.float16).npu()

    def run_npu(self):
        if self.graph:
            pass
        else:
            output, gather_out = torch_npu.npu_all_gather_base_mm(self.x1, self.x2, self.hcomm_info, self.world_size)
            print(output)


class MatMulReduceScatter(AllGatherMatMul):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)

    def run_npu(self):
        if self.graph:
            pass
        else:
            output = torch_npu.npu_mm_reduce_scatter_base(self.x1, self.x2, self.hcomm_info, self.world_size)
            print(output)


class DistributeBarrier(AllGatherMatMul):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)

    def run_npu(self):
        out = torch_npu._npu_distribute_barrier(self.x1, self.hcomm_info, self.world_size)
        print(out)


class AlltoAllAllGatherBatchMatMul(AllGatherMatMul):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)
        self.x_shard_type = 1
        self.out_y2_flag = True
        self.out_y3_flag = False
        self.transpose_weight = False

        self.E, self.C, self.H, self.M = 16, 1024, 1024, 8192
        if self.x_shard_type == 0:
            self.x_shape = (self.E, self.C, self.H / self.tp_size)
        elif self.x_shard_type == 1:
            self.x_shape = (self.E, self.C / self.tp_size, self.H)
        else:
            self.x_shape = (self.E / self.ep_size, self.tp_size * self.ep_size * self.C, self.M / self.tp_size)
        self.weight_shape = (self.E / self.ep_size, self.H, self.M / self.tp_size)
        if self.transpose_weight == True:
            self.weight_shape = (self.E / self.ep_size, self.M / self.tp_size, self.H)
        self.bias_shape = (self.E / self.ep_size, 1, self.M / self.tp_size)

        self.x_shape = tuple(int(item) for item in self.x_shape)
        self.weight_shape = tuple(int(item) for item in self.weight_shape)
        self.bias_shape = tuple(int(item) for item in self.bias_shape)
        self.x_npu = torch.randn(self.x_shape, dtype=torch.float16).npu()
        self.weight_npu = torch.randn(self.weight_shape, dtype=torch.float16).npu()
        self.bias_npu = torch.randn(self.bias_shape, dtype=torch.float16).npu()
        if self.transpose_weight == True:
            self.weight_npu = self.weight_npu.transpose(1, 2)

    def run_npu(self):
        y_npu = npu_alltoall_allgather_bmm(self.x_npu,
                                           self.weight_npu,
                                           self.ep_hcomm_info,
                                           self.ep_size,
                                           self.tp_hcomm_info,
                                           self.tp_size,
                                           bias=self.bias_npu,
                                           shard_type=self.x_shard_type,
                                           act_type="None",
                                           need_allgather_out=self.out_y2_flag,
                                           need_activation_feature=self.out_y3_flag)
        print(y_npu)


class BatchMatMulReduceScatterAlltoAll(AlltoAllAllGatherBatchMatMul):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)
        self.y_shard_type = 1
        if self.y_shard_type == 0:
            self.x_shape = (self.E / self.ep_size, self.ep_size * self.C, self.M / self.tp_size)
            self.bias_shape = (self.E / self.ep_size, 1, self.H / self.tp_size)
        else:
            self.x_shape = (self.E / self.ep_size, self.tp_size * self.ep_size * self.C, self.M / self.tp_size)
            self.bias_shape = (self.E / self.ep_size, 1, self.H)
        self.weight_shape = (self.E / self.ep_size, self.M / self.tp_size, self.H)
        if self.transpose_weight == True:
            self.weight_shape = (self.E / self.ep_size, self.H, self.M / self.tp_size)

        self.x_shape = tuple(int(item) for item in self.x_shape)
        self.weight_shape = tuple(int(item) for item in self.weight_shape)
        self.bias_shape = tuple(int(item) for item in self.bias_shape)
        self.x_npu = torch.randn(self.x_shape, dtype=torch.float16).npu()
        self.weight_npu = torch.randn(self.weight_shape, dtype=torch.float16).npu()
        self.bias_npu = torch.randn(self.bias_shape, dtype=torch.float16).npu()
        if self.transpose_weight == True:
            self.weight_npu = self.weight_npu.transpose(1, 2)

    def run_npu(self):
        y = npu_bmm_reducescatter_alltoall(self.x_npu,
                                           self.weight_npu,
                                           self.ep_hcomm_info1,
                                           self.ep_size,
                                           self.tp_hcomm_info1,
                                           self.tp_size,
                                           bias=self.bias_npu,
                                           shard_type=self.y_shard_type)
        print(y)


class MatMul(MC2_OP):
    def __init__(self, args, rank, hcomm_info_dict):
        # super().__init__(args, rank, hcomm_info_dict)
        pass

class MoeDistributeCombine(MC2_OP):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)

        # 控制模式
        quant_mode = 2  # 2为动态量化
        is_dispatch_scales = True  # 动态量化可选择是否传scales
        input_dtype = torch.bfloat16  # 输出dtype
        server_num = int(args.server_num)
        server_index = 0
        port = 50001
        master_ip = '127.0.0.1'
        dev_num = int(args.dev_num)
        world_size = server_num * dev_num
        rank_per_dev = int(world_size / server_num)  # 每个host有几个die
        sharedExpertRankNum = 2  # 共享专家数
        moeExpertNum = 6  # moe专家数
        bs = 8  # token数量
        h = 7168  # 每个token的长度
        k = 6
        random_seed = 0
        tp_world_size = tp_size
        ep_world_size = int(world_size / tp_world_size)
        moe_rank_num = ep_world_size - sharedExpertRankNum
        local_moe_expert_num = moeExpertNum // moe_rank_num
        globalBS = bs * ep_world_size
        is_shared = (sharedExpertRankNum > 0)
        is_quant = (quant_mode > 0)
        def gen_unique_topk_array(low, high, bs, k):
            array = []
            for i in range(bs):
                top_idx = list(np.arange(low, high, dtype=np.int32))
                random.shuffle(top_idx)
                array.append(top_idx[0:k])
            return np.array(array)

        self.quant_mode = quant_mode
        self.sharedExpertRankNum = sharedExpertRankNum
        self.moeExpertNum = moeExpertNum
        self.tp_world_size = tp_world_size
        self.ep_world_size = ep_world_size
        self.globalBS = globalBS

        self.x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = gen_unique_topk_array(0, moeExpertNum, bs, k).astype(np.int32)
        self.expert_ids = torch.from_numpy(expert_ids).npu()
        self.expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + moeExpertNum, h) if sharedExpertRankNum else (moeExpertNum, h)
        if is_dispatch_scales:
            self.scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            self.scales = None


class MoeDistributeCombineV2(MC2_OP):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)
        # 控制模式
        quant_mode = 2  # 2为动态量化
        is_dispatch_scales = True  # 动态量化可选择是否传scales
        input_dtype = torch.bfloat16  # 输出dtype
        server_num = int(args.server_num)
        server_index = 0
        port = 50001
        master_ip = '127.0.0.1'
        dev_num = int(args.dev_num)
        world_size = server_num * dev_num
        rank_per_dev = int(world_size / server_num)  # 每个host有几个die
        shared_expert_rank_num = 0  # 共享专家数
        moe_expert_num = 8  # moe专家数
        bs = 8  # token数量
        h = 7168  # 每个token的长度
        k = 8
        random_seed = 0
        tp_world_size = tp_size
        ep_world_size = int(world_size / tp_world_size)
        moe_rank_num = ep_world_size - shared_expert_rank_num
        local_moe_expert_num = moe_expert_num // moe_rank_num
        globalBS = bs * ep_world_size
        is_shared = (shared_expert_rank_num > 0)
        is_quant = (quant_mode > 0)

        zero_expert_num = 1
        copy_expert_num = 1
        const_expert_num = 1

        def gen_const_expert_alpha_1():
            const_expert_alpha_1 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
            return const_expert_alpha_1

        def gen_const_expert_alpha_2():
            const_expert_alpha_2 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
            return const_expert_alpha_2

        def gen_const_expert_v():
            const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
            return const_expert_v

        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = torch.tensor([
            [0, 8, 4, 1, 6, 12, 14, 17],
            [14, 10, 7, 3, 0, 12, 11, 17],
            [12, 0, 5, 11, 19, 4, 6, 18],
            [17, 3, 4, 10, 18, 0, 1, 2],
            [13, 16, 9, 10, 15, 6, 7, 14],
            [17, 15, 14, 8, 16, 18, 3, 12],
            [4, 12, 2, 17, 15, 3, 9, 10],
            [16, 7, 12, 9, 18, 3, 19, 17]
        ], dtype=torch.int32).npu()

        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + moe_expert_num, h) if shared_expert_rank_num else (moe_expert_num, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        elastic_info = None
        const_expert_alpha_1 = gen_const_expert_alpha_1().npu()
        const_expert_alpha_2 = gen_const_expert_alpha_2().npu()
        const_expert_v = gen_const_expert_v().npu()
        self.x = x
        self.expert_ids = expert_ids
        self.ep_world_size = ep_world_size
        self.tp_world_size = tp_world_size
        self.shared_expert_rank_num = shared_expert_rank_num
        self.moe_expert_num = moe_expert_num
        self.scales = scales
        self.quant_mode = quant_mode
        self.globalBS = globalBS
        self.expert_scales = expert_scales
        self.elastic_info = elastic_info
        self.const_expert_alpha_1 = const_expert_alpha_1
        self.const_expert_alpha_2 = const_expert_alpha_2
        self.const_expert_v = const_expert_v
        self.zero_expert_num = zero_expert_num
        self.copy_expert_num = copy_expert_num
        self.const_expert_num = const_expert_num

        self.bs = bs  # token数量
        self.h = h  # 每个token的长度
        self.k = k
        self.random_seed = random_seed
        self.input_dtype = input_dtype

        # if rank == 0:
        #     # 打印类成员变量（PyTorch专属，名称+值/张量shape）
        #     print(f"x: {self.x.shape if isinstance(self.x, torch.Tensor) else self.x}")
        #     print(f"expert_ids: {self.expert_ids.shape if isinstance(self.expert_ids, torch.Tensor) else self.expert_ids}")
        #     print(
        #         f"ep_world_size: {self.ep_world_size.shape if isinstance(self.ep_world_size, torch.Tensor) else self.ep_world_size}")
        #     print(
        #         f"tp_world_size: {self.tp_world_size.shape if isinstance(self.tp_world_size, torch.Tensor) else self.tp_world_size}")
        #     print(
        #         f"shared_expert_rank_num: {self.shared_expert_rank_num.shape if isinstance(self.shared_expert_rank_num, torch.Tensor) else self.shared_expert_rank_num}")
        #     print(
        #         f"moe_expert_num: {self.moe_expert_num.shape if isinstance(self.moe_expert_num, torch.Tensor) else self.moe_expert_num}")
        #     print(f"scales: {self.scales.shape if isinstance(self.scales, torch.Tensor) else self.scales}")
        #     print(f"quant_mode: {self.quant_mode.shape if isinstance(self.quant_mode, torch.Tensor) else self.quant_mode}")
        #     print(f"globalBS: {self.globalBS.shape if isinstance(self.globalBS, torch.Tensor) else self.globalBS}")
        #     print(
        #         f"expert_scales: {self.expert_scales.shape if isinstance(self.expert_scales, torch.Tensor) else self.expert_scales}")
        #     print(
        #         f"elastic_info: {self.elastic_info.shape if isinstance(self.elastic_info, torch.Tensor) else self.elastic_info}")
        #     print(
        #         f"const_expert_alpha_1: {self.const_expert_alpha_1.shape if isinstance(self.const_expert_alpha_1, torch.Tensor) else self.const_expert_alpha_1}")
        #     print(
        #         f"const_expert_alpha_2: {self.const_expert_alpha_2.shape if isinstance(self.const_expert_alpha_2, torch.Tensor) else self.const_expert_alpha_2}")
        #     print(
        #         f"const_expert_v: {self.const_expert_v.shape if isinstance(self.const_expert_v, torch.Tensor) else self.const_expert_v}")
        #     print(
        #         f"zero_expert_num: {self.zero_expert_num.shape if isinstance(self.zero_expert_num, torch.Tensor) else self.zero_expert_num}")
        #     print(
        #         f"copy_expert_num: {self.copy_expert_num.shape if isinstance(self.copy_expert_num, torch.Tensor) else self.copy_expert_num}")
        #     print(
        #         f"const_expert_num: {self.const_expert_num.shape if isinstance(self.const_expert_num, torch.Tensor) else self.const_expert_num}")
        #     print(f"bs: {self.bs.shape if isinstance(self.bs, torch.Tensor) else self.bs}")  # token数量
        #     print(f"h: {self.h.shape if isinstance(self.h, torch.Tensor) else self.h}")  # 每个token的长度
        #     print(f"k: {self.k.shape if isinstance(self.k, torch.Tensor) else self.k}")
        #     print(
        #         f"random_seed: {self.random_seed.shape if isinstance(self.random_seed, torch.Tensor) else self.random_seed}")
        #     print(
        #         f"input_dtype: {self.input_dtype.shape if isinstance(self.input_dtype, torch.Tensor) else self.input_dtype}")

class MoeUpdateExpert(MoeDistributeCombineV2):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)
        def gen_unique_topk_array(low, high, bs, k):
            array = []
            for i in range(bs):
                top_idx = list(np.arange(low, high, dtype=np.int32))
                random.shuffle(top_idx)
                array.append(top_idx[0:k])
            return np.array(array)

        expert_ids = gen_unique_topk_array(0, self.moe_expert_num, self.bs, self.k).astype(np.int32)
        expert_ids = torch.from_numpy(expert_ids).npu()
        eplb_table = np.array(
            [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12],
             [1, 13]]).astype(np.int32)
        eplb_table = torch.from_numpy(eplb_table).npu()
        expert_scales = torch.randn(self.bs, self.k, dtype=torch.float32).npu()

        pruning_threshold = None
        active_mask = None
        is_pruning = 1
        if is_pruning:
            expert_scales_temp = -np.sort(-np.random.uniform(low=0, high=0.25, size=(self.bs, self.k)), axis=1)
            pruning_threshold_temp = np.random.uniform(low=0, high=0.15, size=(1, self.k))
            num_true = np.random.randint(0, self.bs + 1)
            active_mask_temp = np.concatenate([np.ones(num_true, dtype=bool), np.zeros(self.bs - num_true, dtype=bool)])
            expert_scales_tensor = torch.from_numpy(expert_scales_temp).to(torch.float32)
            pruning_threshold_tensor = torch.from_numpy(pruning_threshold_temp).to(torch.float32)
            active_mask_tensor = torch.from_numpy(active_mask_temp).to(torch.bool)
            expert_scales = expert_scales_tensor.npu()
            pruning_threshold = pruning_threshold_tensor.npu()
            active_mask = active_mask_tensor.npu()

        self.expert_ids = expert_ids
        self.eplb_table = eplb_table
        self.pruning_threshold = pruning_threshold
        self.active_mask = active_mask
        self.balance_mode = 1
        self.is_pruning = is_pruning
        self.expert_scales = expert_scales


class MoeDistributeCombineAddRmsNorm(MoeDistributeCombineV2):
    def __init__(self, args, rank, hcomm_info_dict):
        super().__init__(args, rank, hcomm_info_dict)
        bs_local = self.expert_ids.shape[0]
        torch.manual_seed(42)
        self.residual_x = torch.rand((bs_local, 1, self.h), dtype=torch.bfloat16).npu()
        torch.manual_seed(self.random_seed)
        self.gamma = torch.ones(self.h).to(self.input_dtype).npu()
        self.norm_eps = 1e-6

def get_op_arg(rank, op_type, run_op):
    arg = {}
    if op_type == 'matmul':
        e, c, h, m, k, n = [32, 1024, 1024, 1024, 16384, 16384]
        device = torch.device("npu")
        np_input = np.random.uniform(-1, 1, [m, k]).astype(np.float16)
        np_weight = np.random.uniform(-1, 1, [k, n]).astype(np.float16)
        arg['input'], arg['weight']= torch.from_numpy(np_input).to(device), torch.from_numpy(np_weight).to(device)
    elif op_type == 'bmm_reducescatter_alltoall':
        arg['x_npu'] = run_op.x_npu
        arg['weight_npu'] = run_op.weight_npu
        arg['ep_group'] = run_op.ep_hcomm_info1
        arg['ep_size'] = run_op.ep_size
        arg['tp_group'] = run_op.tp_hcomm_info1
        arg['tp_size'] = run_op.tp_size
        arg['bias_npu'] = run_op.bias_npu
        arg['y_shard_type'] = run_op.y_shard_type
    elif op_type == 'alltoall_allgather_bmm':
        arg['x_npu'] = run_op.x_npu
        arg['weight_npu'] = run_op.weight_npu
        arg['ep_group'] = run_op.ep_hcomm_info
        arg['ep_size'] = run_op.ep_size
        arg['tp_group'] = run_op.tp_hcomm_info
        arg['tp_size'] = run_op.tp_size
        arg['bias_npu'] = run_op.bias_npu
        arg['x_shard_type'] = run_op.x_shard_type
        arg['act_type'] = "None"
        arg['out_y2_flag'] = run_op.out_y2_flag
        arg['out_y3_flag'] = run_op.out_y3_flag
    elif op_type == 'alltoallv_gmm':
        arg['gmm_x'] = run_op.gmm_x
        arg['gmm_weight'] = run_op.gmm_weight
        arg['hcom'] = run_op.ep_hcomm_info
        arg['ep_world_size'] = run_op.world_size
        arg['send_counts'] = run_op.send_counts
        arg['recv_counts'] = run_op.recv_counts
        arg['mm_x'] = None
        arg['mm_weight'] = None
        arg['trans_gmm_weight'] = False
        arg['trans_mm_weight'] = False
        arg['permute_out_flag'] = False
    elif op_type == 'gmm_alltoallv':
        arg['gmm_x'] = run_op.gmm_x
        arg['gmm_weight'] = run_op.gmm_weight
        arg['hcom'] = run_op.ep_hcomm_info1
        arg['ep_world_size'] = run_op.world_size
        arg['send_counts'] = run_op.send_counts
        arg['recv_counts'] = run_op.recv_counts
        arg['mm_x'] = None
        arg['mm_weight'] = None
        arg['trans_gmm_weight'] = False
        arg['trans_mm_weight'] = False
    elif op_type == 'all_gather_base_mm':
        arg['input'], arg['weight'], arg['hcomm_info'], arg[
            'world_size'] = run_op.x1, run_op.x2, run_op.hcomm_info, run_op.world_size
    elif op_type == 'mm_reduce_scatter_base':
        arg['input'], arg['weight'], arg['hcomm_info'], arg[
            'world_size'] = run_op.x1, run_op.x2, run_op.hcomm_info, run_op.world_size
    elif op_type == 'moe_distribute_combine':
        arg['x'] = run_op.x
        arg['expert_ids'] = run_op.expert_ids
        arg['group_ep'] = run_op.ep_hcomm_info
        arg['group_tp'] = run_op.tp_hcomm_info
        arg['ep_world_size'] = run_op.ep_world_size
        arg['tp_world_size'] = run_op.tp_world_size
        arg['ep_rank_id'] = rank // run_op.tp_world_size
        arg['tp_rank_id'] = rank % run_op.tp_world_size
        arg['expert_shard_type'] = 0
        arg['shared_expert_rank_num'] = run_op.sharedExpertRankNum
        arg['moe_expert_num'] = run_op.moeExpertNum
        arg['scales'] = run_op.scales
        arg['quant_mode'] = run_op.quant_mode
        arg['global_bs'] = run_op.globalBS
        arg['expert_scales'] = run_op.expert_scales
    elif op_type == 'moe_distribute_combine_v2':
        arg['x'] = run_op.x
        arg['expert_ids'] = run_op.expert_ids
        arg['group_ep'] = run_op.ep_hcomm_info
        arg['group_tp'] = run_op.tp_hcomm_info
        arg['ep_world_size'] = run_op.ep_world_size
        arg['tp_world_size'] = run_op.tp_world_size
        arg['ep_rank_id'] = rank // run_op.tp_world_size
        arg['tp_rank_id'] = rank % run_op.tp_world_size
        arg['expert_shard_type'] = 0
        arg['shared_expert_rank_num'] = run_op.shared_expert_rank_num
        arg['moe_expert_num'] = run_op.moe_expert_num
        arg['scales'] = run_op.scales
        arg['quant_mode'] = run_op.quant_mode
        arg['global_bs'] = run_op.globalBS
        arg['expert_scales'] = run_op.expert_scales
        arg['elastic_info'] = run_op.elastic_info
        arg['const_expert_alpha_1'] = run_op.const_expert_alpha_1
        arg['const_expert_alpha_2'] = run_op.const_expert_alpha_2
        arg['const_expert_v'] = run_op.const_expert_v
        arg['zero_expert_num'] = run_op.zero_expert_num
        arg['copy_expert_num'] = run_op.copy_expert_num
        arg['const_expert_num'] = run_op.const_expert_num
    elif op_type == 'moe_combine_add_rms_norm':
        arg['x'] = run_op.x
        arg['expert_ids'] = run_op.expert_ids
        arg['group_ep'] = run_op.ep_hcomm_info
        arg['group_tp'] = run_op.tp_hcomm_info
        arg['ep_world_size'] = run_op.ep_world_size
        arg['tp_world_size'] = run_op.tp_world_size
        arg['ep_rank_id'] = rank // run_op.tp_world_size
        arg['tp_rank_id'] = rank % run_op.tp_world_size
        arg['expert_shard_type'] = 0
        arg['shared_expert_rank_num'] = run_op.shared_expert_rank_num
        arg['moe_expert_num'] = run_op.moe_expert_num
        arg['scales'] = run_op.scales
        arg['quant_mode'] = run_op.quant_mode
        arg['global_bs'] = run_op.globalBS
        arg['expert_scales'] = run_op.expert_scales
        arg['residual_x'] = run_op.residual_x
        arg['gamma'] = run_op.gamma
        arg['norm_eps'] = run_op.norm_eps
        arg['elastic_info'] = run_op.elastic_info
        arg['const_expert_alpha_1'] = run_op.const_expert_alpha_1
        arg['const_expert_alpha_2'] = run_op.const_expert_alpha_2
        arg['const_expert_v'] = run_op.const_expert_v
        arg['zero_expert_num'] = run_op.zero_expert_num
        arg['copy_expert_num'] = run_op.copy_expert_num
        arg['const_expert_num'] = run_op.const_expert_num
    elif op_type == 'distribute_barrier':
        arg['x'], arg['hcomm_info'], arg['world_size'] = run_op.x1, run_op.hcomm_info, run_op.world_size
    elif op_type == 'moe_update_expert':
        arg['x'] = run_op.x
        arg['expert_ids'] = run_op.expert_ids
        arg['group_ep'] = run_op.ep_hcomm_info
        arg['group_tp'] = run_op.tp_hcomm_info
        arg['ep_world_size'] = run_op.ep_world_size
        arg['tp_world_size'] = run_op.tp_world_size
        arg['ep_rank_id'] = rank // run_op.tp_world_size
        arg['tp_rank_id'] = rank % run_op.tp_world_size
        arg['expert_shard_type'] = 0
        arg['shared_expert_rank_num'] = run_op.shared_expert_rank_num
        arg['moe_expert_num'] = run_op.moe_expert_num
        arg['scales'] = run_op.scales
        arg['quant_mode'] = run_op.quant_mode
        arg['global_bs'] = run_op.globalBS
        arg['expert_scales'] = run_op.expert_scales
        arg['elastic_info'] = run_op.elastic_info
        arg['const_expert_alpha_1'] = run_op.const_expert_alpha_1
        arg['const_expert_alpha_2'] = run_op.const_expert_alpha_2
        arg['const_expert_v'] = run_op.const_expert_v
        arg['zero_expert_num'] = run_op.zero_expert_num
        arg['copy_expert_num'] = run_op.copy_expert_num
        arg['const_expert_num'] = run_op.const_expert_num

        arg['eplb_table'] = run_op.eplb_table
        arg['pruning_threshold'] = run_op.pruning_threshold
        arg['active_mask'] = run_op.active_mask
        arg['balance_mode'] = run_op.balance_mode
        arg['is_pruning'] = run_op.is_pruning
    return arg

def run(args, rank, run_op_1, run_op_2):
    hcomm_info_dict = get_hcomm(rank)

    run_op_1 = run_op_1(args, rank, hcomm_info_dict)
    run_op_2 = run_op_2(args, rank, hcomm_info_dict)
    arg1, arg2 = get_op_arg(rank, args.op1, run_op_1), get_op_arg(rank, args.op2, run_op_2)

    model = Model()
    npu_backend = tng.get_npu_backend(compiler_config=None)
    model = torch.compile(model, backend=npu_backend, dynamic=args.dynamic)
    if rank == 0:
        # for key, value in arg1.items():
        #     print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")
        print('dynamic: ', args.dynamic)
    output1, output2 = model(args.op1, args.op2, arg1, arg2)


    if isinstance(output1, (torch.Tensor, tuple)) and isinstance(output2, (torch.Tensor, tuple)):
        output1 = output1 if isinstance(output1, torch.Tensor) else output1[0]
        output2 = output2 if isinstance(output2, torch.Tensor) else output2[0]
        print(f'\nrank {rank} test result: PASS')
    else:
        print(f'\nrank {rank} test result: FAIL')
    if rank == 0:
        print(f'output1: {output1}')
        print(f'output2: {output2}')


    # run_op_1.run_npu()
    # run_op_2.run_npu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_num", type=int, default=16, help="dev_num数量")
    parser.add_argument("--server_num", type=int, default=1, help="server数量")
    parser.add_argument("--stream_num", type=int, default=1, help="stream数量")
    parser.add_argument("--server_index", type=int, default=0, help="server数量")
    parser.add_argument("--type", type=int, default=3, help="组合场景，1、双流纯通信组合，"
                                                            "2、双流纯mc2算子组合，3、双流双通信域组合, 4、所有组合随机")
    parser.add_argument("--loop", type=int, default=1, help="执行次数")
    parser.add_argument("-g", "--graph", action="store_true", help="开启torch的图模式")
    parser.add_argument("-dy", "--dynamic", action="store_true", help="torch图模式下,dynamic=True")
    parser.add_argument("--master_ip", type=str, default="127.0.0.1", help="主节点ip")

    parser.add_argument("--tp", type=int, default=2, help="tp_size")
    parser.add_argument("--op1", type=str, default="all_gather_base_mm", help="流1算子")
    parser.add_argument("--op2", type=str, default="mm_reduce_scatter_base", help="流2算子")
    args = parser.parse_args()

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    test_dict = {'bmm_reducescatter_alltoall': BatchMatMulReduceScatterAlltoAll,
                 'alltoall_allgather_bmm': AlltoAllAllGatherBatchMatMul,
                 'alltoallv_gmm': AlltoallvGmm, "gmm_alltoallv": GmmAlltoallv, 'all_gather_base_mm': AllGatherMatMul,
                 'mm_reduce_scatter_base': MatMulReduceScatter, 'moe_distribute_combine': MoeDistributeCombine,
                 'moe_distribute_combine_v2': MoeDistributeCombineV2, 'moe_update_expert': MoeUpdateExpert,
                 'moe_combine_add_rms_norm': MoeDistributeCombineAddRmsNorm,
                 'distribute_barrier': DistributeBarrier,
                 'matmul': MatMul}

    run_op_1 = test_dict[args.op1]
    run_op_2 = test_dict[args.op2]
    print(f"run_op_1:{run_op_1}, run_op_2:{run_op_2} ===============")

    p_list = []
    tp_size = args.tp
    ep_size = args.dev_num * args.server_num // tp_size
    rank_list = list(range(args.dev_num))


    for rank in rank_list:
        p = Process(target=run, args=(args, rank, run_op_1, run_op_2))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
