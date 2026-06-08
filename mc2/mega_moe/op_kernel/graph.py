import contextlib
import torch
import torch_npu
import torchair
from npu_ops_transformer.ops import mega_moe

FLOAT8_E8M0FNU_DTYPE = getattr(torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None))

class CascadeGraphModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loop = kwargs['loop']
        self.is_performance = kwargs['is_performance']
        self.aic_core_num = kwargs['aic_core_num']
        self.aiv_core_num = kwargs['aiv_core_num']
        self.is_graph = kwargs['is_graph']
        # self.input_dtype = kwargs['input_dtype']


    def forward(
            self, x, expert_ids, expert_scales, x_active_mask, group_ep, group_tp, ep_rank_id, ep_world_size,
            moe_expert_num, quant_mode, global_bs, gmm1_weight, output_dtype, gmm1_weight_scale, gmm2_weight,
            gmm2_weight_scale, y_dtype, comm_quant_mode, rank, weight_dtype, swiglu_quant_dtype, x_dtype
    ):
        local_loop = self.loop if self.is_performance and self.is_graph else 1
        # ctx = torchair.scope.limit_core_num(self.aic_core_num, self.aiv_core_num) if self.is_graph else contextlib.nullcontext()
        # with ctx:
        for _ in range(local_loop):
            output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(
                x=x,
                expert_ids=expert_ids,
                x_active_mask=None,
                group_ep=group_ep,
                ep_world_size=ep_world_size,
                ep_rank_id=ep_rank_id,
                tp_world_size=1,
                moe_expert_num=moe_expert_num,
                expert_shard_type=0,
                shared_expert_num=1,
                shared_expert_rank_num=0,
                quant_mode=quant_mode,
                global_bs=global_bs,
                y_dtype=y_dtype,
            )

            (expand_x_npu, dynamic_scales_npu, assist_info_for_combine_npu, expert_token_nums_npu, ep_recv_counts_npu,
                tp_send_counts_npu, _) = output_dispatch_npu
            dynamic_scales_npu = dynamic_scales_npu.cpu()
            dynamic_scales_npu = dynamic_scales_npu.to(torch.float8_e8m0fnu)
            dynamic_scales_npu = dynamic_scales_npu.npu()

            if quant_mode == 4:
                if expand_x_npu.dtype == gmm1_weight.dtype:
                    y1 = torch_npu.npu_grouped_matmul(
                        x=[expand_x_npu],
                        weight=[gmm1_weight.transpose(1, 2)],
                        group_list=expert_token_nums_npu,
                        group_type=0,
                        group_list_type=1,
                        per_token_scale=[dynamic_scales_npu.reshape(dynamic_scales_npu.shape[0], dynamic_scales_npu.shape[1] // 2, 2)],
                        scale=[gmm1_weight_scale.transpose(1, 2)],
                        split_item=2,
                        output_dtype=output_dtype,
                        per_token_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                        weight_dtype=weight_dtype,
                        )[0]
                    y_swiglu_quant, y_swiglu_scale = torch_npu.npu_swiglu_mx_quant(
                        x=y1,
                        dst_type=swiglu_quant_dtype,
                        activate_left=True,
                        )
                else:
                    gmm1_weight = torch_npu.npu_format_cast(gmm1_weight, 29, customize_dtype=y_dtype, input_dtype=torch_npu.float4_e2m1fn_x2)
                    y1 = torch_npu.npu_grouped_matmul(
                        x=[expand_x_npu],
                        weight=[gmm1_weight.transpose(1, 2)],
                        group_list=expert_token_nums_npu,
                        group_type=0,
                        group_list_type=1,
                        per_token_scale=[dynamic_scales_npu.reshape(dynamic_scales_npu.shape[0], dynamic_scales_npu.shape[1] // 2, 2)],
                        antiquant_scale=[gmm1_weight_scale.transpose(1, 2)],
                        split_item=2,
                        output_dtype=output_dtype,
                        per_token_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                        weight_dtype=weight_dtype,
                        )[0]
                    y_swiglu_quant, y_swiglu_scale = torch_npu.npu_swiglu_mx_quant(
                        x=y1,
                        dst_type=torch.float8_e4m3fn,
                        activate_left=True,
                        )
                
                if weight_dtype is None:

                    y2 = torch_npu.npu_grouped_matmul(x=[y_swiglu_quant],
                                                        weight=[gmm2_weight.transpose(1, 2)],
                                                        # weight=[gmm2_weight],
                                                        scale=[gmm2_weight_scale.transpose(1, 2)] if quant_mode > 0 else None,
                                                        #scale=[gmm2_weight_scale] if quant_mode > 0 else None,
                                                        per_token_scale=[y_swiglu_scale] if quant_mode > 0 else None,
                                                        split_item=2,
                                                        group_list_type=1,
                                                        group_type=0,
                                                        group_list=expert_token_nums_npu,
                                                        output_dtype=output_dtype,
                                                        scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                                                        per_token_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                                                        weight_dtype=weight_dtype,
                                                        x_dtype=y_dtype,)[0]
                else:
                    # y1_swiglu = torch_npu.npu_swiglu(y1)
                    # y_swiglu_quant, y_swiglu_scale = torch_npu.npu_dynamic_mx_quant(
                    #     input=y1_swiglu,
                    #     axis=-1,
                    #     dst_type=swiglu_quant_dtype,
                    #     block_size=32
                    #     )
                    gmm2_weight = torch_npu.npu_format_cast(gmm2_weight, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2)
                    y2 = torch_npu.npu_grouped_matmul(
                        x=[y_swiglu_quant],
                        weight=[gmm2_weight.transpose(1, 2)],
                        group_list=expert_token_nums_npu,
                        group_type=0,
                        group_list_type=1,
                        per_token_scale=[y_swiglu_scale],
                        antiquant_scale=[gmm2_weight_scale.transpose(1, 2)],
                        split_item=2,
                        output_dtype=output_dtype,
                        per_token_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                        weight_dtype=weight_dtype,
                        )[0]

            ####非量化暂不支持
            # else:
            #     y1_int32 = torch_npu.npu_grouped_matmul(
            #         x=[expand_x_npu],
            #         weight=[gmm1_weight],
            #         split_item=3,
            #         group_list_type=1,  # 默认为0，代表前缀和形式
            #         group_type=0,  # 0代表m轴分组
            #         group_list=expert_token_nums_npu,
            #         output_dtype=output_dtype)[0]
            #     y1 = torch_npu.npu_swiglu(y1_int32)
            #     y2 = torch_npu.npu_grouped_matmul(x=[y1],
            #                                         weight=[gmm2_weight],
            #                                         scale=None,
            #                                         per_token_scale=None,
            #                                         split_item=2,
            #                                         group_list_type=1,
            #                                         group_type=0,
            #                                         group_list=expert_token_nums_npu,
            #                                         output_dtype=output_dtype,
            #                                     )[0]

            output_combine_npu = torch_npu.npu_moe_distribute_combine_v2(
                expand_x=y2,
                expert_ids=expert_ids,
                assist_info_for_combine=assist_info_for_combine_npu,
                ep_send_counts=ep_recv_counts_npu,
                expert_scales=expert_scales,
                x_active_mask=None,
                group_ep=group_ep,
                ep_world_size=ep_world_size,
                ep_rank_id=ep_rank_id,
                moe_expert_num=moe_expert_num,
                tp_send_counts=tp_send_counts_npu,
                tp_world_size=1,
                expert_shard_type=0,
                shared_expert_num=1,
                shared_expert_rank_num=0,
                global_bs=global_bs,
                comm_quant_mode=comm_quant_mode,
            )

        return output_combine_npu, output_dispatch_npu

class MoeDispatchFFNCombineGraphModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loop = kwargs['loop']
        self.is_performance = kwargs['is_performance']
        self.aic_core_num = kwargs['aic_core_num']
        self.aiv_core_num = kwargs['aiv_core_num']
        self.is_graph = kwargs['is_graph']

    def forward(
            self, context, x, topk_ids, topk_weights, l1_weights, l2_weights, sym_buffer,
            rank, l1_weights_sf, l2_weights_sf, x_active_mask, weight1_type, weight2_type
    ):
        output_combine_npu = None
        combine_x = None
        local_loop = self.loop if self.is_performance and self.is_graph else 1
        # ctx = torchair.scope.limit_core_num(self.aic_core_num, self.aiv_core_num) if self.is_graph else contextlib.nullcontext()
        # with ctx:
        for _ in range(local_loop):
            output_combine_npu, expert_token_nums = mega_moe(
                context=context,
                x=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                l1_weights=[l1_weights],
                l2_weights=[l2_weights],
                sym_buffer=sym_buffer,
                scales=None,
                l1_weights_sf=[l1_weights_sf] if l1_weights_sf is not None else None,
                l2_weights_sf=[l2_weights_sf] if l2_weights_sf is not None else None,
                x_active_mask=None,
                weight1_type=weight1_type,
                weight2_type=weight2_type,
            )

        return output_combine_npu, expert_token_nums