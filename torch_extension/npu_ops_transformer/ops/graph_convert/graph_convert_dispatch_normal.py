try:
    import torch
    import torch_npu
    import torchair
    from typing import Optional, List
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:
    @auto_convert_to_tensor(
        [False, False, False, False, False, False, False],
        [False, False, False, True, True, True, True, True, True],
    )
    def DispatchNormal(
        context: Tensor,
        x: Tensor,
        topk_idx: Tensor,
        send_offset: Tensor,
        send_token_idx: Tensor,
        recv_offset: List[Tensor],
        expert_output_offset: List[Tensor],
        *,
        ep_world_size: int,
        ep_rank_id: int,
        moe_expert_num: int,
        ccl_buffer_size: int,
        quant_mode: int = 0,
        real_max_bs: int = 0,
        global_bs: int = 0,
        round: int = 4,
        per_round_tokens: int = 1024,
        dispatch_wait_recv_cost_stats: Optional[Tensor] = None,
        dependencies=[],
        node_name=None,
    ):
        inputs = {
            "context": context,
            "x": x,
            "topk_idx": topk_idx,
            "send_offset": send_offset,
            "send_tokenIdx": send_token_idx,
            "recv_offset": recv_offset,
            "expert_output_offset": expert_output_offset,
        }
        if dispatch_wait_recv_cost_stats is not None:
            inputs["dispatch_wait_recv_cost_stats"] = dispatch_wait_recv_cost_stats

        attrs = {
            "ep_world_size": attr.Int(ep_world_size),
            "ep_rank_id": attr.Int(ep_rank_id),
            "moe_expert_num": attr.Int(moe_expert_num),
            "ccl_buffer_size": attr.Int(ccl_buffer_size),
            "quant_mode": attr.Int(quant_mode),
            "real_max_bs": attr.Int(real_max_bs),
            "global_bs": attr.Int(global_bs),
            "round": attr.Int(round),
            "per_round_tokens": attr.Int(per_round_tokens),
        }

        outputs = [
            "recv_x",
            "x_scales",
            "assist_info_for_combine",
            "dispatch_wait_recv_cost_stats",
            "ext_handle_data",
        ]

        return ge_op(
            op_type="DispatchNormal",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("DispatchNormal")
            .input("context", "DT_INT32")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("topk_idx", "DT_INT32")
            .input("send_offset", "DT_INT32")
            .input("send_tokenIdx", "DT_INT32")
            .dynamic_input("recv_offset", "DT_INT32")
            .dynamic_input("expert_output_offset", "DT_INT32")
            .required_attr("ep_world_size", attr.Int)
            .required_attr("ep_rank_id", attr.Int)
            .required_attr("moe_expert_num", attr.Int)
            .required_attr("ccl_buffer_size", attr.Int)
            .attr("quant_mode", attr.Int(0))
            .attr("real_max_bs", attr.Int(0))
            .attr("global_bs", attr.Int(0))
            .attr("round", attr.Int(4))
            .attr("per_round_tokens", attr.Int(1024))
            .output("recv_x", "DT_BF16, DT_INT8, DT_FLOAT16")
            .output("x_scales", "DT_FLOAT")
            .output("assist_info_for_combine", "DT_INT32")
            .output("dispatch_wait_recv_cost_stats", "DT_INT32")
            .output("ext_handle_data", "DT_INT32"),
        )

    @register_fx_node_ge_converter(torch.ops.npu_ops_transformer.npu_dispatch_normal.default)
    def convert_npu_dispatch_normal(
        context: Tensor,
        x: Tensor,
        topk_idx: Tensor,
        send_offset: Tensor,
        send_token_idx: Tensor,
        recv_offset: List[Tensor],
        expert_output_offset: List[Tensor],
        ep_world_size: int,
        ep_rank_id: int,
        moe_expert_num: int,
        ccl_buffer_size: int,
        *,
        num_recv_tokens: int,
        quant_mode: int = 0,
        real_max_bs: int = 0,
        global_bs: int = 0,
        round: int = 4,
        per_round_tokens: int = 1024,
        dispatch_wait_recv_cost_stats: Tensor = None,
        ext_handle_data: Tensor = None,
        meta_outputs: TensorSpec = None,
    ):
        if ext_handle_data is not None:
            raise AssertionError(
                "GE converter does not support preallocated ext_handle_data input."
            )
        outputs = DispatchNormal(
            context=context,
            x=x,
            topk_idx=topk_idx,
            send_offset=send_offset,
            send_token_idx=send_token_idx,
            recv_offset=recv_offset,
            expert_output_offset=expert_output_offset,
            ep_world_size=ep_world_size,
            ep_rank_id=ep_rank_id,
            moe_expert_num=moe_expert_num,
            ccl_buffer_size=ccl_buffer_size,
            quant_mode=quant_mode,
            real_max_bs=real_max_bs,
            global_bs=global_bs,
            round=round,
            per_round_tokens=per_round_tokens,
            dispatch_wait_recv_cost_stats=dispatch_wait_recv_cost_stats,
        )
        # Keep runtime signature aligned with python API: return 4 tensors.
        return outputs[0], outputs[1], outputs[2], outputs[4]
