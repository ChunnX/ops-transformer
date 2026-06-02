try:
    import torch
    import torch_npu
    import torchair
    from typing import Optional
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
        [False, False, False, False, False, True, True, True, True, True, True, True],
    )
    def CombineNormal(
        context: Tensor,
        recv_x: Tensor,
        token_src_info: Tensor,
        ep_recv_counts: Tensor,
        recv_topk_weights: Tensor,
        token_idx: Tensor,
        tp_recv_counts: Optional[Tensor] = None,
        *,
        ep_world_size: int,
        ep_rank_id: int,
        moe_expert_num: int,
        ccl_buffer_size: int,
        tp_world_size: int = 0,
        tp_rank_id: int = 0,
        real_max_bs: int = 0,
        round: int = 4,
        per_round_tokens: int = 1024,
        combine_send_cost_stats: Optional[Tensor] = None,
        dependencies=[],
        node_name=None,
    ):
        inputs = {
            "context": context,
            "recv_x": recv_x,
            "token_src_info": token_src_info,
            "ep_recv_counts": ep_recv_counts,
            "recv_topk_weights": recv_topk_weights,
            "token_idx": token_idx,
        }
        if tp_recv_counts is not None:
            inputs["tp_recv_counts"] = tp_recv_counts
        if combine_send_cost_stats is not None:
            inputs["combine_send_cost_stats"] = combine_send_cost_stats

        attrs = {
            "ep_world_size": attr.Int(ep_world_size),
            "ep_rank_id": attr.Int(ep_rank_id),
            "moe_expert_num": attr.Int(moe_expert_num),
            "ccl_buffer_size": attr.Int(ccl_buffer_size),
            "tp_world_size": attr.Int(tp_world_size),
            "tp_rank_id": attr.Int(tp_rank_id),
            "real_max_bs": attr.Int(real_max_bs),
            "round": attr.Int(round),
            "per_round_tokens": attr.Int(per_round_tokens),
        }

        outputs = [
            "x",
            "combine_send_cost_stats",
        ]

        return ge_op(
            op_type="CombineNormal",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("CombineNormal")
            .input("context", "DT_INT32")
            .input("recv_x", "DT_BF16, DT_FLOAT16, DT_INT32")
            .input("token_src_info", "DT_INT32")
            .input("ep_recv_counts", "DT_INT32")
            .input("recv_topk_weights", "DT_FLOAT")
            .input("token_idx", "DT_INT32")
            .optional_input("tp_recv_counts", "DT_INT32")
            .required_attr("ep_world_size", attr.Int)
            .required_attr("ep_rank_id", attr.Int)
            .required_attr("moe_expert_num", attr.Int)
            .required_attr("ccl_buffer_size", attr.Int)
            .attr("tp_world_size", attr.Int(0))
            .attr("tp_rank_id", attr.Int(0))
            .attr("real_max_bs", attr.Int(0))
            .attr("round", attr.Int(4))
            .attr("per_round_tokens", attr.Int(1024))
            .output("x", "DT_BF16, DT_FLOAT16")
            .output("combine_send_cost_stats", "DT_INT32"),
        )

    @register_fx_node_ge_converter(torch.ops.npu_ops_transformer.npu_combine_normal.default)
    def convert_npu_combine_normal(
        context: Tensor,
        recv_x: Tensor,
        token_src_info: Tensor,
        ep_recv_counts: Tensor,
        recv_topk_weights: Tensor,
        token_idx: Tensor,
        ep_world_size: int,
        ep_rank_id: int,
        moe_expert_num: int,
        ccl_buffer_size: int,
        tp_recv_counts: Optional[Tensor] = None,
        *,
        tp_world_size: int = 0,
        tp_rank_id: int = 0,
        real_max_bs: int = 0,
        round: int = 4,
        per_round_tokens: int = 1024,
        combine_send_cost_stats: Tensor = None,
        meta_outputs: TensorSpec = None,
    ):
        outputs = CombineNormal(
            context=context,
            recv_x=recv_x,
            token_src_info=token_src_info,
            ep_recv_counts=ep_recv_counts,
            recv_topk_weights=recv_topk_weights,
            token_idx=token_idx,
            tp_recv_counts=tp_recv_counts,
            ep_world_size=ep_world_size,
            ep_rank_id=ep_rank_id,
            moe_expert_num=moe_expert_num,
            ccl_buffer_size=ccl_buffer_size,
            tp_world_size=tp_world_size,
            tp_rank_id=tp_rank_id,
            real_max_bs=real_max_bs,
            round=round,
            per_round_tokens=per_round_tokens,
            combine_send_cost_stats=combine_send_cost_stats,
        )
        return outputs[0]
