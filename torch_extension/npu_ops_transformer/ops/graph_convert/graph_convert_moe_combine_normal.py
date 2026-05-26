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
        [False, False, False, False, False],
        [False, False, False, False, False, True, True, True, True, True, True, True, True],
    )
    def MoeCombineNormal(
        recv_x: Tensor,
        token_src_info: Tensor,
        ep_recv_counts: Tensor,
        recv_topk_weights: Tensor,
        tp_recv_counts: Optional[Tensor] = None,
        *,
        ep_group_name: str,
        ep_world_size: int,
        ep_rank_id: int,
        moe_expert_num: int,
        tp_group_name: str = "",
        tp_world_size: int = 0,
        tp_rank_id: int = 0,
        global_bs: int = 0,
        combine_send_cost_stats: Optional[Tensor] = None,
        dependencies=[],
        node_name=None,
    ):
        inputs = {
            "recv_x": recv_x,
            "token_src_info": token_src_info,
            "ep_recv_counts": ep_recv_counts,
            "recv_topk_weights": recv_topk_weights,
        }
        if tp_recv_counts is not None:
            inputs["tp_recv_counts"] = tp_recv_counts

        attrs = {
            "ep_group_name": attr.String(ep_group_name),
            "ep_world_size": attr.Int(ep_world_size),
            "ep_rank_id": attr.Int(ep_rank_id),
            "moe_expert_num": attr.Int(moe_expert_num),
            "tp_group_name": attr.String(tp_group_name),
            "tp_world_size": attr.Int(tp_world_size),
            "tp_rank_id": attr.Int(tp_rank_id),
            "global_bs": attr.Int(global_bs),
        }

        outputs = [
            "x",
            "combine_send_cost_stats",
        ]

        return ge_op(
            op_type="MoeCombineNormal",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MoeCombineNormal")
            .input("recv_x", "DT_BF16, DT_FLOAT16, DT_INT32")
            .input("token_src_info", "DT_INT32")
            .input("ep_recv_counts", "DT_INT32")
            .input("recv_topk_weights", "DT_FLOAT")
            .optional_input("tp_recv_counts", "DT_INT32")
            .required_attr("ep_group_name", attr.String)
            .required_attr("ep_world_size", attr.Int)
            .required_attr("ep_rank_id", attr.Int)
            .required_attr("moe_expert_num", attr.Int)
            .attr("tp_group_name", attr.String(""))
            .attr("tp_world_size", attr.Int(0))
            .attr("tp_rank_id", attr.Int(0))
            .attr("global_bs", attr.Int(0))
            .output("x", "DT_BF16, DT_FLOAT16")
            .output("combine_send_cost_stats", "DT_INT32"),
        )

    @register_fx_node_ge_converter(torch.ops.npu_ops_transformer.npu_moe_combine_normal.default)
    def convert_npu_moe_combine_normal(
        recv_x: Tensor,
        token_src_info: Tensor,
        ep_recv_counts: Tensor,
        recv_topk_weights: Tensor,
        tp_recv_counts: Optional[Tensor] = None,
        ep_group_name: str = "",
        ep_world_size: int = 0,
        ep_rank_id: int = 0,
        moe_expert_num: int = 0,
        *,
        tp_group_name: str = "",
        tp_world_size: int = 0,
        tp_rank_id: int = 0,
        global_bs: int = 0,
        combine_send_cost_stats: Tensor = None,
        meta_outputs: TensorSpec = None,
    ):
        outputs = MoeCombineNormal(
            recv_x=recv_x,
            token_src_info=token_src_info,
            ep_recv_counts=ep_recv_counts,
            recv_topk_weights=recv_topk_weights,
            tp_recv_counts=tp_recv_counts,
            ep_group_name=ep_group_name,
            ep_world_size=ep_world_size,
            ep_rank_id=ep_rank_id,
            moe_expert_num=moe_expert_num,
            tp_group_name=tp_group_name,
            tp_world_size=tp_world_size,
            tp_rank_id=tp_rank_id,
            global_bs=global_bs,
            combine_send_cost_stats=combine_send_cost_stats,
        )
        return outputs[0]