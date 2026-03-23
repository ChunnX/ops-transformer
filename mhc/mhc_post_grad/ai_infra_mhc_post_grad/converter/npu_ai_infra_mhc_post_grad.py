from typing import Any
import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor
from torchair.ge import attr


@register_fx_node_ge_converter(torch.ops.custom.npu_ai_infra_mhc_post_grad.default)
def convert_npu_ai_infra_mhc_post_grad(
    grad_output: Tensor,
    x: Tensor,
    h_res: Tensor,
    h_out: Tensor,
    h_post: Tensor,
    *,
    meta_outputs: Any = None
):
    return torchair.ge.custom_op(
        "AiInfraMhcPostGrad",
        inputs={
            "grad_output": grad_output,
            "x": x,
            "h_res": h_res,
            "h_out": h_out,
            "h_post": h_post
        },
        attrs={},
        outputs=["grad_x", "grad_h_res", "grad_h_out", "grad_h_post"]
    )
