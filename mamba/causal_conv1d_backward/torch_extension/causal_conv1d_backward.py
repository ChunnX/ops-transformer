# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class CausalConv1dBackwardOpBuilder(OpBuilder):
    def __init__(self):
        super(CausalConv1dBackwardOpBuilder, self).__init__(
            "causal_conv1d_backward", category="mamba"
        )

    def sources(self):
        """Path to C++ source code."""
        # 打包后 csrc 落在 cann_ops_transformer/csrc/<category>/ 下（与正向 causal_conv1d_fn 同构）
        return ["csrc/mamba/causal_conv1d_backward.cpp"]

    def schema(self) -> str:
        """PyTorch operator schema."""
        return (
            "causal_conv1d_backward("
            "Tensor grad_y, Tensor x, Tensor weight, "
            "Tensor? query_start_loc=None, "
            "*, "
            'str activation="silu"'
            ") -> (Tensor, Tensor)"
        )

    def register_meta(self):
        """Register Meta implementation for shape/dtype inference."""

        @impl(get_as_library(), self.name, "Meta")
        def causal_conv1d_backward_meta(
            grad_y: torch.Tensor,
            x: torch.Tensor,
            weight: torch.Tensor,
            query_start_loc: Optional[torch.Tensor] = None,
            *,
            activation: str = "silu",
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del grad_y, query_start_loc, activation
            # weight 可能是非连续视图，输出按仓惯例恒为连续，故强制 contiguous_format
            grad_x = torch.empty_like(x, memory_format=torch.contiguous_format)
            grad_weight = torch.empty_like(weight, memory_format=torch.contiguous_format)
            return grad_x, grad_weight


_causal_conv1d_backward_op_builder = CausalConv1dBackwardOpBuilder()
_causal_conv1d_backward_op_builder._ensure_initialized()


@impl(get_as_library(), _causal_conv1d_backward_op_builder.name, "PrivateUse1")
def _causal_conv1d_backward(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    query_start_loc: Optional[torch.Tensor] = None,
    *,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    _op_module = _causal_conv1d_backward_op_builder.load()
    return _op_module.causal_conv1d_backward(
        grad_y,
        x,
        weight,
        query_start_loc,
        activation,
    )


def causal_conv1d_backward(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    query_start_loc: Optional[torch.Tensor] = None,
    *,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """因果一维深度卷积的反向（grad_x / grad_weight），封装 aclnnCausalConv1dBackward。

    对应正向 causal_conv1d_fn 在「bias=None、冷启动、单次 prefill」下的 VJP：
    正向 weight[W-1] 配当前 token x[t]、weight[0] 配最老 token x[t-(W-1)]，
    反向沿用同一份 [W, dim] 权重张量，无需翻序。

    Args:
        grad_y (Tensor): 卷积输出 y 的梯度。shape 为 [batch, seq_len, dim]（固定 batch）
            或 [cu_seq_len, dim]（变长），dtype 支持 float16/bfloat16，与 x 同形。
        x (Tensor): 卷积输入，与 grad_y 同形同 dtype。grad_weight 需要它。
        weight (Tensor): 卷积权重，shape 为 [kW, dim]，kW∈{2, 3, 4}，dtype 与 grad_y 一致。
            布局与因果序与正向 causal_conv1d_fn 的 weight 完全相同。
            允许非连续：可直接传 [dim, 1, kW] 参数的 transpose(0, 1) 视图，不必 contiguous。
        query_start_loc (Tensor, optional): 变长序列起始位置，shape 为 [batch+1]，int32。
            段内 token 才允许互相读写；None 表示整轴一段（无文档边界）。
            2D 变长输入必须提供，3D 固定 batch 输入可传 None。
        activation (str): 正向所用的激活，"silu" 或 "none"，默认 "silu"。

    Returns:
        tuple：grad_x（shape 与 dtype 同 x）、grad_weight（shape 与 dtype 同 weight）。
        两者恒为连续张量。
    """
    # 经 torch.ops 分发：eager 走 PrivateUse1 kernel（下方 @impl），dynamo/torchair
    # 图模式按 Meta + GE converter（graph_convert_causal_conv1d_backward.py）trace。
    return torch.ops.cann_ops_transformer.causal_conv1d_backward(
        grad_y,
        x,
        weight,
        query_start_loc,
        activation=activation,
    )
