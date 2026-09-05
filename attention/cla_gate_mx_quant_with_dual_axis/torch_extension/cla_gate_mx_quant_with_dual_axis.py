# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


class ClaGateMxQuantWithDualAxisOpBuilder(OpBuilder):
    """ClaGateMxQuantWithDualAxis 四输入 Torch Extension 构建器。"""

    def __init__(self):
        super().__init__("cla_gate_mx_quant_with_dual_axis")

    def sources(self) -> list:
        return [self.resolve_source("cla_gate_mx_quant_with_dual_axis.cpp")]

    def schema(self) -> str:
        return (
            "cla_gate_mx_quant_with_dual_axis("
            "Tensor global_out, "
            "Tensor local_out, "
            "Tensor global_gate_logits, "
            "Tensor local_gate_logits, "
            "*, "
            "int scale_alg=1, "
            "int dst_type=36"
            ") -> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def cla_gate_mx_quant_with_dual_axis_meta(
            global_out: torch.Tensor,
            local_out: torch.Tensor,
            global_gate_logits: torch.Tensor,
            local_gate_logits: torch.Tensor,
            *,
            scale_alg: int = 1,
            dst_type: int = 36,
        ):
            torch._check(global_out.dim() == 4, lambda: f"global_out must be 4D, but got {global_out.dim()}D")
            torch._check(local_out.dim() == 4, lambda: f"local_out must be 4D, but got {local_out.dim()}D")
            torch._check(
                local_out.shape == global_out.shape,
                lambda: f"local_out shape {tuple(local_out.shape)} must match global_out {tuple(global_out.shape)}",
            )

            gate_dim = global_gate_logits.dim()
            torch._check(
                gate_dim in (3, 4),
                lambda: f"global_gate_logits must be 3D or 4D, but got {gate_dim}D",
            )
            torch._check(
                local_gate_logits.dim() == gate_dim,
                lambda: "local_gate_logits rank must match global_gate_logits",
            )
            torch._check(
                local_gate_logits.shape == global_gate_logits.shape,
                lambda: "local_gate_logits shape must match global_gate_logits",
            )
            torch._check(
                scale_alg in (0, 1),
                lambda: f"scale_alg must be 0 or 1, but got {scale_alg}",
            )
            torch._check(
                dst_type in (35, 36),
                lambda: f"dst_type must be 35(float8_e5m2) or 36(float8_e4m3fn), but got {dst_type}",
            )

            if dst_type == 35:
                data_dtype = torch.float8_e5m2
            else:
                data_dtype = torch.float8_e4m3fn

            s, b, n, d = global_out.shape
            m = s * b
            k = n * d

            row_data = torch.empty([m, k], dtype=data_dtype, device="meta")
            col_data = torch.empty([m, k], dtype=data_dtype, device="meta")
            row_scale = torch.empty([m, k // 64, 2], dtype=torch.uint8, device="meta")
            col_scale = torch.empty([m // 64, k, 2], dtype=torch.uint8, device="meta")

            return row_data, row_scale, col_data, col_scale


cla_gate_mx_quant_with_dual_axis_builder = ClaGateMxQuantWithDualAxisOpBuilder()
cla_gate_mx_quant_with_dual_axis_builder._ensure_initialized()


@impl(
    get_as_library(),
    cla_gate_mx_quant_with_dual_axis_builder.name,
    "PrivateUse1",
)
def cla_gate_mx_quant_with_dual_axis(
    global_out: torch.Tensor,
    local_out: torch.Tensor,
    global_gate_logits: torch.Tensor,
    local_gate_logits: torch.Tensor,
    *,
    scale_alg: int = 1,
    dst_type: int = 36,
):
    """NPU 上的四输入 ClaGateMxQuantWithDualAxis 融合算子。"""
    op_module = cla_gate_mx_quant_with_dual_axis_builder.load()
    return op_module.cla_gate_mx_quant_with_dual_axis(
        global_out,
        local_out,
        global_gate_logits,
        local_gate_logits,
        scale_alg,
        dst_type,
    )
