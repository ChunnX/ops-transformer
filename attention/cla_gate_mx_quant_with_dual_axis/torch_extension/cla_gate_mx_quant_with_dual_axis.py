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

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class ClaGateMxQuantWithDualAxisOpBuilder(OpBuilder):
    """ClaGateMxQuantWithDualAxis 四输入 Torch Extension 构建器。"""

    def __init__(self):
        super().__init__("cla_gate_mx_quant_with_dual_axis", category="attention")

    def sources(self) -> list:
        return ["csrc/attention/cla_gate_mx_quant_with_dual_axis.cpp"]

    def schema(self) -> str:
        return (
            "cla_gate_mx_quant_with_dual_axis("
            "Tensor global_out, "
            "Tensor local_out, "
            "Tensor global_gate_logits, "
            "Tensor local_gate_logits, "
            "*, "
            "str round_mode='rint', "
            "int scale_alg=1, "
            "int dst_type=36, "
            "int layout=0, "
            "int quant_mode=1"
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
            round_mode: str = "rint",
            scale_alg: int = 1,
            dst_type: int = 36,
            layout: int = 0,
            quant_mode: int = 1,
        ):
            torch._check(global_out.dim() == 3, lambda: f"global_out must be 3D [T,N,D], but got {global_out.dim()}D")
            torch._check(local_out.dim() == 3, lambda: f"local_out must be 3D [T,N,D], but got {local_out.dim()}D")
            torch._check(
                local_out.shape == global_out.shape,
                lambda: f"local_out shape {tuple(local_out.shape)} must match global_out {tuple(global_out.shape)}",
            )
            t, n, d = global_out.shape
            torch._check(1 <= n <= 128, lambda: f"N must be in [1,128], got {n}")
            torch._check(d in (128, 256), lambda: f"D must be 128 or 256, got {d}")

            gate_dim = global_gate_logits.dim()
            torch._check(
                gate_dim == 2,
                lambda: f"global_gate_logits must be 2D [T,N], but got {gate_dim}D",
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
                global_gate_logits.shape[0] == t and global_gate_logits.shape[1] == n,
                lambda: "gate logits shape must match [T,N]",
            )
            torch._check(
                dst_type in (35, 36, 40, 41),
                lambda: f"torch extension supports dst_type 35/36(FP8) or 40/41(FP4), got {dst_type}",
            )
            torch._check(
                round_mode in ("rint", "round", "floor"),
                lambda: f"round_mode must be rint/round/floor, got {round_mode}",
            )
            torch._check(
                scale_alg in (0, 1),
                lambda: f"scale_alg must be 0 or 1, but got {scale_alg}",
            )
            torch._check(
                layout == 0,
                lambda: f"layout must be 0 (TND), but got {layout}",
            )
            torch._check(
                quant_mode in (0, 1),
                lambda: f"quant_mode must be 0 or 1, but got {quant_mode}",
            )
            if dst_type in (40, 41):
                torch._check(
                    scale_alg == 0,
                    lambda: "FP4 only supports scale_alg=0",
                )
            else:
                torch._check(
                    round_mode == "rint",
                    lambda: f"FP8 only supports round_mode=rint, got {round_mode}",
                )

            k = n * d
            if dst_type in (40, 41):
                # torch has no native FP4 dtype; expose packed uint8 (2 values/byte).
                data_dtype = torch.uint8
                data_cols = k // 2
            else:
                data_dtype = torch.float8_e5m2 if dst_type == 35 else torch.float8_e4m3fn
                data_cols = k

            row_data = torch.empty([t, data_cols], dtype=data_dtype, device="meta")
            row_scale = torch.empty([t, (k + 63) // 64, 2], dtype=torch.uint8, device="meta")
            if quant_mode == 0:
                col_data = torch.empty([0], dtype=data_dtype, device="meta")
                col_scale = torch.empty([0], dtype=torch.uint8, device="meta")
            else:
                col_data = torch.empty([t, data_cols], dtype=data_dtype, device="meta")
                col_scale = torch.empty([(t + 63) // 64, k, 2], dtype=torch.uint8, device="meta")

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
    round_mode: str = "rint",
    scale_alg: int = 1,
    dst_type: int = 36,
    layout: int = 0,
    quant_mode: int = 1,
):
    """NPU 上的四输入 ClaGateMxQuantWithDualAxis 融合算子。"""
    op_module = cla_gate_mx_quant_with_dual_axis_builder.load()
    return op_module.cla_gate_mx_quant_with_dual_axis(
        global_out,
        local_out,
        global_gate_logits,
        local_gate_logits,
        round_mode,
        scale_alg,
        dst_type,
        layout,
        quant_mode,
    )
