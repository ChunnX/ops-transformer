# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent FP64 reference for value-domain FP8 adjudication."""
from __future__ import annotations
import torch
from attn_prologue_golden import (
    apply_tail_rope, dequantize_mxfp8, quantize_mxfp8, rms_norm,
)


class FP64Shadow:
    def __init__(self, weights):
        self.weights = weights
        self._dequantized = {}

    def _weight(self, name):
        if name not in self._dequantized:
            self._dequantized[name] = dequantize_mxfp8(
                self.weights.fields[name], self.weights.fields[f"descale_{name}"],
                torch.float64)
        return self._dequantized[name]

    def recompute_rows(self, inputs, rows, output, *, norm_eps=1e-6):
        """Return (FP8 bytes, actual cast values) for complete independent rows."""
        payload, _, cast_values, _ = self.recompute_full(
            inputs, rows, output, norm_eps=norm_eps)
        return payload, cast_values

    def recompute_full(self, inputs, rows, output, *, norm_eps=1e-6):
        """Return FP64 (payload, scale, cast values, pre-scale values)."""
        rows = torch.as_tensor(rows, dtype=torch.int64, device="cpu")
        # CPU advanced indexing for float8 is not available in all PyTorch
        # releases. Gather bytes then reinterpret without changing their values.
        x = inputs.x.view(torch.uint8)[rows].contiguous().view(torch.float8_e4m3fn)
        x = dequantize_mxfp8(x, inputs.descale_x[rows], torch.float64)
        if output == "qr":
            projected = x @ self._weight("wqa").T
            normalized = rms_norm(projected, inputs.norm_weight_qr, norm_eps,
                                  torch.float64)
            qr, scale, cast_values = quantize_mxfp8(normalized, torch.float64)
            return qr.view(torch.uint8), scale, cast_values, normalized
        if output == "kv":
            projected = x @ self._weight("wkv").T
            normalized = rms_norm(projected, inputs.norm_weight_kv, norm_eps,
                                  torch.float64)
            roped = apply_tail_rope(normalized, inputs.rope_sin[rows],
                                    inputs.rope_cos[rows], compute_dtype=torch.float64)
            cast_values = roped.clamp(-448., 448.)
            return (cast_values.to(torch.float8_e4m3fn).view(torch.uint8), None,
                    cast_values, cast_values)
        raise ValueError("output must be qr or kv")

    def review(self, inputs, candidates, *, boundary_rel=1e-5, norm_eps=1e-6):
        """Re-evaluate every supplied candidate, preserving original metadata.

        candidates must contain output ('qr'/'kv'), row, column and got_byte;
        may contain FP32 expected_byte and expected_cast_value diagnostics.
        Return exact / adjacent-boundary / hard per candidate, using FP64 values
        throughout the boundary calculation (no FP32 downcast of the reference).
        """
        if boundary_rel != 1e-5:
            raise ValueError("shadow uses the frozen boundary_rel=1e-5")
        result = []
        for output in ("qr", "kv"):
            group = [item for item in candidates if item["output"] == output]
            if not group:
                continue
            rows = sorted({item["row"] for item in group})
            indices = {row: index for index, row in enumerate(rows)}
            expected, values = self.recompute_rows(inputs, rows, output, norm_eps=norm_eps)
            for item in group:
                index, column = indices[item["row"]], item["column"]
                got_byte, ref_byte = item["got_byte"], int(expected[index,column])
                truth = float(values[index,column])
                decoded = torch.tensor([got_byte,ref_byte],dtype=torch.uint8).view(torch.float8_e4m3fn).double()
                got, ref = float(decoded[0]), float(decoded[1])
                exact = got_byte == ref_byte
                midpoint = (got+ref)*.5
                near = abs(truth-midpoint) <= boundary_rel*max(abs(truth),1e-30)
                adjacent = abs(got_byte-ref_byte) == 1
                finite = bool(torch.isfinite(decoded).all())
                between = min(got,ref) <= truth <= max(got,ref)
                tie = not exact and adjacent and finite and between and near
                result.append(dict(**item, shadow_expected_byte=ref_byte,
                    shadow_cast_value=truth,
                    shadow_classification="exact" if exact else "boundary" if tie else "hard",
                    shadow_midpoint_relative=abs(truth-midpoint)/max(abs(truth),1e-30),
                    boundary_rel=boundary_rel, diagnostic_only=True))
        return result
