# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU Golden 后端实现 — 调用 MLA golden 做参考计算（TND/NZ/NTD per R1）。"""

import torch
from typing import Any, Dict

from .base import Backend


class CPUBackend(Backend):
    name = "cpu"

    def __init__(self):
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def is_available(self) -> bool:
        return True

    def compute(
        self, inputs: Dict[str, torch.Tensor], params: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        from core.backends.cpu_golden import mla_forward

        scale = params.get("scale", 1 / (params["D"] ** 0.5))
        out_cpu, lse = mla_forward(
            inputs["q"],
            inputs["k_cache"],
            cu_seqlens_q=params["cu_seqlens_q"],
            cache_seqlens=params["cache_seqlens"],
            block_table=params["block_table"],
            block_size=params.get("block_size", 128),
            seqused_q=params.get("seqused_q"),
            mask_mode=params.get("mask_mode", 0),
            scale=scale,
            N1=params.get("N1"),
        )
        # T11 迁移：out==q==TND（golden 与设备同布局契约，见 cpu_golden.mla_forward）
        t_total = int(inputs["q"].shape[0])
        n1 = int(inputs["q"].shape[1])
        assert tuple(out_cpu.shape) == (t_total, n1, int(params["D"])), (
            f"golden out shape {tuple(out_cpu.shape)} != TND ({t_total}, {n1}, {params['D']})"
        )
        lse = lse.clone()
        lse[lse.isinf() & (lse > 0)] = float("inf")
        return {"out": out_cpu, "lse": lse}
