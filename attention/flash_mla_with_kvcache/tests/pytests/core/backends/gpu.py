# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""GPU 后端占位（MLA 无 GPU 实现）。

flash_attn 的 tests/pytests/core/backends/gpu.py 提供 paged/varlen/fixed 三路径 GPU
对照；flash_mla_with_kvcache（DeepSeek MLA，Ascend 950 / DAV_3510 专用）当前**没有**
GPU 参考实现，本文件仅为对齐 flash_attn 的 tests 骨架而存在：

  - 可 import（模块级无任何初始化副作用，不触碰 torch/cuda）；
  - 所有方法体 raise NotImplementedError，供未来填充 GPU golden 时替换；
  - 本类**不**注册进 core/backends/__init__.py，也**不**被 orchestrator 的默认加载
    路径选中（orchestrator 通过 `from core.backends.base import Backend` 显式导入
    backend，本占位只由显式 ``from core.backends.gpu import GPUBackend`` 触发）。

接口对齐 flash_attn 同名文件：name / device / is_available / clear_cache / compute。
"""

from typing import Any, Dict

from .base import Backend


class GPUBackend(Backend):
    """GPU 后端占位：MLA 无 GPU 实现（详见模块 docstring）。"""

    name = "gpu"

    def __init__(self, device_id: int = 0):
        raise NotImplementedError(
            "flash_mla_with_kvcache 无 GPU 实现：GPUBackend 仅为 flash_attn 骨架占位，"
            "待未来补充 GPU golden（参照 flash_attn/tests/pytests/core/backends/gpu.py 的 "
            "paged/varlen/fixed 三路径，并适配 MLA 的 TND/NZ/NTD R1 布局）。"
        )

    @property
    def device(self):
        raise NotImplementedError("MLA 无 GPU 实现（占位）")

    def is_available(self) -> bool:
        raise NotImplementedError("MLA 无 GPU 实现（占位）")

    def clear_cache(self):
        raise NotImplementedError("MLA 无 GPU 实现（占位）")

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            "MLA 无 GPU 实现：compute 占位。若需 GPU golden，请参照 flash_attn 的 "
            "core/backends/gpu.py + gpu_impl.py（flash_attn_func / flash_attn_varlen_func "
            "/ flash_attn_with_kvcache）实现 MLA 语义后替换本方法。"
        )
