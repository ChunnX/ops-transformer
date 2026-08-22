#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""GPU flash_attn API 薄封装层占位（MLA 无 GPU 实现）。

flash_attn 的 gpu_impl.py 在 `from flash_attn import ...` 可用时导出
gpu_fixed_attn / gpu_varlen_attn / gpu_paged_attn 三个实现；flash_mla_with_kvcache
当前没有 GPU 参考算子，本文件保留**同名接口占位**：

  - FLASH_ATTN_AVAILABLE = False（与 flash_attn 侧 ImportError 时语义一致）；
  - 三个函数体 raise NotImplementedError，注明 MLA 语义说明；
  - 模块可独立 import（无 torch/flash_attn 依赖），零初始化副作用。

不会被 core/backends/__init__.py 导出，也不会被 orchestrator 自动加载
（orchestrator 显式 `from core.backends.base import Backend` 导入）。
"""

FLASH_ATTN_AVAILABLE = False

_MLA_NOT_IMPLEMENTED = (
    "MLA 无 GPU 实现（占位）：flash_mla_with_kvcache 依赖 Ascend 950 / DAV_3510 的 "
    "flash_mla_metadata + flash_mla_with_kvcache，无 flash_attn GPU API 等价物。"
    "待未来补充 GPU golden（TND/NZ/NTD R1 布局）时替换本函数。"
)


def gpu_fixed_attn(q_bshd, k_bshd, v_bshd, softmax_scale, causal):
    """固定长度 attention 占位（见模块 docstring）。"""
    raise NotImplementedError(_MLA_NOT_IMPLEMENTED)


def gpu_varlen_attn(
    q_tnd, k_tnd, v_tnd, cu_q, cu_kv, max_sq, max_skv, softmax_scale, causal, device
):
    """变长 attention 占位（见模块 docstring）。"""
    raise NotImplementedError(_MLA_NOT_IMPLEMENTED)


def gpu_paged_attn(
    q_bshd, k_cache, v_cache, block_table, cache_seqlens, softmax_scale, causal
):
    """Paged KV cache attention 占位（见模块 docstring）。"""
    raise NotImplementedError(_MLA_NOT_IMPLEMENTED)
