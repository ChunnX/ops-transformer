# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from importlib import import_module

_EXPORTS = {
    "npu_moe_distribute_dispatch_v2": ".moe_distribute_dispatch_v2",
    "npu_moe_distribute_combine_v2": ".moe_distribute_combine_v2",
    "npu_moe_distribute_dispatch_v3": ".moe_distribute_dispatch_v3",
    "npu_moe_distribute_combine_v3": ".moe_distribute_combine_v3",
    "npu_chunk_gated_delta_rule": ".chunk_gated_delta_rule",
    "npu_chunk_gated_delta_rule_C128": ".chunk_gated_delta_rule_C128",
    "npu_chunk_gated_delta_rule_neumann": ".chunk_gated_delta_rule_neumann",
    "npu_tri_inverse": ".tri_inverse",
    "MoeDistributeBuffer": ".deep_ep",
    "converter_moe_distribute_dispatch_v3": ".graph_convert.graph_convert_moe_distribute_dispatch_v3",
    "convert_npu_moe_distribute_combine_v3": ".graph_convert.graph_convert_moe_distribute_combine_v3",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
