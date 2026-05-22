# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

_LAZY_IMPORTS = {
    "npu_sparse_flash_mla": ".sparse_flash_mla",
    "npu_moe_distribute_dispatch_v2": ".moe_distribute_dispatch_v2",
    "npu_moe_distribute_combine_v2": ".moe_distribute_combine_v2",
    "npu_moe_distribute_dispatch_v3": ".moe_distribute_dispatch_v3",
    "npu_moe_distribute_combine_v3": ".moe_distribute_combine_v3",
    "get_symm_buffer_for_mega_moe": ".mega_moe",
    "mega_moe": ".mega_moe",
    "npu_get_mega_moe_ccl_buffer_size": ".mega_moe",
    "MoeDistributeBuffer": ".deep_ep",
    "converter_moe_distribute_dispatch_v3": ".graph_convert.graph_convert_moe_distribute_dispatch_v3",
    "convert_npu_moe_distribute_combine_v3": ".graph_convert.graph_convert_moe_distribute_combine_v3",
    "convert_npu_mega_moe": ".graph_convert.graph_convert_mega_moe",
    "npu_flash_attn": ".flash_attn",
    "convert_npu_flash_attn": ".graph_convert.graph_convert_flash_attn",
    "npu_flash_attn_metadata": ".flash_attn_metadata",
    "npu_lightning_indexer_v2_metadata": ".lightning_indexer_v2_metadata",
    "npu_quant_lightning_indexer_v2_metadata": ".quant_lightning_indexer_v2_metadata",
    "convert_npu_lightning_indexer_v2_metadata": ".graph_convert.graph_convert_lightning_indexer_v2_metadata",
    "convert_npu_quant_lightning_indexer_v2_metadata": ".graph_convert.graph_convert_quant_lightning_indexer_v2_metadata",
}

_LOADED = {}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        if name not in _LOADED:
            import importlib
            module = importlib.import_module(_LAZY_IMPORTS[name], __package__)
            _LOADED[name] = getattr(module, name)
        return _LOADED[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LAZY_IMPORTS.keys())
