# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from importlib import import_module


_EXPORTS = {
    "sparse_flash_mla_grad": ".sparse_flash_mla_grad",
    "sparse_flash_mla_grad_metadata": ".sparse_flash_mla_grad",
    "npu_moe_distribute_dispatch": ".moe_distribute_dispatch",
    "npu_moe_distribute_combine": ".moe_distribute_combine",
    "get_symm_buffer_for_mega_moe": ".mega_moe",
    "mega_moe": ".mega_moe",
    "npu_get_mega_moe_ccl_buffer_size": ".mega_moe",
    "MoeDistributeBuffer": ".deep_ep",
    "converter_moe_distribute_dispatch": ".graph_convert.graph_convert_moe_distribute_dispatch",
    "convert_npu_moe_distribute_combine": ".graph_convert.graph_convert_moe_distribute_combine",
    "convert_npu_mega_moe": ".graph_convert.graph_convert_mega_moe",
    "npu_flash_attn": ".flash_attn",
    "convert_npu_flash_attn": ".graph_convert.graph_convert_flash_attn",
    "npu_flash_attn_metadata": ".flash_attn_metadata",
    "mixed_quant_sparse_flash_mla": ".mixed_quant_sparse_flash_mla",
    "mixed_quant_sparse_flash_mla_metadata": ".mixed_quant_sparse_flash_mla",
    "convert_mixed_quant_sparse_flash_mla_metadata": ".graph_convert.graph_convert_mixed_quant_sparse_flash_mla",
    "sparse_flash_mla": ".sparse_flash_mla",
    "sparse_flash_mla_metadata": ".sparse_flash_mla",
    "convert_sparse_flash_mla_metadata": ".graph_convert.graph_convert_sparse_flash_mla",
    "sparse_lightning_indexer_kl_loss_grad": ".sparse_lightning_indexer_kl_loss_grad",
    "sparse_lightning_indexer_kl_loss_grad_metadata": ".sparse_lightning_indexer_kl_loss_grad",
    "lightning_indexer": ".lightning_indexer",
    "lightning_indexer_metadata": ".lightning_indexer",
    "convert_lightning_indexer_metadata": ".graph_convert.graph_convert_lightning_indexer",
    "quant_lightning_indexer": ".quant_lightning_indexer",
    "quant_lightning_indexer_metadata": ".quant_lightning_indexer",
    "convert_quant_lightning_indexer_metadata": ".graph_convert.graph_convert_quant_lightning_indexer",
    "mhc_post": ".mhc_post",
    "mhc_pre_sinkhorn": ".mhc_pre_sinkhorn",
    "qkv_rms_norm_rope_cache_with_k_scale": ".qkv_rms_norm_rope_cache_with_k_scale",
    "qkv_rms_norm_rope_cache_with_k_scale_functional": ".qkv_rms_norm_rope_cache_with_k_scale",
}

__all__ = tuple(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
