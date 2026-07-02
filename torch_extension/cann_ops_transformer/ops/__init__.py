# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from .sparse_flash_mla_grad import sparse_flash_mla_grad, sparse_flash_mla_grad_metadata
from .moe_distribute_dispatch import npu_moe_distribute_dispatch
from .moe_distribute_combine import npu_moe_distribute_combine
from .moe_token_permute import moe_token_permute
from .mega_moe import (
    get_symm_buffer_for_mega_moe,
    mega_moe,
    get_mega_moe_ccl_buffer_size
)
from .deep_ep import MoeDistributeBuffer
from .flash_attn import flash_attn, flash_attn_metadata
from .mixed_quant_sparse_flash_mla import mixed_quant_sparse_flash_mla, mixed_quant_sparse_flash_mla_metadata
from .scatter_pa_kv_cache_with_k_scale import scatter_pa_kv_cache_with_k_scale
from .sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata
from .sparse_lightning_indexer_kl_loss_grad import (
    sparse_lightning_indexer_kl_loss_grad,
    sparse_lightning_indexer_kl_loss_grad_metadata
)
from .lightning_indexer import lightning_indexer, lightning_indexer_metadata
from .quant_lightning_indexer import quant_lightning_indexer, quant_lightning_indexer_metadata
from .mhc_post import mhc_post
from .mhc_pre_sinkhorn import mhc_pre_sinkhorn
from .kv_compress_epilog import kv_compress_epilog
from .indexer_quant_cache import indexer_quant_cache
from .compressor import compressor
from .inplace_partial_rotary_mul import inplace_partial_rotary_mul
from . import graph_convert as _graph_convert
from .causal_conv1d_fn import causal_conv1d_fn
from .causal_conv1d_update import causal_conv1d_update


def __getattr__(name):
    if name in ("ElasticBuffer", "EPHandle"):
        from .elastic_buffer import ElasticBuffer, EPHandle
        value = {"ElasticBuffer": ElasticBuffer, "EPHandle": EPHandle}[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "sparse_flash_mla_grad",
    "sparse_flash_mla_grad_metadata",
    "npu_moe_distribute_dispatch",
    "npu_moe_distribute_combine",
    "moe_token_permute",
    "get_symm_buffer_for_mega_moe",
    "mega_moe",
    "get_mega_moe_ccl_buffer_size",
    "MoeDistributeBuffer",
    "flash_attn",
    "flash_attn_metadata",
    "mixed_quant_sparse_flash_mla",
    "mixed_quant_sparse_flash_mla_metadata",
    "scatter_pa_kv_cache_with_k_scale",
    "sparse_flash_mla",
    "sparse_flash_mla_metadata",
    "sparse_lightning_indexer_kl_loss_grad",
    "sparse_lightning_indexer_kl_loss_grad_metadata",
    "lightning_indexer",
    "lightning_indexer_metadata",
    "quant_lightning_indexer",
    "quant_lightning_indexer_metadata",
    "mhc_post",
    "mhc_pre_sinkhorn",
    "kv_compress_epilog",
    "indexer_quant_cache",
    "compressor",
    "inplace_partial_rotary_mul",
    "causal_conv1d_fn",
    "causal_conv1d_update",
]
