# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import importlib
import logging
import os
import sys


logger = logging.getLogger(__name__)


# The target branch still contains these operators in the centralized layout.
# Resolve them on first use so importing KeyPool does not JIT-compile unrelated
# operators or import an optional torchair dependency.
_LEGACY_EXPORTS = {
    "sparse_flash_mla_grad": (".sparse_flash_mla_grad", "sparse_flash_mla_grad"),
    "sparse_flash_mla_grad_metadata": (
        ".sparse_flash_mla_grad",
        "sparse_flash_mla_grad_metadata",
    ),
    "moe_finalize_routing": (".moe_finalize_routing", "moe_finalize_routing"),
    "moe_token_permute": (".moe_token_permute", "moe_token_permute"),
    "get_symm_buffer_for_mega_moe": (".mega_moe", "get_symm_buffer_for_mega_moe"),
    "mega_moe": (".mega_moe", "mega_moe"),
    "MoeDistributeBuffer": (".deep_ep", "MoeDistributeBuffer"),
    "flash_attn": (".flash_attn", "flash_attn"),
    "flash_attn_metadata": (".flash_attn", "flash_attn_metadata"),
    "quant_flash_attn": (".quant_flash_attn", "quant_flash_attn"),
    "quant_flash_attn_metadata": (
        ".quant_flash_attn",
        "quant_flash_attn_metadata",
    ),
    "mixed_quant_sparse_flash_mla": (
        ".mixed_quant_sparse_flash_mla",
        "mixed_quant_sparse_flash_mla",
    ),
    "mixed_quant_sparse_flash_mla_metadata": (
        ".mixed_quant_sparse_flash_mla",
        "mixed_quant_sparse_flash_mla_metadata",
    ),
    "quant_sparse_flash_mla": (
        ".quant_sparse_flash_mla",
        "quant_sparse_flash_mla",
    ),
    "quant_sparse_flash_mla_metadata": (
        ".quant_sparse_flash_mla",
        "quant_sparse_flash_mla_metadata",
    ),
    "scatter_pa_kv_cache_with_k_scale": (
        ".scatter_pa_kv_cache_with_k_scale",
        "scatter_pa_kv_cache_with_k_scale",
    ),
    "sparse_flash_mla": (".sparse_flash_mla", "sparse_flash_mla"),
    "sparse_flash_mla_metadata": (
        ".sparse_flash_mla",
        "sparse_flash_mla_metadata",
    ),
    "sparse_flash_mla_softmax_l1_norm": (
        ".sparse_flash_mla_softmax_l1_norm",
        "sparse_flash_mla_softmax_l1_norm",
    ),
    "sparse_flash_mla_softmax_l1_norm_metadata": (
        ".sparse_flash_mla_softmax_l1_norm",
        "sparse_flash_mla_softmax_l1_norm_metadata",
    ),
    "sparse_lightning_indexer_kl_loss_grad": (
        ".sparse_lightning_indexer_kl_loss_grad",
        "sparse_lightning_indexer_kl_loss_grad",
    ),
    "sparse_lightning_indexer_kl_loss_grad_metadata": (
        ".sparse_lightning_indexer_kl_loss_grad",
        "sparse_lightning_indexer_kl_loss_grad_metadata",
    ),
    "dense_lightning_indexer_kl_loss_grad": (
        ".dense_lightning_indexer_kl_loss_grad",
        "dense_lightning_indexer_kl_loss_grad",
    ),
    "dense_lightning_indexer_kl_loss_grad_metadata": (
        ".dense_lightning_indexer_kl_loss_grad",
        "dense_lightning_indexer_kl_loss_grad_metadata",
    ),
    "lightning_indexer": (".lightning_indexer", "lightning_indexer"),
    "lightning_indexer_metadata": (
        ".lightning_indexer",
        "lightning_indexer_metadata",
    ),
    "quant_lightning_indexer": (
        ".quant_lightning_indexer",
        "quant_lightning_indexer",
    ),
    "quant_lightning_indexer_metadata": (
        ".quant_lightning_indexer",
        "quant_lightning_indexer_metadata",
    ),
    "mhc_post": (".mhc_post", "mhc_post"),
    "mhc_pre_sinkhorn": (".mhc_pre_sinkhorn", "mhc_pre_sinkhorn"),
    "mhc_pre_sinkhorn_backward": (
        ".mhc_pre_sinkhorn_backward",
        "mhc_pre_sinkhorn_backward",
    ),
    "kv_compress_epilog": (".kv_compress_epilog", "kv_compress_epilog"),
    "indexer_quant_cache": (".indexer_quant_cache", "indexer_quant_cache"),
    "compressor": (".compressor", "compressor"),
    "quant_compressor": (".quant_compressor", "quant_compressor"),
    "inplace_partial_rotary_mul": (
        ".inplace_partial_rotary_mul",
        "inplace_partial_rotary_mul",
    ),
    "inplace_partial_rotary_mul_backward": (
        ".inplace_partial_rotary_mul_backward",
        "inplace_partial_rotary_mul_backward",
    ),
    "stem_oam_prep_varlen_q": (
        ".stem_oam_prep_varlen_q",
        "stem_oam_prep_varlen_q",
    ),
    "causal_conv1d_fn": (".causal_conv1d_fn", "causal_conv1d_fn"),
    "causal_conv1d_update": (
        ".causal_conv1d_update",
        "causal_conv1d_update",
    ),
    "stem_oam_prep_paged_kv": (
        ".stem_oam_prep_paged_kv",
        "stem_oam_prep_paged_kv",
    ),
    "qkv_rms_norm_rope_cache_with_k_scale": (
        ".qkv_rms_norm_rope_cache_with_k_scale",
        "qkv_rms_norm_rope_cache_with_k_scale",
    ),
    "qkv_rms_norm_rope_cache_with_k_scale_": (
        ".qkv_rms_norm_rope_cache_with_k_scale",
        "qkv_rms_norm_rope_cache_with_k_scale_",
    ),
    "ElasticBuffer": (".elastic_buffer", "ElasticBuffer"),
    "EPHandle": (".elastic_buffer", "EPHandle"),
    "grouped_matmul_activation_quant": (
        ".grouped_matmul_activation_quant",
        "grouped_matmul_activation_quant",
    ),
    "dense_lightning_indexer_softmax_lse": (
        ".dense_lightning_indexer_softmax_lse",
        "dense_lightning_indexer_softmax_lse",
    ),
    "dense_lightning_indexer_softmax_lse_metadata": (
        ".dense_lightning_indexer_softmax_lse",
        "dense_lightning_indexer_softmax_lse_metadata",
    ),
}


def _load_distributed_ops():
    """Load operators copied from <category>/<op>/torch_extension."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for category in sorted(os.listdir(current_dir)):
        if category.startswith((".", "_", "__")):
            continue
        category_dir = os.path.join(current_dir, category)
        if not os.path.isdir(category_dir):
            continue
        for op_name in sorted(os.listdir(category_dir)):
            if op_name.startswith((".", "_", "__")):
                continue
            op_dir = os.path.join(category_dir, op_name)
            if not os.path.isdir(op_dir) or not os.path.isfile(
                os.path.join(op_dir, "__init__.py")
            ):
                continue
            module_name = f"{__name__}.{category}.{op_name}"
            try:
                module = importlib.import_module(module_name)
                for exported_name in getattr(module, "__all__", ()):
                    if not exported_name.startswith("_"):
                        globals()[exported_name] = getattr(module, exported_name)
                # Preserve the pre-migration module path for existing callers
                # while the implementation lives under its category directory.
                legacy_module_name = f"{__name__}.{op_name}"
                if legacy_module_name not in sys.modules:
                    sys.modules[legacy_module_name] = module
            except (ImportError, RuntimeError, AttributeError) as exc:
                logger.warning(
                    "Failed to load distributed operator '%s': %s", op_name, exc
                )


def __getattr__(name):
    target = _LEGACY_EXPORTS.get(name)
    if target is None and name == "key_pool":
        target = (".attention.key_pool", "key_pool")
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LEGACY_EXPORTS) | {"key_pool"})


__all__ = sorted(set(_LEGACY_EXPORTS) | {"key_pool"})


_load_distributed_ops()
del _load_distributed_ops
