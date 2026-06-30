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

_ATTR_TO_MODULE = {
    'sparse_flash_mla_grad':                      'sparse_flash_mla_grad',
    'sparse_flash_mla_grad_metadata':             'sparse_flash_mla_grad',
    'npu_moe_distribute_dispatch':                'moe_distribute_dispatch',
    'npu_moe_distribute_combine':                 'moe_distribute_combine',
    'get_symm_buffer_for_mega_moe':               'mega_moe',
    'mega_moe':                                   'mega_moe',
    'npu_get_mega_moe_ccl_buffer_size':           'mega_moe',
    'MoeDistributeBuffer':                        'deep_ep',
    'flash_attn':                                 'flash_attn',
    'flash_attn_metadata':                        'flash_attn',
    'mixed_quant_sparse_flash_mla':               'mixed_quant_sparse_flash_mla',
    'mixed_quant_sparse_flash_mla_metadata':      'mixed_quant_sparse_flash_mla',
    'scatter_pa_kv_cache_with_k_scale':           'scatter_pa_kv_cache_with_k_scale',
    'sparse_flash_mla':                           'sparse_flash_mla',
    'sparse_flash_mla_metadata':                  'sparse_flash_mla',
    'sparse_lightning_indexer_kl_loss_grad':      'sparse_lightning_indexer_kl_loss_grad',
    'sparse_lightning_indexer_kl_loss_grad_metadata': 'sparse_lightning_indexer_kl_loss_grad',
    'lightning_indexer':                          'lightning_indexer',
    'lightning_indexer_metadata':                 'lightning_indexer',
    'quant_lightning_indexer':                    'quant_lightning_indexer',
    'quant_lightning_indexer_metadata':           'quant_lightning_indexer',
    'mhc_post':                                   'mhc_post',
    'mhc_pre_sinkhorn':                           'mhc_pre_sinkhorn',
    'kv_compress_epilog':                         'kv_compress_epilog',
    'indexer_quant_cache':                        'indexer_quant_cache',
    'compressor':                                 'compressor',
    'inplace_partial_rotary_mul':                 'inplace_partial_rotary_mul',
    'causal_conv1d_fn':                           'causal_conv1d_fn',
    'causal_conv1d_update':                       'causal_conv1d_update',
    'ElasticBuffer':                              'elastic_buffer',
}


def __getattr__(name):
    if name in _ATTR_TO_MODULE:
        mod = importlib.import_module(f'.{_ATTR_TO_MODULE[name]}', __package__)
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr
    if name == '_graph_convert':
        from . import graph_convert as _mod
        globals()['_graph_convert'] = _mod
        return _mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
