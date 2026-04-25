# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY

class KvQuantSparseFlashMlaOpBuilder(OpBuilder):
    def __init__(self):
        super(KvQuantSparseFlashMlaOpBuilder, self).__init__("npu_kv_quant_sparse_flash_mla")
    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/kv_quant_sparse_flash_mla.cpp']
    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_kv_quant_sparse_flash_mla(Tensor q, int kv_quant_mode, *," \
                                             "Tensor? ori_kv=None, Tensor? ori_sparse_indices=None, Tensor? ori_block_table=None, " \
                                             "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? seqused_q=None, " \
                                             "Tensor? seqused_kv=None, Tensor? metadata=None, int tile_size=128, int rope_head_dim=64, " \
                                             "float softmax_scale= 0.0416666666666667, int ori_mask_mode=3, str layout_query=\"BSND\", str layout_kv=\"BSND\", " \
                                             "bool return_softmax_lse=False, int? key_dtype=None, int? value_dtype=None) -> Tensor"

        # return "npu_kv_quant_sparse_flash_mla(Tensor q, Tensor key, Tensor value, Tensor sparse_indices, " \
        #        "float scale_value, int key_quant_mode, int value_quant_mode, *, Tensor? key_dequant_scale=None, " \
        #        "Tensor? value_dequant_scale=None, Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, " \
        #        "Tensor? actual_seq_lengths_kv=None, Tensor? metadata=None, int sparse_block_size=1, str layout_query=\"BSND\", " \
        #        "str layout_kv=\"BSND\", int sparse_mode=3, int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, " \
        #        "int attention_mode=0, int quant_scale_repo_mode=1, int tile_size=128, int rope_head_dim=64, int? key_dtype=None, " \
        #        "int? value_dtype=None) -> Tensor"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_kv_quant_sparse_flash_mla_meta(q, kv_quant_mode, ori_kv=None, ori_sparse_indices=None,
                                               ori_block_table=None,cu_seqlens_q=None, cu_seqlens_ori_kv=None, seqused_q=None, seqused_kv=None,
                                               metadata=None, tile_size=128, rope_head_dim=64, softmax_scale= 0.0416666666666667,
                                               ori_mask_mode=3, layout_query='BSND', layout_kv='BSND', return_softmax_lse=False,key_dtype=None,
                                               value_dtype=None):
            keyHeadNum = key.shape[1] if layout_key == "TND" else key.shape[2]
            if layout_query == "BSND":
                ## 添加softmax_lse
                attn_out = torch.empty(q.shape, dtype = q.dtype, device="meta")
                if return_softmax_lse:
                    softmax_lse = torch.empty([q.shape[0], ori_kv.shape[2],q.shape[1], q.shape[2] / ori_kv.shape[2]],
                                              dtype = torch.float32, device="meta")
            else:
                attn_out = torch.empty(q.shape, dtype = q.dtype, device="meta")
                if return_softmax_lse:
                    softmax_lse = torch.empty([ori_kv.shape[1], q.shape[0], q.shape[2] / ori_kv.shape[2]],
                                              dtype = torch.float32, device="meta")

            return (attn_out, softmax_lse)

# Instantiate the builder
npu_kv_quant_sparse_flash_mla_op_builder = KvQuantSparseFlashMlaOpBuilder()
op_module = npu_kv_quant_sparse_flash_mla.load()  # Compiles/loads the .so file

@impl(AS_LIBRARY, npu_kv_quant_sparse_flash_mla_op_builder.name, "PrivateUse1")
def npu_kv_quant_sparse_flash_mla_meta(q, kv_quant_mode, ori_kv=None, ori_sparse_indices=None,
                                       ori_block_table=None,cu_seqlens_q=None, cu_seqlens_ori_kv=None, seqused_q=None, seqused_kv=None,
                                       metadata=None, tile_size=128, rope_head_dim=64, softmax_scale= 0.0416666666666667,
                                       ori_mask_mode=3, layout_query='BSND', layout_kv='BSND', return_softmax_lse=False,key_dtype=None,
                                       value_dtype=None):
    """
    dispatcher implementation for NPU.zhe
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    return op_module.npu_kv_quant_sparse_flash_mla(q, kv_quant_mode, ori_kv=None, ori_sparse_indices=None,
                                                   ori_block_table=None,cu_seqlens_q=None, cu_seqlens_ori_kv=None, seqused_q=None, seqused_kv=None,
                                                   metadata=None, tile_size=128, rope_head_dim=64, softmax_scale= 0.0416666666666667,
                                                   ori_mask_mode=3, layout_query='BSND', layout_kv='BSND', return_softmax_lse=False,key_dtype=None,
                                                   value_dtype=None)