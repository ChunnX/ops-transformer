# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional

import torch
import torch_npu
from torch.library import impl
from npu_ops_transformer.op_builder.builder import AS_LIBRARY
from npu_ops_transformer.op_builder.builder import OpBuilder


SMLA_META_SIZE = 1024


def _infer_batch_size(batch_size, layout_q, cu_seqlens_q, seqused_q):
    if batch_size is not None:
        return batch_size
    if seqused_q is not None:
        return seqused_q.size(0)
    if layout_q == "TND" and cu_seqlens_q is not None:
        return cu_seqlens_q.size(0) - 1
    return 0


class SparseFlashMlaMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super(SparseFlashMlaMetadataOpBuilder, self).__init__("npu_sparse_flash_mla_metadata")

    def sources(self):
        return ["ops/csrc/sparse_flash_mla_metadata.cpp"]

    def schema(self) -> str:
        return "npu_sparse_flash_mla_metadata(int num_heads_q, int num_heads_kv, int head_dim, *, " \
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, " \
            "Tensor? seqused_q=None, Tensor? seqused_ori_kv=None, " \
            "int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_kv=None, " \
            "int ori_topk=0, int cmp_topk=0, int cmp_ratio=-1, " \
            "int ori_mask_mode=4, int cmp_mask_mode=3, int ori_win_left=127, int ori_win_right=0, " \
            "str layout_q=\"BSND\", str layout_kv=\"PA_ND\", " \
            "bool has_ori_kv=True, bool has_cmp_kv=True) -> Tensor"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_sparse_flash_mla_metadata_meta(num_heads_q, num_heads_kv, head_dim, *,
                                               cu_seqlens_q=None, cu_seqlens_ori_kv=None,
                                               cu_seqlens_cmp_kv=None, seqused_q=None, seqused_ori_kv=None,
                                               batch_size: Optional[int] = None,
                                               max_seqlen_q: Optional[int] = None,
                                               max_seqlen_kv: Optional[int] = None,
                                               ori_topk=0, cmp_topk=0, cmp_ratio=-1,
                                               ori_mask_mode=4, cmp_mask_mode=3,
                                               ori_win_left=127, ori_win_right=0,
                                               layout_q="BSND", layout_kv="PA_ND",
                                               has_ori_kv=True, has_cmp_kv=True):
            return torch.empty((SMLA_META_SIZE,), dtype=torch.int32, device="meta")


sparse_flash_mla_metadata_op_builder = SparseFlashMlaMetadataOpBuilder()
op_module = sparse_flash_mla_metadata_op_builder.load()


@impl(AS_LIBRARY, sparse_flash_mla_metadata_op_builder.name, "PrivateUse1")
def npu_sparse_flash_mla_metadata(num_heads_q, num_heads_kv, head_dim, *,
                                  cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None,
                                  seqused_q=None, seqused_ori_kv=None,
                                  batch_size: Optional[int] = None,
                                  max_seqlen_q: Optional[int] = None,
                                  max_seqlen_kv: Optional[int] = None,
                                  ori_topk=0, cmp_topk=0, cmp_ratio=-1,
                                  ori_mask_mode=4, cmp_mask_mode=3,
                                  ori_win_left=127, ori_win_right=0,
                                  layout_q="BSND", layout_kv="PA_ND",
                                  has_ori_kv=True, has_cmp_kv=True):
    batch_size = _infer_batch_size(batch_size, layout_q, cu_seqlens_q, seqused_q)
    max_seqlen_q = 0 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = 0 if max_seqlen_kv is None else max_seqlen_kv
    metadata = torch.empty((SMLA_META_SIZE,), dtype=torch.int32, device="npu")

    return op_module.npu_sparse_flash_mla_metadata(
        cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv,
        num_heads_q, num_heads_kv, head_dim, batch_size, max_seqlen_q, max_seqlen_kv,
        ori_topk, cmp_topk, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
        layout_q, layout_kv, has_ori_kv, has_cmp_kv, metadata)
