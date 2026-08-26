# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

SFA_TOKENS_DEFAULT = 9223372036854775807
SFA_VALUE_NONE_MIN_CANN_VERSION = "9.2.0"


def is_support_sfa_value_none():
    """value=None is only supported when CANN >= 9.2.0.

    Mirrors op-plugin's meta check (get_cann_version() >= "9.2.0") but keeps a
    boolean API so both the Meta impl and the PrivateUse1 backend share one gate.
    """
    try:
        import torch_npu

        return bool(torch_npu._C._is_gte_cann_version(SFA_VALUE_NONE_MIN_CANN_VERSION, "CANN"))
    except Exception:
        # torch_npu missing / version API unavailable: fail safe and disallow.
        return False


class SparseFlashAttentionOpBuilder(OpBuilder):
    def __init__(self):
        super(SparseFlashAttentionOpBuilder, self).__init__(
            "sparse_flash_attention", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/sparse_flash_attention.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            "sparse_flash_attention(Tensor query, Tensor key, Tensor? value, Tensor sparse_indices, "
            "float scale_value, *, "
            "Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, "
            "Tensor? actual_seq_lengths_kv=None, Tensor? query_rope=None, Tensor? key_rope=None, "
            'Tensor? sinks=None, int sparse_block_size=1, str layout_query="BSND", '
            'str layout_kv="BSND", int sparse_mode=3, int pre_tokens=9223372036854775807, '
            "int next_tokens=9223372036854775807, int attention_mode=2, "
            "bool return_softmax_lse=False) -> (Tensor, Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(get_as_library(), self.name, "Meta")
        def sparse_flash_attention_meta(
            query: torch.Tensor,
            key: torch.Tensor,
            value: Optional[torch.Tensor],
            sparse_indices: torch.Tensor,
            scale_value: float,
            *,
            block_table: Optional[torch.Tensor] = None,
            actual_seq_lengths_query: Optional[torch.Tensor] = None,
            actual_seq_lengths_kv: Optional[torch.Tensor] = None,
            query_rope: Optional[torch.Tensor] = None,
            key_rope: Optional[torch.Tensor] = None,
            sinks: Optional[torch.Tensor] = None,
            sparse_block_size: int = 1,
            layout_query: str = "BSND",
            layout_kv: str = "BSND",
            sparse_mode: int = 3,
            pre_tokens: int = SFA_TOKENS_DEFAULT,
            next_tokens: int = SFA_TOKENS_DEFAULT,
            attention_mode: int = 2,
            return_softmax_lse: bool = False,
        ):
            require_param = {
                "query": query,
                "key": key,
                "sparse_indices": sparse_indices,
            }
            for item_name, item in require_param.items():
                if item is None:
                    raise ValueError(
                        f"{item_name} should not be None, but the actual value is None."
                    )
                if item.numel() == 0:
                    raise ValueError(f"Input {item_name} should not be empty.")
            if value is None:
                if not is_support_sfa_value_none():
                    raise NotImplementedError(
                        "npu_sparse_flash_attention: value=None is only supported with CANN >= 9.2.0."
                    )
            elif value.numel() == 0:
                raise ValueError(f"Input value should not be empty.")
            require_layout = {"TND": 3, "BSND": 4}
            if layout_query not in require_layout:
                raise ValueError(f"Not support layout of query: {layout_query}.")
            for layout_name, layout_dim in require_layout.items():
                if layout_query == layout_name:
                    if query.dim() != layout_dim:
                        raise ValueError(
                            f"When the layout of query is {layout_name}, the query dimension must be {layout_dim}, but got {query.dim()}."
                        )
                    if layout_kv == layout_name:
                        if key.dim() != layout_dim:
                            raise ValueError(
                                f"When the layout of key is {layout_name}, the key dimension must be {layout_dim}, but got {key.dim()}."
                            )
                    elif layout_kv == "PA_BSND":
                        if key.dim() != 4:
                            raise ValueError(
                                f"When the layout of key is PA_BSND, the key dimension must be 4, but got {key.dim()}."
                            )
                    else:
                        raise ValueError(
                            f"When layoutKV is not PA_BSND, layout_kv and layout_query must be the same, but found query: {layout_query}, key: {layout_kv}."
                        )

            key_head_num = key.shape[-2]
            attention_out = torch.empty(query.shape, dtype=query.dtype, device="meta")
            if return_softmax_lse:
                if layout_query == "BSND":
                    lse_shape = [
                        query.shape[0],
                        key_head_num,
                        query.shape[1],
                        query.shape[2] // key_head_num,
                    ]
                else:
                    lse_shape = [
                        key_head_num,
                        query.shape[0],
                        query.shape[1] // key_head_num,
                    ]
                softmax_max = torch.empty(lse_shape, dtype=torch.float32, device="meta")
                softmax_sum = torch.empty(lse_shape, dtype=torch.float32, device="meta")
            else:
                softmax_max = torch.empty([0], dtype=torch.float32, device="meta")
                softmax_sum = torch.empty([0], dtype=torch.float32, device="meta")
            return (attention_out, softmax_max, softmax_sum)


# Instantiate the builder
sparse_flash_attention_op_builder = SparseFlashAttentionOpBuilder()
sparse_flash_attention_op_builder._ensure_initialized()


@impl(get_as_library(), sparse_flash_attention_op_builder.name, "PrivateUse1")
def sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: Optional[torch.Tensor],
    sparse_indices: torch.Tensor,
    scale_value: float,
    *,
    block_table: Optional[torch.Tensor] = None,
    actual_seq_lengths_query: Optional[torch.Tensor] = None,
    actual_seq_lengths_kv: Optional[torch.Tensor] = None,
    query_rope: Optional[torch.Tensor] = None,
    key_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    sparse_block_size: int = 1,
    layout_query: str = "BSND",
    layout_kv: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = SFA_TOKENS_DEFAULT,
    next_tokens: int = SFA_TOKENS_DEFAULT,
    attention_mode: int = 2,
    return_softmax_lse: bool = False,
):
    require_param = {
        "query": query,
        "key": key,
        "sparse_indices": sparse_indices,
    }
    for item_name, item in require_param.items():
        if item is None:
            raise ValueError(
                f"{item_name} should not be None, but the actual value is None."
            )
    op_module = sparse_flash_attention_op_builder.load()
    return op_module.sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        sinks,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
        pre_tokens,
        next_tokens,
        attention_mode,
        return_softmax_lse,
    )
