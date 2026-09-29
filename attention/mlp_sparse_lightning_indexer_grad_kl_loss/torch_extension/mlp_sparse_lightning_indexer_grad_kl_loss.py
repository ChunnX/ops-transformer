# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").

from typing import List, Optional
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from torch.library import impl

from cann_ops_transformer.op_builder.builder import get_as_library
from cann_ops_transformer.op_builder.builder import OpBuilder


class MlpSparseLightningIndexerGradKLLossOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("mlp_sparse_lightning_indexer_grad_kl_loss")

    def sources(self):
        local = Path(__file__).parent / "csrc/mlp_sparse_lightning_indexer_grad_kl_loss.cpp"
        return [str(local) if local.is_file() else "csrc/attention/mlp_sparse_lightning_indexer_grad_kl_loss.cpp"]

    def _custom_opp_paths(self):
        # This adapter uses aclnn_common's runtime symbol lookup, not generated
        # per-OPP headers or link libraries. Keep JIT flags invariant across A/B
        # packages and avoid unrelated global vendor include/library paths.
        # Runtime ASCEND_CUSTOM_OPP_PATH still selects the freshly built ACLNN.
        return []

    def schema(self):
        return (
            "mlp_sparse_lightning_indexer_grad_kl_loss("
            "Tensor query, Tensor key, Tensor query_index, Tensor key_index, Tensor weight, "
            "Tensor sparse_indices, Tensor softmax_max, Tensor softmax_sum, "
            "Tensor? query_rope=None, Tensor? key_rope=None, Tensor? mask=None, "
            "int[]? cur_seq_lengths_query=None, int[]? cur_seq_lengths_key=None, *, "
            "float scale_value=1.0, str layout=\"TND\", int sparse_mode=3, "
            "int pre_tokens=2147483647, int next_tokens=2147483647, int block_size=1, "
            "bool deterministic=True, int valid_token_num=-1) -> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def meta(
            query,
            key,
            query_index,
            key_index,
            weight,
            sparse_indices,
            softmax_max,
            softmax_sum,
            query_rope=None,
            key_rope=None,
            mask=None,
            cur_seq_lengths_query=None,
            cur_seq_lengths_key=None,
            *,
            scale_value=1.0,
            layout="TND",
            sparse_mode=3,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            block_size=1,
            deterministic=True,
            valid_token_num=-1,
        ):
            del query, key, sparse_indices, softmax_max, softmax_sum
            del query_rope, key_rope, mask, cur_seq_lengths_query, cur_seq_lengths_key
            del scale_value, layout, sparse_mode, pre_tokens, next_tokens, block_size
            del deterministic, valid_token_num
            return (
                torch.empty_like(query_index, device="meta"),
                torch.empty_like(key_index, device="meta"),
                torch.empty_like(weight, dtype=torch.float32, device="meta"),
                torch.empty((1,), dtype=torch.float32, device="meta"),
            )


_builder = MlpSparseLightningIndexerGradKLLossOpBuilder()
_builder.ensure_initialized()


@impl(get_as_library(), _builder.name, "PrivateUse1")
def mlp_sparse_lightning_indexer_grad_kl_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weight: torch.Tensor,
    sparse_indices: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    query_rope: Optional[torch.Tensor] = None,
    key_rope: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    cur_seq_lengths_query: Optional[List[int]] = None,
    cur_seq_lengths_key: Optional[List[int]] = None,
    *,
    scale_value: float = 1.0,
    layout: str = "TND",
    sparse_mode: int = 3,
    pre_tokens: int = 2147483647,
    next_tokens: int = 2147483647,
    block_size: int = 1,
    deterministic: bool = True,
    valid_token_num: int = -1,
):
    module = _builder.load()
    return module.mlp_sparse_lightning_indexer_grad_kl_loss(
        query,
        key,
        query_index,
        key_index,
        weight,
        sparse_indices,
        softmax_max,
        softmax_sum,
        query_rope,
        key_rope,
        mask,
        cur_seq_lengths_query,
        cur_seq_lengths_key,
        scale_value,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
        block_size,
        deterministic,
        valid_token_num,
    )


torch.compiler.allow_in_graph(mlp_sparse_lightning_indexer_grad_kl_loss)
