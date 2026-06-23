# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").

import torch
import torch_npu
from torch.library import impl

from cann_ops_transformer.op_builder.builder import AS_LIBRARY
from cann_ops_transformer.op_builder.builder import OpBuilder


class SparseAttentionScoreOpBuilder(OpBuilder):
    def __init__(self):
        super(SparseAttentionScoreOpBuilder, self).__init__("npu_sparse_attention_score")

    def sources(self):
        return ["ops/csrc/sparse_attention_score.cpp"]

    def schema(self) -> str:
        return (
            "npu_sparse_attention_score("
            "Tensor query, Tensor key, Tensor value, Tensor select_idx, "
            "Tensor block_table, *, "
            "Tensor? select_num_idx=None, "
            "Tensor? q_dequant_scale=None, Tensor? k_dequant_scale=None, "
            "Tensor? v_dequant_scale=None, "
            "Tensor? actual_seq_lengths=None, Tensor? actual_seq_lengths_kv=None, "
            "str q_input_layout=\"TND\", str kv_input_layout=\"BNSD\", "
            "int num_key_value_heads=1, float scale_value=1.0, "
            "int block_size=128, int top_k=16, int inner_precise=0) -> Tensor"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_sparse_attention_score_meta(
            query,
            key,
            value,
            select_idx,
            block_table,
            *,
            select_num_idx=None,
            q_dequant_scale=None,
            k_dequant_scale=None,
            v_dequant_scale=None,
            actual_seq_lengths=None,
            actual_seq_lengths_kv=None,
            q_input_layout="TND",
            kv_input_layout="BNSD",
            num_key_value_heads=1,
            scale_value=1.0,
            block_size=128,
            top_k=16,
            inner_precise=0,
        ):
            torch._check(
                q_input_layout == "TND",
                lambda: "npu_sparse_attention_score only supports query TND layout",
            )
            out_dtype = torch.float16 if query.dtype == torch.float8_e4m3fn else query.dtype
            return torch.empty(query.shape, dtype=out_dtype, device="meta")


if getattr(torch_npu, "_ops_transformer_sparse_attention_score_registered", False):
    npu_sparse_attention_score = torch_npu.npu_sparse_attention_score
else:
    sparse_attention_score_op_builder = SparseAttentionScoreOpBuilder()

    def _load_sparse_attention_score_module():
        return sparse_attention_score_op_builder.load()

    @impl(AS_LIBRARY, sparse_attention_score_op_builder.name, "PrivateUse1")
    def npu_sparse_attention_score(
        query,
        key,
        value,
        select_idx,
        block_table,
        *,
        select_num_idx=None,
        q_dequant_scale=None,
        k_dequant_scale=None,
        v_dequant_scale=None,
        actual_seq_lengths=None,
        actual_seq_lengths_kv=None,
        q_input_layout="TND",
        kv_input_layout="BNSD",
        num_key_value_heads=1,
        scale_value=1.0,
        block_size=128,
        top_k=16,
        inner_precise=0,
    ):
        op_module = _load_sparse_attention_score_module()
        return op_module.npu_sparse_attention_score(
            query,
            key,
            value,
            select_idx,
            block_table,
            select_num_idx,
            q_dequant_scale,
            k_dequant_scale,
            v_dequant_scale,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            q_input_layout,
            kv_input_layout,
            num_key_value_heads,
            scale_value,
            block_size,
            top_k,
            inner_precise,
        )

    torch_npu.npu_sparse_attention_score = npu_sparse_attention_score
    torch_npu._ops_transformer_sparse_attention_score_registered = True
