#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
import os
import sys

import torch
import torch_npu


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
QBSA_ROOT = os.path.dirname(CURRENT_DIR)
TORCH_OPS_EXTENSION_DIR = os.path.join(QBSA_ROOT, "torch_ops_extension")


def _import_custom_ops():
    # build_and_install.sh 安装的是 wheel，custom_ops_lib*.so 位于 site-packages/custom_ops 下；
    # 如果这里先把源码目录插到 sys.path 最前面，Python 会优先加载源码树里的 custom_ops，
    # 但源码树通常没有 in-place 生成的 custom_ops_lib*.so，反而会遮住已经正确安装的 wheel。
    # 因此优先尝试已安装包；只有没安装时，才回退到源码目录，支持 `setup.py build_ext --inplace` 场景。
    try:
        import custom_ops as custom_ops_module  # noqa: F401

        return custom_ops_module
    except ImportError as install_import_error:
        sys.modules.pop("custom_ops", None)
        if TORCH_OPS_EXTENSION_DIR not in sys.path:
            sys.path.insert(0, TORCH_OPS_EXTENSION_DIR)
        try:
            import custom_ops as custom_ops_module  # noqa: F401

            return custom_ops_module
        except ImportError as source_import_error:
            raise RuntimeError(
                "import custom_ops failed. Please build the QBSA PTA extension first, for example: "
                "`cd experimental/attention/quant_block_sparse_attn/torch_ops_extension && "
                "bash build_and_install.sh`. "
                f"Installed-package import error: {install_import_error}. "
                f"Source-tree import error: {source_import_error}."
            ) from source_import_error


try:
    custom_ops = _import_custom_ops()  # noqa: F401  # 注册 torch.ops.custom.npu_quant_block_sparse_attn
except RuntimeError as import_error:
    raise RuntimeError(
        "import custom_ops failed. Please confirm the Python used to run this example is the same one used to "
        f"install the PTA wheel. Detail: {import_error}"
    ) from import_error


def _get_dtype(dtype_name):
    if hasattr(torch, dtype_name):
        return getattr(torch, dtype_name)
    if hasattr(torch_npu, dtype_name):
        return getattr(torch_npu, dtype_name)
    raise RuntimeError(f"{dtype_name} is not available in current torch/torch_npu runtime.")


def _ones_npu(shape, dtype):
    return torch.ones(shape, dtype=torch.float32).npu().to(dtype)


def _e8m0_ones_npu(shape, dtype):
    # fp8_e8m0 在 torch/torch_npu 侧通常不支持 CPU 数值构造或普通 cast。
    # MX scale 本质是 1-byte 指数编码；1.0 的 e8m0 编码为 0x7f。
    # 这里先在 NPU 上构造 uint8 原始字节，再用 dtype view 改成 fp8_e8m0，
    # 避免走 `float32 -> fp8_e8m0` 的 cast kernel。
    raw = torch.full(shape, 0x7F, dtype=torch.uint8).npu()
    return raw.view(dtype)


def _check_custom_op_registered():
    if not hasattr(torch.ops, "custom") or not hasattr(torch.ops.custom, "npu_quant_block_sparse_attn"):
        raise RuntimeError("torch.ops.custom.npu_quant_block_sparse_attn is not registered.")
    if not hasattr(torch.ops.custom, "npu_quant_block_sparse_attn_metadata"):
        raise RuntimeError("torch.ops.custom.npu_quant_block_sparse_attn_metadata is not registered.")


def main():
    torch.manual_seed(42)
    torch_npu.npu.set_device(0)
    _check_custom_op_registered()

    fp8_e4m3 = _get_dtype("float8_e4m3fn")
    fp8_e8m0 = _get_dtype("float8_e8m0fnu")

    # MXFP8 全量化基础 case：
    #   - Q: TND [Tq=128, Nq=1, D=128]
    #   - K/V: PA BNBD [blockNum=4, Nkv=1, blockSize=128, D=128]
    #   - sparse_indices + block_table 将 4 个 128-token PA block 拼成 512-token S2。
    batch_size = 1
    num_heads_q = 1
    num_heads_kv = 1
    q_token_num = 128
    kv_token_num = 512
    head_dim = 128
    value_dim = 128
    sparse_q_block_size = 128
    sparse_kv_block_size = 128
    pa_block_num = kv_token_num // sparse_kv_block_size
    q_block_num = q_token_num // sparse_q_block_size
    kv_block_num = kv_token_num // sparse_kv_block_size

    scale_group_size = 64
    scale_last_dim = 2
    qk_scale_d_size = head_dim // scale_group_size
    value_scale_block_size = kv_token_num // scale_group_size
    softmax_scale = 1.0 / math.sqrt(float(head_dim))

    query = _ones_npu((q_token_num, num_heads_q, head_dim), fp8_e4m3)
    key = _ones_npu((pa_block_num, num_heads_kv, sparse_kv_block_size, head_dim), fp8_e4m3)
    value = _ones_npu((pa_block_num, num_heads_kv, sparse_kv_block_size, value_dim), fp8_e4m3)

    q_descale = _e8m0_ones_npu((q_token_num, num_heads_q, qk_scale_d_size, scale_last_dim), fp8_e8m0)
    k_descale = _e8m0_ones_npu((kv_token_num, num_heads_kv, qk_scale_d_size, scale_last_dim), fp8_e8m0)
    v_descale = _e8m0_ones_npu((value_scale_block_size, num_heads_kv, value_dim, scale_last_dim), fp8_e8m0)
    p_scale = _e8m0_ones_npu((1,), fp8_e8m0)

    cu_seqlens_q = torch.tensor([0, q_token_num], dtype=torch.int32).npu()
    cu_seqlens_kv = torch.tensor([0, kv_token_num], dtype=torch.int32).npu()
    seqused_q = torch.tensor([q_token_num], dtype=torch.int32).npu()
    seqused_kv = torch.tensor([kv_token_num], dtype=torch.int32).npu()

    sparse_seq_len = torch.full((batch_size, num_heads_q, q_block_num), kv_block_num, dtype=torch.int32).npu()
    sparse_indices = torch.arange(kv_block_num, dtype=torch.int32).reshape(
        batch_size, num_heads_q, q_block_num, kv_block_num).npu()
    block_table = torch.arange(pa_block_num, dtype=torch.int32).reshape(batch_size, pa_block_num).npu()

    # 当前 MXFP8 入口要求 mask_mode=3。全一 uint8 mask 作为最小 smoke case，
    # 若需要严格因果语义，可替换为 tril/triu 形式。
    atten_mask = torch.triu(torch.ones((2048, 2048), dtype=torch.uint8)).npu()

    metadata = torch.ops.custom.npu_quant_block_sparse_attn_metadata(
        sparse_seq_len,
        num_heads_q,
        num_heads_kv,
        head_dim,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        sparse_block_size_q=sparse_q_block_size,
        sparse_block_size_k=sparse_kv_block_size,
        quant_mode=2,
        mask_mode=3,
        max_seqlen_q=q_token_num,
        max_seqlen_kv=kv_token_num,
        layout_q="TND",
        layout_kv="PA_BNSD",
        layout_sparse_indices="B_N_Qb_Kb",
    )
    torch_npu.npu.synchronize()

    attention_out, softmax_lse = torch.ops.custom.npu_quant_block_sparse_attn(
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        atten_mask,
        softmax_scale,
        sparse_q_block_size,
        sparse_kv_block_size,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        block_table=block_table,
        metadata=metadata,
        max_seqlen_q=q_token_num,
        max_seqlen_kv=kv_token_num,
        layout_kv="PA_BNSD",
        layout_q="TND",
        layout_sparse_indices="B_N_Qb_Kb",
        layout_out="TND",
        quant_mode=2,
        mask_mode=3,
        return_softmax_lse=True,
    )
    torch_npu.npu.synchronize()

    print("************ QBSA MXFP8 example end *************")
    print("attention_out:", attention_out.cpu(), attention_out.shape, attention_out.dtype)
    print("softmax_lse:", softmax_lse.cpu(), softmax_lse.shape, softmax_lse.dtype)


if __name__ == "__main__":
    main()
