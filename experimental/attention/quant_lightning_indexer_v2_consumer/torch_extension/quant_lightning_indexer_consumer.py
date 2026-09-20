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
import os
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

QLI_METADATA_SIZE = 1024
QLI_METADATA_OP_NAME = "quant_lightning_indexer_metadata"


class QuantLightningIndexerV2ConsumerOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantLightningIndexerV2ConsumerOpBuilder, self).__init__(
            "quant_lightning_indexer_v2_consumer", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/quant_lightning_indexer_consumer.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            # P13 拆分: consumer 专属入口 (mode=2), candidate_topk_index 必传
            "quant_lightning_indexer_v2_consumer_candidate(Tensor query, Tensor key, Tensor weights, "
            "Tensor query_dequant_scale, Tensor key_dequant_scale, int topk, int quant_mode, *, "
            "Tensor candidate_topk_index, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, "
            "Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? cmp_residual_k=None, "
            "Tensor? block_table=None, Tensor? output_idx_offset=None, Tensor? metadata=None, "
            "int max_seqlen_q=-1, str layout_q=\"BSND\", str layout_k=\"BSND\", int mask_mode=0, "
            "int cmp_ratio=1, int candidate_topk_blocks=2048, "
            "int candidate_block_size=8) -> (Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(get_as_library(), self.name, "Meta")
        def quant_lightning_indexer_meta(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
            topk,
            quant_mode,
            *,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            block_table=None,
            output_idx_offset=None,
            metadata=None,
            max_seqlen_q=-1,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=0,
            cmp_ratio=1,
            return_value=0,
        ):
            key_head_num = key.shape[1] if layout_k == "TND" else key.shape[2]

            if layout_q == "BSND":
                sparse_indices_out = torch.empty(
                    [query.shape[0], query.shape[1], key_head_num, topk],
                    dtype=torch.int32,
                    device="meta",
                )
            else:
                sparse_indices_out = torch.empty(
                    [query.shape[0], key_head_num, topk],
                    dtype=torch.int32,
                    device="meta",
                )
            if return_value:
                if layout_q == "BSND":
                    sparse_values_out = torch.empty(
                        [query.shape[0], query.shape[1], key_head_num, topk],
                        dtype=torch.bfloat16,
                        device="meta",
                    )
                else:
                    sparse_values_out = torch.empty(
                        [query.shape[0], key_head_num, topk],
                        dtype=torch.bfloat16,
                        device="meta",
                    )
            else:
                sparse_values_out = torch.empty(
                    [0], dtype=torch.bfloat16, device="meta"
                )
            return (sparse_indices_out, sparse_values_out)

        @torch.library.register_fake("cann_ops_transformer::quant_lightning_indexer_v2_consumer_candidate")
        def quant_lightning_indexer_v2_consumer_candidate_meta(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
            topk,
            quant_mode,
            *,
            candidate_topk_index=None,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            block_table=None,
            output_idx_offset=None,
            metadata=None,
            max_seqlen_q=-1,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=0,
            cmp_ratio=1,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
        ):
            key_head_num = key.shape[1] if layout_k == "TND" else key.shape[2]
            if layout_q == "BSND":
                sparse_indices_out = torch.empty(
                    [query.shape[0], query.shape[1], key_head_num, topk],
                    dtype=torch.int32,
                    device="meta",
                )
            else:
                sparse_indices_out = torch.empty(
                    [query.shape[0], key_head_num, topk],
                    dtype=torch.int32,
                    device="meta",
                )
            sparse_values_out = torch.empty([0], dtype=torch.bfloat16, device="meta")
            return (sparse_indices_out, sparse_values_out)


# Instantiate the builder (schema def 幂等: 直接判 torch.ops, 避免 ops/__init__
# 自动发现链回滚重试导致的重复 define)
quant_lightning_indexer_v2_consumer_op_builder = QuantLightningIndexerV2ConsumerOpBuilder()
try:
    quant_lightning_indexer_v2_consumer_op_builder._ensure_initialized()
except RuntimeError:
    # define 报"已注册": schema 已在 torch Library 中 (此前导入尝试回滚后残留)。
    # _ensure_initialized 在 register_schema 处中断, 其余初始化 (路径属性) 需补齐,
    # 否则 load() 访问 _package_path 抛 AttributeError。
    _b = quant_lightning_indexer_v2_consumer_op_builder
    _b._initialized = True
    import torch_npu as _torch_npu
    _b._torch_npu_path = os.path.dirname(os.path.abspath(_torch_npu.__file__))
    _b._package_path = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def quant_lightning_indexer_v2_consumer_candidate(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    topk,
    quant_mode,
    *,
    candidate_topk_index=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    candidate_topk_blocks=2048,
    candidate_block_size=8,
):
    """consumer 专属入口 (mode=2): candidate_topk_index 必传, 候选块内选 topk"""
    op_module = quant_lightning_indexer_v2_consumer_op_builder.load()
    return op_module.quant_lightning_indexer_v2_consumer_candidate(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        topk,
        quant_mode,
        candidate_topk_index,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        metadata,
        max_seqlen_q,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
        candidate_topk_blocks,
        candidate_block_size,
    )


try:
    impl(get_as_library(), "quant_lightning_indexer_v2_consumer_candidate", "PrivateUse1")(
        quant_lightning_indexer_v2_consumer_candidate)
except RuntimeError:
    pass  # 重复 import (循环导入回滚重试) 时 schema 已注册
torch.compiler.allow_in_graph(quant_lightning_indexer_v2_consumer_candidate)




# =============================================================================================
# 新接口规范适配层 (P13 拆分): consumer 专属算子的 py 分发封装, 内部映射到
# quant_lightning_indexer_v2_consumer_candidate (mode=2 固定)。
# 调用侧按芯片选择接口 (本后端走本适配层, 新芯片走其自带实现)。
#
# 与内部接口的映射约定:
#   q_descale / k_descale     <-> query_dequant_scale / key_dequant_scale (仅改名)
#   candidate_block_indices   <-> candidate_topk_index (仅改名; 块级索引, 相对块号)
#   seqused_q (新名, 每 batch key 有效长度, 即内部 seqused_k)  <-> seqused_k
#   candidate_block_length    mode=1 输出: py 按行级有效长度公式生成 (mask 规则与 key 长度取小);
#                             mode=2 输入: 本后端忽略 (算子内部已按 mask 规则与 key 长度取小截断, 语义冗余)
#   candidate_topk_blocks=-1  (规范默认, "无 candidate 机制") 在 source/consumer 场景下无意义, 取 2048
#   layout 参数消失           q 恒 3 维 (T1, N1, D): 有 cu_seqlens_q 视为 TND 变长拼接, 否则 B=1 BSND
# =============================================================================================


def _qli_newapi_check(cond, msg):
    if not cond:
        raise RuntimeError("[quant_lightning_indexer new-api] " + msg)


def _qli_newapi_layout(cu_seqlens_q):
    # 新规范无 layout 参数: 由 cu_seqlens_q 有无推断 (有 = TND 变长拼接; 无 = B=1 BSND)
    return ("TND", True) if cu_seqlens_q is not None else ("BSND", False)


def _qli_newapi_build_metadata(q, k, layout_q, layout_k, cu_seqlens_q, seqused_k, cmp_residual_k,
                               topk, quant_mode, mask_mode, cmp_ratio):
    # 本后端主算子 metadata 必传; 新规范调用方不传时按 shape 自动推导。
    # 注: max_seqlen_q/k 需要标量, 此处 .item() 触发一次主机同步 (调用方可自行预生成 metadata 传入绕过)。
    num_heads_q = int(q.shape[1])  # N1 (q 恒 3 维)
    num_heads_k = int(k.shape[2])  # N2 (k 为 PA 物理池布局 (block_num, block_size, N2, D))
    head_dim = int(q.shape[2])
    if layout_q == "TND":
        batch_size = int(cu_seqlens_q.numel()) - 1
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    else:
        batch_size = 1
        max_seqlen_q = int(q.shape[0])
    max_seqlen_k = int(seqused_k.max().item())
    from cann_ops_transformer.ops.attention.quant_lightning_indexer_v2.quant_lightning_indexer import (
        quant_lightning_indexer_metadata as _main_metadata)
    return _main_metadata(
        num_heads_q, num_heads_k, head_dim, topk, quant_mode,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=None, seqused_q=None, seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k, batch_size=batch_size,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        layout_q=layout_q, layout_k=layout_k, mask_mode=mask_mode, cmp_ratio=cmp_ratio)


def _qli_newapi_block_length(q, k, cu_seqlens_q, seqused_k, cmp_residual_k, cmp_ratio, mask_mode):
    # source 第 4 输出 (T1, N2): 每行实际参与搜索的 key 长度 (mask 规则与 key 长度取小)。
    # mask_mode=3 (causal): vl(i) = clamp((act_k - S1 + i + 1) // cmp_ratio, 0, K),
    #   act_k = K * cmp_ratio + residual 为未压缩原始 key 长度, i 为 batch 内行号 (与 golden 同公式);
    # mask_mode=0 (无 mask): 恒为 K。全程张量化, 不引入主机同步。
    device = q.device
    t1 = int(q.shape[0])
    n2 = int(k.shape[2])
    s2_t = seqused_k.to(torch.int64)  # (B,) 每 batch key 有效长度
    res_t = cmp_residual_k.to(torch.int64) if cmp_residual_k is not None else torch.zeros_like(s2_t)
    act_k_t = s2_t * int(cmp_ratio) + res_t
    if cu_seqlens_q is not None:  # TND 变长拼接
        cu_q = cu_seqlens_q.to(torch.int64)
        row_b = torch.repeat_interleave(torch.arange(s2_t.numel(), device=device), cu_q[1:] - cu_q[:-1])
        i_in_b = torch.arange(t1, device=device) - cu_q[row_b]  # batch 内行号
        s1_row = (cu_q[1:] - cu_q[:-1])[row_b]  # 所在 batch 的 query 行数
    else:  # B=1 BSND
        row_b = torch.zeros(t1, dtype=torch.int64, device=device)
        i_in_b = torch.arange(t1, device=device)
        s1_row = torch.full((t1,), t1, dtype=torch.int64, device=device)
    if int(mask_mode) == 3:
        vl = (act_k_t[row_b] - s1_row + i_in_b + 1) // int(cmp_ratio)
        vl = torch.clamp(vl, min=0)  # 全无效行 (vl < 0) 记 0
        vl = torch.minimum(vl, s2_t[row_b])  # 与 key 长度取小
    else:
        vl = s2_t[row_b]
    return vl.to(torch.int32).unsqueeze(-1).expand(t1, n2).contiguous()


def quant_lightning_indexer_v2_consumer(q, k, w, q_descale, k_descale, candidate_block_indices,
                                        candidate_block_length, topk, quant_mode, *,
    cu_seqlens_q=None, seqused_q=None, cmp_residual_k=None, block_table=None,
    output_idx_offset=None, metadata=None, max_seqlen_q=-1, mask_mode=0, cmp_ratio=1,
    return_value=False, candidate_block_size=8):
    """新规范接口2 (consumer, 内部 candidate_mode=2): 在候选块内选 topk。

    candidate_block_indices: (T1, N2, cb) source 侧输出的候选块索引 (块级, 相对块号);
    candidate_block_length: 本后端忽略 (算子内部已按 mask 规则与 key 长度取小截断, 该输入语义冗余);
    candidate_topk_blocks 由 candidate_block_indices.shape[-1] 推导 (新规范无此属性)。
    返回 (sparse_indices, sparse_values); 布局与内部一致: B=1 BSND (B,S1,N2,topk) / TND (T1,N2,topk)。
    """
    _qli_newapi_check(block_table is not None,
                      "block_table is required on this backend (key only supports PA_BBND paged layout).")
    _qli_newapi_check(not return_value, "return_value=True is not supported on this backend.")
    _qli_newapi_check(candidate_block_indices is not None,
                      "candidate_block_indices is required (consumer input).")
    _qli_newapi_check(candidate_block_indices.dim() == 3,
                      "candidate_block_indices must be 3-D (T1, N2, candidate_topk_blocks).")
    _qli_newapi_check(seqused_q is not None,
                      "seqused_q (per-batch key valid length) is required: varlen dispatch depends on it.")
    _qli_newapi_check(cu_seqlens_q is not None or int(seqused_q.numel()) == 1,
                      "batch > 1 requires cu_seqlens_q (TND varlen layout); without it q is treated "
                      "as B=1 BSND and the batch dimension would be lost.")
    topk_blocks = int(candidate_block_indices.shape[-1])
    layout_q, is_tnd = _qli_newapi_layout(cu_seqlens_q)
    if is_tnd:
        q_in, w_in, q_descale_in, cand_in = q, w, q_descale, candidate_block_indices
    else:
        _qli_newapi_check(q.dim() == 3, "q must be 3-D (T1, N1, D).")
        q_in, w_in, q_descale_in = q.unsqueeze(0), w.unsqueeze(0), q_descale.unsqueeze(0)
        cand_in = candidate_block_indices.unsqueeze(0)  # (1, T1, N2, cb)
    if metadata is None:
        metadata = _qli_newapi_build_metadata(q, k, layout_q, "PA_BBND", cu_seqlens_q, seqused_q,
                                              cmp_residual_k, topk, quant_mode, mask_mode, cmp_ratio)
    idx, vals = quant_lightning_indexer_v2_consumer_candidate(
        q_in, k, w_in, q_descale_in, k_descale, topk, quant_mode,
        candidate_topk_index=cand_in,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=None, seqused_q=None, seqused_k=seqused_q,
        cmp_residual_k=cmp_residual_k, block_table=block_table, output_idx_offset=output_idx_offset,
        metadata=metadata, max_seqlen_q=max_seqlen_q, layout_q=layout_q, layout_k="PA_BBND",
        mask_mode=mask_mode, cmp_ratio=cmp_ratio,
        candidate_topk_blocks=topk_blocks, candidate_block_size=int(candidate_block_size))
    return idx, vals
