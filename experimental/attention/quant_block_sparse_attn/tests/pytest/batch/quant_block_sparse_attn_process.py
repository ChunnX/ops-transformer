#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os

import pytest
import torch
import torchair
from torchair.configs.compiler_config import CompilerConfig

try:
    import torch_npu
except ImportError as import_error:  # pragma: no cover - depends on NPU environment
    torch_npu = None
    TORCH_NPU_IMPORT_ERROR = import_error
else:
    TORCH_NPU_IMPORT_ERROR = None

import custom_ops
import combined_kv_cache


QBSA_METADATA_OUTPUT_SIZE = 2048
QBSA_HEAD_METADATA_SIZE = 8
QBSA_AIC_CORE_NUM = 36
QBSA_AIV_CORE_NUM = 72
QBSA_METADATA_SIZE = 8
QBSA_FD_METADATA_SIZE = 8
QBSA_CORE_ENABLE_INDEX = 0
QBSA_BN1_START_INDEX = 1
QBSA_S1_START_INDEX = 2
QBSA_BN1_END_INDEX = 4
QBSA_S1_END_INDEX = 5
PRINT_METADATA_AIC_BASIC_BLOCKS = os.environ.get(
    "BSA_PRINT_METADATA_AIC_BASIC_BLOCKS", ""
).upper() in ("1", "TRUE", "YES")


def _to_npu(tensor):
    if tensor is None:
        return None
    return tensor.npu()


def _to_cpu(tensor):
    if tensor is None:
        return None
    return tensor.detach().cpu()


def _check_runtime_available():
    if torch_npu is None:
        pytest.skip(f"torch_npu is not available: {TORCH_NPU_IMPORT_ERROR}")
    if not custom_ops.has_quant_block_sparse_attn_op():
        detail = "; ".join(custom_ops.load_errors()[:3])
        suffix = f" Load errors: {detail}" if detail else ""
        pytest.skip(f"torch.ops.custom.npu_quant_block_sparse_attn is not registered.{suffix}")
    if not custom_ops.has_quant_block_sparse_attn_metadata_op():
        detail = "; ".join(custom_ops.load_errors()[:3])
        suffix = f" Load errors: {detail}" if detail else ""
        pytest.skip(f"torch.ops.custom.npu_quant_block_sparse_attn_metadata is not registered.{suffix}")


def _prepare_kv_inputs(inputs):
    if "kv_cache_storage" not in inputs:
        return None, _to_npu(inputs["key"]), _to_npu(inputs["value"]), _to_npu(inputs["k_descale"])

    kv_cache_storage = _to_npu(inputs["kv_cache_storage"])
    combined_kv_cache.assert_combined_kv_views(kv_cache_storage, inputs["kv_cache_meta"])
    key, value, k_descale = combined_kv_cache.make_combined_kv_views(
        kv_cache_storage, inputs["kv_cache_meta"])
    return kv_cache_storage, key, value, k_descale


def _normalize_npu_output(output, return_softmax_lse):
    if isinstance(output, dict):
        attention_out = output.get("attention_out")
        softmax_lse = output.get("softmax_lse")
    elif isinstance(output, (tuple, list)):
        attention_out = output[0]
        softmax_lse = output[1] if len(output) > 1 else None
    else:
        attention_out = output
        softmax_lse = None

    attention_out_cpu = _to_cpu(attention_out)
    nonzero_count = int(torch.count_nonzero(attention_out_cpu).item()) if attention_out_cpu is not None else 0
    total_count = int(attention_out_cpu.numel()) if attention_out_cpu is not None else 0
    result = {"attention_out": attention_out_cpu}
    if return_softmax_lse and softmax_lse is not None:
        result["softmax_lse"] = _to_cpu(softmax_lse)
    return result


def _golden_output(test_data):
    inputs = test_data["input"]
    if "golden" in test_data:
        golden = {"attention_out": test_data["golden"]["attention_out"]}
        if inputs["return_softmax_lse"]:
            golden["softmax_lse"] = test_data["golden"]["softmax_lse"]
        return golden

    golden = {"attention_out": test_data["cpu_output"]}
    if inputs["return_softmax_lse"]:
        golden["softmax_lse"] = test_data["cpu_softmax_lse"]
    return golden


def _metadata_section_num(metadata):
    metadata_cpu = metadata.detach().cpu().reshape(-1).to(torch.int64)
    if metadata_cpu.numel() < QBSA_HEAD_METADATA_SIZE:
        return 0
    section_num = int(metadata_cpu[0].item())
    min_size = (
        QBSA_HEAD_METADATA_SIZE
        + section_num * QBSA_AIC_CORE_NUM * QBSA_METADATA_SIZE
        + QBSA_AIV_CORE_NUM * QBSA_FD_METADATA_SIZE
    )
    if section_num < 0 or metadata_cpu.numel() < min_size:
        return 0
    return section_num


def _metadata_is_valid(metadata):
    return _metadata_section_num(metadata) > 0


def _metadata_needs_generation(metadata):
    if metadata is None:
        return True
    if metadata.device.type == "cpu" and int(torch.count_nonzero(metadata).item()) == 0:
        return True
    return not _metadata_is_valid(metadata)


def _core_metadata_offset(section_idx, core_idx):
    return (
        QBSA_HEAD_METADATA_SIZE
        + section_idx * QBSA_AIC_CORE_NUM * QBSA_METADATA_SIZE
        + core_idx * QBSA_METADATA_SIZE
    )


def _section_bn1_range(core_metadata):
    if core_metadata[QBSA_CORE_ENABLE_INDEX] == 0:
        return None
    bn1_start = int(core_metadata[QBSA_BN1_START_INDEX])
    bn1_end = int(core_metadata[QBSA_BN1_END_INDEX])
    s1_end = int(core_metadata[QBSA_S1_END_INDEX])
    if s1_end != 0:
        bn1_end += 1
    if bn1_end <= bn1_start:
        return None
    return bn1_start, bn1_end


def _section_s1_range(core_metadata, bn1_idx, bn1_start, bn1_end, qb_limit):
    s1_start = int(core_metadata[QBSA_S1_START_INDEX]) if bn1_idx == bn1_start else 0
    if bn1_idx == bn1_end - 1:
        s1_end = int(core_metadata[QBSA_S1_END_INDEX]) or qb_limit
    else:
        s1_end = qb_limit
    s1_start = max(0, min(s1_start, qb_limit))
    s1_end = max(0, min(s1_end, qb_limit))
    if s1_end <= s1_start:
        return None
    return s1_start, s1_end


def _count_aic_basic_blocks_from_metadata(metadata, sparse_seq_len):
    metadata_cpu = metadata.detach().cpu().reshape(-1).to(torch.int64)
    sparse_seq_len_cpu = sparse_seq_len.detach().cpu().to(torch.int64)
    if sparse_seq_len_cpu.dim() != 3:
        raise ValueError(
            f"sparse_seq_len should be 3D [B,N,Qb], got {tuple(sparse_seq_len_cpu.shape)}"
        )

    section_num = _metadata_section_num(metadata_cpu)
    if section_num == 0:
        raise ValueError(
            f"invalid metadata shape/content: numel={metadata_cpu.numel()}, "
            "section_num is zero or metadata is incomplete"
        )

    batch, head_count, qb_limit = sparse_seq_len_cpu.shape
    bn1_limit = batch * head_count
    totals = [0 for _ in range(QBSA_AIC_CORE_NUM)]
    section_totals = [
        [0 for _ in range(QBSA_AIC_CORE_NUM)] for _ in range(section_num)
    ]

    for section_idx in range(section_num):
        for core_idx in range(QBSA_AIC_CORE_NUM):
            offset = _core_metadata_offset(section_idx, core_idx)
            core_metadata = metadata_cpu[offset : offset + QBSA_METADATA_SIZE].tolist()
            bn1_range = _section_bn1_range(core_metadata)
            if bn1_range is None:
                continue
            bn1_start, bn1_end = bn1_range
            bn1_start = max(0, min(bn1_start, bn1_limit))
            bn1_end = max(0, min(bn1_end, bn1_limit))
            for bn1_idx in range(bn1_start, bn1_end):
                s1_range = _section_s1_range(
                    core_metadata, bn1_idx, bn1_start, bn1_end, qb_limit
                )
                if s1_range is None:
                    continue
                batch_idx = bn1_idx // head_count
                head_idx = bn1_idx % head_count
                s1_start, s1_end = s1_range
                count = int(
                    sparse_seq_len_cpu[
                        batch_idx, head_idx, s1_start:s1_end
                    ].sum().item()
                )
                section_totals[section_idx][core_idx] += count
                totals[core_idx] += count
    return section_num, totals, section_totals


def print_metadata_aic_basic_blocks(metadata, sparse_seq_len, case_name=None):
    section_num, totals, section_totals = _count_aic_basic_blocks_from_metadata(
        metadata, sparse_seq_len
    )
    title = "===== metadata AIC basic block counts ====="
    print(title, flush=True)
    if case_name:
        print(f"case_name={case_name}", flush=True)
    print(f"section_num={section_num}", flush=True)
    for core_idx, total in enumerate(totals):
        per_section = [
            section_totals[section_idx][core_idx] for section_idx in range(section_num)
        ]
        section_text = ",".join(str(value) for value in per_section)
        print(
            f"AIC[{core_idx:02d}] basic_block_count={total} "
            f"section_counts=[{section_text}]",
            flush=True,
        )


def _call_npu_quant_block_sparse_attn_metadata(test_data):
    inputs = test_data["input"]
    metadata_input = test_data.get("metadata_input", {})
    sparse_seq_len = _to_npu(inputs["sparse_seq_len"])
    metadata = torch.ops.custom.npu_quant_block_sparse_attn_metadata(
        sparse_seq_len,
        metadata_input["num_heads_q"],
        metadata_input["num_heads_kv"],
        metadata_input["head_dim"],
        cu_seqlens_q=_to_npu(inputs["cu_seqlens_q"]),
        cu_seqlens_kv=_to_npu(inputs["cu_seqlens_kv"]),
        seqused_q=_to_npu(inputs["seqused_q"]),
        seqused_kv=_to_npu(inputs["seqused_kv"]),
        batch_size=metadata_input.get("batch_size", int(inputs["sparse_seq_len"].shape[0])),
        sparse_block_size_q=inputs["sparse_q_block_size"],
        sparse_block_size_k=inputs["sparse_kv_block_size"],
        quant_mode=inputs["quant_mode"],
        mask_mode=inputs["mask_mode"],
        layout_q=metadata_input.get("layout_q", inputs["layout_q"]),
        layout_kv=metadata_input.get("layout_kv", inputs["layout_kv"]),
        layout_sparse_indices=metadata_input.get("layout_sparse_indices", inputs["layout_sparse_indices"]),
    )
    torch_npu.npu.synchronize()
    return metadata


def _prepare_metadata(test_data):
    metadata = test_data["input"].get("metadata")
    if not _metadata_needs_generation(metadata):
        metadata_npu = _to_npu(metadata)
    else:
        metadata_npu = _call_npu_quant_block_sparse_attn_metadata(test_data)
    if PRINT_METADATA_AIC_BASIC_BLOCKS:
        print_metadata_aic_basic_blocks(
            metadata_npu,
            test_data["input"]["sparse_seq_len"],
            case_name=test_data.get("Testcase_Name"),
        )
    return metadata_npu


def _call_npu_quant_block_sparse_attn(test_data, device_id, normalize_output=True):
    _check_runtime_available()
    torch_npu.npu.set_device(device_id)
    inputs = test_data["input"]
    kv_cache_storage, key, value, k_descale = _prepare_kv_inputs(inputs)
    metadata = _prepare_metadata(test_data)
    output = torch.ops.custom.npu_quant_block_sparse_attn(
        _to_npu(inputs["query"]),
        key,
        value,
        _to_npu(inputs["q_descale"]),
        k_descale,
        _to_npu(inputs["v_descale"]),
        _to_npu(inputs["p_scale"]),
        _to_npu(inputs["sparse_indices"]),
        _to_npu(inputs["sparse_seq_len"]),
        _to_npu(inputs.get("atten_mask")),
        inputs["softmax_scale"],
        inputs["sparse_q_block_size"],
        inputs["sparse_kv_block_size"],
        cu_seqlens_q=_to_npu(inputs["cu_seqlens_q"]),
        cu_seqlens_kv=_to_npu(inputs["cu_seqlens_kv"]),
        seqused_q=_to_npu(inputs["seqused_q"]),
        seqused_kv=_to_npu(inputs["seqused_kv"]),
        block_table=_to_npu(inputs["block_table"]),
        metadata=metadata,
        layout_kv=inputs["layout_kv"],
        layout_q=inputs["layout_q"],
        layout_sparse_indices=inputs["layout_sparse_indices"],
        layout_out=inputs["layout_out"],
        quant_mode=inputs["quant_mode"],
        mask_mode=inputs["mask_mode"],
        return_softmax_lse=inputs["return_softmax_lse"],
    )
    torch_npu.npu.synchronize()
    del kv_cache_storage
    if not normalize_output:
        return output
    return _normalize_npu_output(output, inputs["return_softmax_lse"])


class _QuantBlockSparseAttnGraph(torch.nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.inputs = inputs

    def forward(self, query, key, value, q_descale, k_descale, v_descale, p_scale, sparse_indices,
                sparse_seq_len, atten_mask, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
                block_table, metadata):
        return torch.ops.custom.npu_quant_block_sparse_attn(
            query, key, value, q_descale, k_descale, v_descale, p_scale, sparse_indices, sparse_seq_len,
            atten_mask, self.inputs["softmax_scale"], self.inputs["sparse_q_block_size"],
            self.inputs["sparse_kv_block_size"],
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, seqused_q=seqused_q,
            seqused_kv=seqused_kv, block_table=block_table, metadata=metadata,
            layout_kv=self.inputs["layout_kv"], layout_q=self.inputs["layout_q"],
            layout_sparse_indices=self.inputs["layout_sparse_indices"],
 	        layout_out=self.inputs["layout_out"], quant_mode=self.inputs["quant_mode"],
            mask_mode=self.inputs["mask_mode"], return_softmax_lse=self.inputs["return_softmax_lse"])


def test_quant_block_sparse_attn_process_graph(test_data, device_id=0):
    _check_runtime_available()
    torch_npu.npu.set_device(device_id)
    inputs = test_data["input"]
    kv_cache_storage, key, value, k_descale = _prepare_kv_inputs(inputs)
    metadata = _prepare_metadata(test_data)
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(
        _QuantBlockSparseAttnGraph(inputs).npu(), fullgraph=True, backend=npu_backend, dynamic=False)
    output = model(
        _to_npu(inputs["query"]), key, value, _to_npu(inputs["q_descale"]), k_descale,
        _to_npu(inputs["v_descale"]), _to_npu(inputs["p_scale"]), _to_npu(inputs["sparse_indices"]),
        _to_npu(inputs["sparse_seq_len"]), _to_npu(inputs.get("atten_mask")), _to_npu(inputs["cu_seqlens_q"]),
        _to_npu(inputs["cu_seqlens_kv"]), _to_npu(inputs["seqused_q"]), _to_npu(inputs["seqused_kv"]),
        _to_npu(inputs["block_table"]), metadata)
    torch_npu.npu.synchronize()
    del kv_cache_storage
    return _normalize_npu_output(output, inputs["return_softmax_lse"]), _golden_output(test_data)


def test_quant_block_sparse_attn_process_ci(test_data, device_id=0):
    npu_result = _call_npu_quant_block_sparse_attn(test_data, device_id)
    return npu_result, _golden_output(test_data)


def test_quant_block_sparse_attn_process_perf(test_data, device_id=0):
    return _call_npu_quant_block_sparse_attn(
        test_data, device_id, normalize_output=False
    )
