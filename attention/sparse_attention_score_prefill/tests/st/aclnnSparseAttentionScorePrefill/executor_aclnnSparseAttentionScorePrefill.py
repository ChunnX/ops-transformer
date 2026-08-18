#!/usr/bin/env python3
"""ATK reference and aclnn executor for SparseAttentionScorePrefill.

The case intentionally uses one valid paged KV block per KV head.  This keeps
the ATK smoke test small while exercising the real CSR metadata, GQA mapping,
the three A2 precision tiling keys, and the optional LSE output.
"""

from __future__ import annotations

import ctypes
import math

import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor


Q_TOKENS = 16
Q_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 128
BLOCK_SIZE = 128
TOP_K = 1
SCALE = 1.0 / math.sqrt(HEAD_DIM)


def _deterministic_bf16(shape: tuple[int, ...], salt: int) -> torch.Tensor:
    """Create identical bounded BF16 tensors for CPU and NPU backends."""
    values = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    values = ((values.remainder(97.0) - 48.0) / 96.0) + float(salt) / 256.0
    return values.to(torch.bfloat16).contiguous()


def _install_deterministic_inputs(input_data: InputDataset) -> None:
    """Replace ATK's generic random tensors with valid, reproducible metadata."""
    kwargs = input_data.kwargs
    continuous_kv = int(kwargs.get("blockSize", BLOCK_SIZE)) == 0
    kwargs["query"] = _deterministic_bf16((Q_TOKENS, Q_HEADS, HEAD_DIM), 1)
    if continuous_kv:
        kwargs["key"] = _deterministic_bf16((BLOCK_SIZE, KV_HEADS, HEAD_DIM), 7)
        kwargs["value"] = _deterministic_bf16((BLOCK_SIZE, KV_HEADS, HEAD_DIM), 13)
    else:
        kwargs["key"] = _deterministic_bf16((1, BLOCK_SIZE, KV_HEADS, HEAD_DIM), 7)
        kwargs["value"] = _deterministic_bf16((1, BLOCK_SIZE, KV_HEADS, HEAD_DIM), 13)

    # One packed KV row per KV head, with every query token attached to slot 0.
    kwargs["blockTable"] = torch.zeros((1, TOP_K), dtype=torch.int32)
    kwargs["k2qRowPtr"] = torch.tensor(
        [[0, Q_TOKENS], [0, Q_TOKENS]], dtype=torch.int32)
    q_indices = torch.arange(Q_TOKENS, dtype=torch.int32).repeat(KV_HEADS, 1)
    kwargs["k2qQIndices"] = q_indices
    kwargs["k2qSlotIndices"] = torch.zeros((KV_HEADS, Q_TOKENS), dtype=torch.int32)
    kwargs["actualSeqLengths"] = torch.tensor([Q_TOKENS], dtype=torch.int32)
    kwargs["actualSeqLengthsKv"] = torch.tensor([BLOCK_SIZE], dtype=torch.int32)
    kwargs["numKeyValueHeads"] = KV_HEADS
    kwargs["scaleValue"] = SCALE
    kwargs["blockSize"] = 0 if continuous_kv else BLOCK_SIZE
    kwargs["topK"] = TOP_K
    kwargs["softmaxLseFlag"] = True


def _reference(input_data: InputDataset):
    _install_deterministic_inputs(input_data)
    kwargs = input_data.kwargs
    query = kwargs["query"].detach().cpu().to(torch.float32)
    key = kwargs["key"].detach().cpu().to(torch.float32)
    value = kwargs["value"].detach().cpu().to(torch.float32)
    block_table = kwargs["blockTable"].detach().cpu()
    row_ptr = kwargs["k2qRowPtr"].detach().cpu()
    q_indices = kwargs["k2qQIndices"].detach().cpu()
    slot_indices = kwargs["k2qSlotIndices"].detach().cpu()
    q_seqlen = int(kwargs["actualSeqLengths"][0])
    kv_seqlen = int(kwargs["actualSeqLengthsKv"][0])
    inner_precise = int(kwargs["innerPrecise"])
    continuous_kv = int(kwargs["blockSize"]) == 0

    output = torch.zeros((Q_TOKENS, Q_HEADS, HEAD_DIM), dtype=torch.float32)
    lse = torch.full((Q_TOKENS, Q_HEADS, 1), float("-inf"), dtype=torch.float32)
    group_size = Q_HEADS // KV_HEADS
    history_len = kv_seqlen - q_seqlen

    for kv_head in range(KV_HEADS):
        start = int(row_ptr[kv_head, 0])
        end = int(row_ptr[kv_head, 1])
        for entry in range(start, end):
            q_token = int(q_indices[kv_head, entry])
            slot = int(slot_indices[kv_head, entry])
            if continuous_kv:
                physical_block = 0
                block_begin = 0
                block_end = min(BLOCK_SIZE, kv_seqlen)
            else:
                physical_block = int(block_table[0, slot])
                block_begin = slot * BLOCK_SIZE
                block_end = min(block_begin + BLOCK_SIZE, kv_seqlen)
            causal_end = min(block_end, history_len + q_token + 1)
            valid_len = max(causal_end - block_begin, 0)
            if valid_len == 0:
                continue

            if continuous_kv:
                k = key[:valid_len, kv_head, :]
                v = value[:valid_len, kv_head, :]
            else:
                k = key[physical_block, :valid_len, kv_head, :]
                v = value[physical_block, :valid_len, kv_head, :]
            for q_head in range(kv_head * group_size, (kv_head + 1) * group_size):
                scores = torch.matmul(query[q_token, q_head], k.transpose(0, 1))
                if inner_precise != 0:
                    scores = scores.to(torch.bfloat16).to(torch.float32)
                scores = scores * SCALE

                row_max = scores.max()
                p_fp32 = torch.exp(scores - row_max)
                row_sum = p_fp32.sum()
                o_partial = torch.matmul(p_fp32.to(torch.bfloat16).float(), v)
                if inner_precise == 1:
                    o_partial = o_partial.to(torch.bfloat16).to(torch.float32)

                output[q_token, q_head] = o_partial / row_sum
                lse[q_token, q_head, 0] = row_max + torch.log(row_sum)

    # Return outputs in the ACLNN signature order.
    return [output.to(torch.bfloat16).contiguous(), lse.contiguous()]


@register("ascend_sparse_attention_score_prefill")
class SparseAttentionScorePrefillReference(BaseApi):
    """CPU golden implementation used by ATK's accuracy task."""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        return _reference(input_data)


@register("ascend_sparse_attention_score_prefill_npu")
class SparseAttentionScorePrefillNpu(AclnnBaseApi):
    """ATK aclnn executor for the installed A2 custom operator."""

    def __init__(self, task_result: TaskResult, backend):
        super().__init__(task_result, backend)
        self.input_data = None

    def init_by_input_data(self, input_data: InputDataset):
        _install_deterministic_inputs(input_data)
        self.input_data = input_data
        input_args, output_packages = super().init_by_input_data(input_data)
        if int(input_data.kwargs["blockSize"]) == 0:
            # The ACLNN contract makes blockTable optional for concatenated KV.
            # Keep a descriptor in the ATK case for stable argument ordering,
            # then pass a typed null ACL tensor pointer to the C API.
            acl_tensor_ptr = ctypes.POINTER(AclTensor)
            input_args[3] = ctypes.cast(ctypes.c_void_p(None), acl_tensor_ptr)
        return input_args, output_packages

    def after_call(self, output_packages):
        return [self.acl_tensor_to_torch(package).cpu() for package in output_packages]

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnSparseAttentionScorePrefillGetWorkspaceSize("
            "const aclTensor *query, const aclTensor *key, const aclTensor *value, "
            "const aclTensor *blockTable, const aclTensor *k2qRowPtr, "
            "const aclTensor *k2qQIndices, const aclTensor *k2qSlotIndices, "
            "const aclTensor *actualSeqLengthsOptional, "
            "const aclTensor *actualSeqLengthsKvOptional, int64_t numKeyValueHeads, "
            "double scaleValue, int64_t blockSize, int64_t topK, int64_t innerPrecise, "
            "bool softmaxLseFlag, aclTensor *attentionOut, aclTensor *softmaxLse, "
            "uint64_t *workspaceSize, aclOpExecutor **executor)"
        )
