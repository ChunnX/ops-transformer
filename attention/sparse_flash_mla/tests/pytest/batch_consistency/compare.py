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

"""Origin-aligned comparison for batch consistency testing.

Reuses :func:`result_compare_method.check_result` for numerical tolerance
so that consistency comparisons use the same thresholds as the golden
comparison.  MD5 is used only as a diagnostic for bitwise equality.
"""

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import torch

import result_compare_method
from .model import TokenOrigin


def _tensor_md5(t: torch.Tensor) -> str:
    """Return hex MD5 of a tensor's raw bytes (CPU, contiguous).

    Uses ``view(torch.uint8)`` to handle BF16 and other non-NumPy dtypes
    that would cause ``TypeError`` with ``tensor.numpy()``.
    """
    if t is None:
        return ""
    t = t.detach().cpu().contiguous()
    return hashlib.md5(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def compare_pair(expected: torch.Tensor, actual: torch.Tensor) -> Dict[str, Any]:
    """Thin adapter around ``result_compare_method.check_result``.

    Returns a dict with ``pass``, ``fulfill_percent``, and diagnostic md5.
    MD5 is computed once per tensor (not three times).
    """
    expected_md5 = _tensor_md5(expected)
    actual_md5 = _tensor_md5(actual)
    if expected.shape != actual.shape:
        return {
            "pass": False,
            "fulfill_percent": 0.0,
            "error": f"shape mismatch: expected {tuple(expected.shape)}, actual {tuple(actual.shape)}",
            "expected_md5": expected_md5,
            "actual_md5": actual_md5,
            "bitwise_equal": False,
        }
    result = result_compare_method.check_result(expected, actual)
    status, percent = result[0], result[1]
    return {
        "pass": status.lower() in {"pass", "passed"},
        "fulfill_percent": float(percent),
        "expected_md5": expected_md5,
        "actual_md5": actual_md5,
        "bitwise_equal": expected_md5 == actual_md5,
    }


def _align_by_origin(
    output: torch.Tensor,
    origins: List[TokenOrigin],
    target_origins: List[TokenOrigin],
    layout_q: str,
    B: int,
    q_lengths: List[int],
) -> torch.Tensor:
    """Reorder *output* so its tokens are in *target_origins* order.

    Each token in *output* is identified by its *origins* entry.  We gather
    them into the order given by *target_origins*.

    For BSND output ``[B, S1, N1, D]`` the token at ``(b, s)`` is at
    ``output[b, s, :, :]``.

    For TND output ``[T, N1, D]`` tokens are laid out contiguously per batch
    in the same order as *origins*.
    """
    # Build a lookup: origin -> position in output
    origin_to_pos: Dict[TokenOrigin, int] = {}
    for pos, origin in enumerate(origins):
        origin_to_pos[origin] = pos

    if layout_q == "BSND":
        # output shape: [B, S1, N1, D]
        S1 = output.shape[1]
        # Build the target tensor
        target = torch.zeros_like(output)
        for target_idx, target_origin in enumerate(target_origins):
            src_pos = origin_to_pos[target_origin]
            # In BSND, position maps to (batch, token)
            src_b = src_pos // S1 if S1 > 0 else 0
            src_s = src_pos % S1 if S1 > 0 else 0
            tgt_b = target_idx // S1 if S1 > 0 else 0
            tgt_s = target_idx % S1 if S1 > 0 else 0
            target[tgt_b, tgt_s, :, :] = output[src_b, src_s, :, :]
        return target

    else:  # TND
        # output shape: [T, N1, D]
        N1 = output.shape[1]
        D = output.shape[2]
        total_T = sum(q_lengths)
        target = torch.zeros((total_T, N1, D), dtype=output.dtype, device=output.device)
        tgt_offset = 0
        for target_origin in target_origins:
            src_pos = origin_to_pos[target_origin]
            target[tgt_offset] = output[src_pos]
            tgt_offset += 1
        return target


def compare_consistency(
    baseline_output: torch.Tensor,
    baseline_origins: List[TokenOrigin],
    derived_output: torch.Tensor,
    derived_origins: List[TokenOrigin],
    layout_q: str,
    baseline_q_lengths: List[int],
    compare_lse: bool = False,
    baseline_lse: Optional[torch.Tensor] = None,
    derived_lse: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Compare derived output against baseline by token origin.

    The derived output is realigned to baseline order, then
    :func:`compare_pair` is applied.

    Returns a dict with overall pass/fail and per-component results.
    """
    # Verify origin sets match
    baseline_origin_set = set(baseline_origins)
    derived_origin_set = set(derived_origins)
    if len(baseline_origin_set) != len(baseline_origins) or len(derived_origin_set) != len(derived_origins):
        return {"pass": False, "error": "Origin list contains duplicate tokens"}
    if baseline_origin_set != derived_origin_set:
        missing = baseline_origin_set - derived_origin_set
        extra = derived_origin_set - baseline_origin_set
        return {
            "pass": False,
            "error": f"Origin mismatch: missing={missing}, extra={extra}",
        }

    # Align derived output to baseline order
    aligned_output = _align_by_origin(
        derived_output,
        derived_origins,
        baseline_origins,
        layout_q,
        len(baseline_q_lengths),
        baseline_q_lengths,
    )

    output_result = compare_pair(baseline_output, aligned_output)
    result: Dict[str, Any] = {
        "output": output_result,
        "pass": output_result["pass"],
    }

    if compare_lse and (baseline_lse is None or derived_lse is None):
        result["lse"] = {
            "pass": False,
            "fulfill_percent": 0.0,
            "error": "return_softmax_lse=True but an LSE result is missing",
        }
        result["pass"] = False
    elif compare_lse:
        aligned_lse = _align_lse_by_origin(
            derived_lse, derived_origins, baseline_origins, layout_q, baseline_q_lengths
        )
        lse_result = compare_pair(baseline_lse, aligned_lse)
        result["lse"] = lse_result
        result["pass"] = result["pass"] and lse_result["pass"]

    return result


def _align_lse_by_origin(
    lse: torch.Tensor,
    origins: List[TokenOrigin],
    target_origins: List[TokenOrigin],
    layout_q: str,
    q_lengths: List[int],
) -> torch.Tensor:
    """Align LSE tensor by token origin.

    LSE shape for BSND: [B, N2, S1, G]
    LSE shape for TND:  [T, N2, G]
    """
    origin_to_pos: Dict[TokenOrigin, int] = {}
    for pos, origin in enumerate(origins):
        origin_to_pos[origin] = pos

    if layout_q == "BSND":
        S1 = lse.shape[2]
        target = torch.zeros_like(lse)
        for target_idx, target_origin in enumerate(target_origins):
            src_pos = origin_to_pos[target_origin]
            src_b = src_pos // S1 if S1 > 0 else 0
            src_s = src_pos % S1 if S1 > 0 else 0
            tgt_b = target_idx // S1 if S1 > 0 else 0
            tgt_s = target_idx % S1 if S1 > 0 else 0
            target[tgt_b, :, tgt_s, :] = lse[src_b, :, src_s, :]
        return target
    else:  # TND
        N2 = lse.shape[1] if lse.dim() >= 2 else 1
        G = lse.shape[2] if lse.dim() >= 3 else 1
        total_T = sum(q_lengths)
        target = torch.zeros((total_T, N2, G), dtype=lse.dtype, device=lse.device)
        tgt_offset = 0
        for target_origin in target_origins:
            src_pos = origin_to_pos[target_origin]
            target[tgt_offset] = lse[src_pos]
            tgt_offset += 1
        return target


def merge_split_outputs(
    sub_results: List[Tuple[torch.Tensor, List[TokenOrigin], Dict[str, Any]]],
    layout_q: str,
    baseline_q_lengths: List[int],
) -> Tuple[torch.Tensor, List[TokenOrigin]]:
    """Merge sub-case outputs from Mode 2 into a single tensor.

    Each element of *sub_results* is ``(output, origins, meta)``.
    The merged tensor is ordered by the baseline batch/token order.

    BSND output: [B, S1, N1, D]
    TND output:  [T, N1, D]
    """
    all_tokens: Dict[TokenOrigin, torch.Tensor] = {}

    for output, origins, _meta in sub_results:
        if layout_q == "BSND":
            S1 = output.shape[1]
            for pos, origin in enumerate(origins):
                b = pos // S1 if S1 > 0 else 0
                s = pos % S1 if S1 > 0 else 0
                if origin in all_tokens:
                    raise ValueError(f"Duplicate output origin while merging: {origin}")
                all_tokens[origin] = output[b, s, :, :].clone()
        else:  # TND
            for pos, origin in enumerate(origins):
                if origin in all_tokens:
                    raise ValueError(f"Duplicate output origin while merging: {origin}")
                all_tokens[origin] = output[pos].clone()

    # Build merged tensor in baseline order
    baseline_origins = []
    for b, length in enumerate(baseline_q_lengths):
        for s in range(length):
            baseline_origins.append(TokenOrigin(source_batch=b, source_token=s))

    if layout_q == "BSND":
        B = len(baseline_q_lengths)
        S1 = max(baseline_q_lengths) if baseline_q_lengths else 0
        sample_token = next(iter(all_tokens.values()))
        N1 = sample_token.shape[0]
        D = sample_token.shape[1]
        merged = torch.zeros((B, S1, N1, D), dtype=sample_token.dtype, device=sample_token.device)
        for origin in baseline_origins:
            b = origin.source_batch
            s = origin.source_token
            merged[b, s, :, :] = all_tokens[origin]
    else:  # TND
        total_T = sum(baseline_q_lengths)
        sample_token = next(iter(all_tokens.values()))
        N1 = sample_token.shape[0]
        D = sample_token.shape[1]
        merged = torch.zeros((total_T, N1, D), dtype=sample_token.dtype, device=sample_token.device)
        offset = 0
        for b, length in enumerate(baseline_q_lengths):
            for s in range(length):
                merged[offset] = all_tokens[TokenOrigin(source_batch=b, source_token=s)]
                offset += 1

    return merged, baseline_origins


def merge_split_lse(
    sub_lses: List[Tuple[torch.Tensor, List[TokenOrigin]]],
    layout_q: str,
    baseline_q_lengths: List[int],
) -> Tuple[torch.Tensor, List[TokenOrigin]]:
    """Merge sub-case LSE outputs from Mode 2 into a single tensor.

    Each element of *sub_lses* is ``(lse, origins)``.

    BSND LSE: [B, N2, S1, G]
    TND LSE:  [T, N2, G]
    """
    all_tokens: Dict[TokenOrigin, torch.Tensor] = {}

    for lse, origins in sub_lses:
        if lse is None:
            raise ValueError("return_softmax_lse=True but a split result has no LSE tensor")
        if layout_q == "BSND":
            S1 = lse.shape[2]
            for pos, origin in enumerate(origins):
                b = pos // S1 if S1 > 0 else 0
                s = pos % S1 if S1 > 0 else 0
                if origin in all_tokens:
                    raise ValueError(f"Duplicate LSE origin while merging: {origin}")
                all_tokens[origin] = lse[b, :, s, :].clone()
        else:  # TND
            for pos, origin in enumerate(origins):
                if origin in all_tokens:
                    raise ValueError(f"Duplicate LSE origin while merging: {origin}")
                all_tokens[origin] = lse[pos].clone()

    baseline_origins = []
    for b, length in enumerate(baseline_q_lengths):
        for s in range(length):
            baseline_origins.append(TokenOrigin(source_batch=b, source_token=s))

    if layout_q == "BSND":
        B = len(baseline_q_lengths)
        S1 = max(baseline_q_lengths) if baseline_q_lengths else 0
        sample_token = next(iter(all_tokens.values()))
        N2 = sample_token.shape[0]
        G = sample_token.shape[1]
        merged = torch.zeros((B, N2, S1, G), dtype=sample_token.dtype, device=sample_token.device)
        for origin in baseline_origins:
            b = origin.source_batch
            s = origin.source_token
            merged[b, :, s, :] = all_tokens[origin]
    else:  # TND
        total_T = sum(baseline_q_lengths)
        sample_token = next(iter(all_tokens.values()))
        N2 = sample_token.shape[0]
        G = sample_token.shape[1]
        merged = torch.zeros((total_T, N2, G), dtype=sample_token.dtype, device=sample_token.device)
        offset = 0
        for b, length in enumerate(baseline_q_lengths):
            for s in range(length):
                merged[offset] = all_tokens[TokenOrigin(source_batch=b, source_token=s)]
                offset += 1

    return merged, baseline_origins
