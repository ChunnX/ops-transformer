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

"""Origin-aware numeric and bitwise comparison for consistency tests."""

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import result_compare_method
from .consistency_model import TokenOrigin


class TensorDigest:
    """Computes dtype-agnostic tensor digests used by the STC MD5 checks."""

    @staticmethod
    def md5(tensor: Optional[torch.Tensor]) -> str:
        if tensor is None:
            return ""
        contiguous = tensor.detach().cpu().contiguous()
        return hashlib.md5(contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()


class OriginComparator:
    """Maps tensors between source and target token-origin orderings."""

    @staticmethod
    def _positions(
        q_lengths: Sequence[int],
        layout_q: str,
        valid_q_lengths: Optional[Sequence[int]] = None,
    ) -> List[Any]:
        valid_q_lengths = valid_q_lengths or q_lengths
        if len(q_lengths) != len(valid_q_lengths):
            raise ValueError("physical and valid Q length counts do not match")
        if any(valid < 0 or valid > physical for physical, valid in zip(q_lengths, valid_q_lengths)):
            raise ValueError(
                f"valid Q lengths {list(valid_q_lengths)} exceed physical lengths {list(q_lengths)}"
            )
        if layout_q == "BSND":
            return [
                (batch, token)
                for batch, length in enumerate(valid_q_lengths)
                for token in range(length)
            ]
        positions = []
        offset = 0
        for physical, valid in zip(q_lengths, valid_q_lengths):
            positions.extend(range(offset, offset + valid))
            offset += physical
        return positions

    @classmethod
    def _read_token(cls, tensor: torch.Tensor, position: Any, layout_q: str, lse: bool) -> torch.Tensor:
        if layout_q == "TND":
            return tensor[position].clone()
        batch, token = position
        return tensor[batch, :, token, :].clone() if lse else tensor[batch, token, :, :].clone()

    @classmethod
    def _write_token(
        cls, tensor: torch.Tensor, position: Any, value: torch.Tensor, layout_q: str, lse: bool
    ) -> None:
        if layout_q == "TND":
            tensor[position] = value
            return
        batch, token = position
        if lse:
            tensor[batch, :, token, :] = value
        else:
            tensor[batch, token, :, :] = value

    @classmethod
    def align_to_template(
        cls,
        source: torch.Tensor,
        source_origins: Sequence[TokenOrigin],
        source_q_lengths: Sequence[int],
        target_origins: Sequence[TokenOrigin],
        target_q_lengths: Sequence[int],
        target_template: torch.Tensor,
        layout_q: str,
        lse: bool = False,
        source_valid_q_lengths: Optional[Sequence[int]] = None,
        target_valid_q_lengths: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        source_positions = cls._positions(source_q_lengths, layout_q, source_valid_q_lengths)
        target_positions = cls._positions(target_q_lengths, layout_q, target_valid_q_lengths)
        if len(source_positions) != len(source_origins):
            raise ValueError(
                f"source origin count {len(source_origins)} != token positions {len(source_positions)}"
            )
        if len(target_positions) != len(target_origins):
            raise ValueError(
                f"target origin count {len(target_origins)} != token positions {len(target_positions)}"
            )

        source_map = {
            origin: cls._read_token(source, position, layout_q, lse)
            for origin, position in zip(source_origins, source_positions)
        }
        if len(source_map) != len(source_origins):
            raise ValueError("Source origin list contains duplicate tokens")

        aligned = torch.zeros_like(target_template)
        for origin, target_position in zip(target_origins, target_positions):
            if origin not in source_map:
                raise ValueError(f"Source output is missing origin {origin}")
            cls._write_token(aligned, target_position, source_map[origin], layout_q, lse)
        return aligned

    @classmethod
    def pack_valid_tokens(
        cls,
        tensor: torch.Tensor,
        q_lengths: Sequence[int],
        layout_q: str,
        valid_q_lengths: Optional[Sequence[int]] = None,
        lse: bool = False,
    ) -> torch.Tensor:
        """Pack only semantically valid tokens into a canonical token-major tensor."""
        positions = cls._positions(q_lengths, layout_q, valid_q_lengths)
        values = [cls._read_token(tensor, position, layout_q, lse) for position in positions]
        if values:
            return torch.stack(values, dim=0)
        if layout_q == "TND":
            token_shape = tuple(tensor.shape[1:])
        elif lse:
            token_shape = (tensor.shape[1], tensor.shape[3])
        else:
            token_shape = tuple(tensor.shape[2:])
        return tensor.new_empty((0,) + token_shape)

    @staticmethod
    def validate_origin_sets(
        baseline_origins: Sequence[TokenOrigin], derived_origins: Sequence[TokenOrigin]
    ) -> Optional[str]:
        baseline_set = set(baseline_origins)
        derived_set = set(derived_origins)
        if len(baseline_set) != len(baseline_origins) or len(derived_set) != len(derived_origins):
            return "Origin list contains duplicate tokens"
        if baseline_set != derived_set:
            return f"Origin mismatch: missing={baseline_set - derived_set}, extra={derived_set - baseline_set}"
        return None


class SplitMerger:
    """Retains full-partition merge as an optional diagnostic, not mode-2's primary oracle."""

    @staticmethod
    def _baseline_origins(q_lengths: Sequence[int]) -> List[TokenOrigin]:
        return [
            TokenOrigin(source_batch=batch, source_token=token)
            for batch, length in enumerate(q_lengths)
            for token in range(length)
        ]

    @classmethod
    def merge_outputs(
        cls,
        sub_results: List[Tuple[torch.Tensor, List[TokenOrigin], Dict[str, Any]]],
        layout_q: str,
        baseline_q_lengths: List[int],
        baseline_valid_q_lengths: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, List[TokenOrigin]]:
        baseline_valid_q_lengths = baseline_valid_q_lengths or baseline_q_lengths
        all_tokens: Dict[TokenOrigin, torch.Tensor] = {}
        for output, origins, transform_meta in sub_results:
            physical_lengths, valid_lengths = cls._case_lengths(origins, transform_meta)
            positions = OriginComparator._positions(physical_lengths, layout_q, valid_lengths)
            for origin, position in zip(origins, positions):
                if origin in all_tokens:
                    raise ValueError(f"Duplicate output origin while merging: {origin}")
                all_tokens[origin] = OriginComparator._read_token(output, position, layout_q, False)

        baseline_origins = cls._baseline_origins(baseline_valid_q_lengths)
        missing = [origin for origin in baseline_origins if origin not in all_tokens]
        if missing:
            raise ValueError(f"Cannot merge an incomplete partition; missing origins: {missing[:4]}")
        sample = next(iter(all_tokens.values()))
        if layout_q == "BSND":
            merged = torch.zeros(
                (len(baseline_q_lengths), max(baseline_q_lengths), sample.shape[0], sample.shape[1]),
                dtype=sample.dtype,
                device=sample.device,
            )
        else:
            merged = torch.zeros(
                (sum(baseline_q_lengths), sample.shape[0], sample.shape[1]),
                dtype=sample.dtype,
                device=sample.device,
            )
        baseline_positions = OriginComparator._positions(
            baseline_q_lengths,
            layout_q,
            baseline_valid_q_lengths,
        )
        for origin, position in zip(baseline_origins, baseline_positions):
            OriginComparator._write_token(merged, position, all_tokens[origin], layout_q, False)
        return merged, baseline_origins

    @classmethod
    def merge_lse(
        cls,
        sub_lses: List[Tuple[Any, ...]],
        layout_q: str,
        baseline_q_lengths: List[int],
        baseline_valid_q_lengths: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, List[TokenOrigin]]:
        baseline_valid_q_lengths = baseline_valid_q_lengths or baseline_q_lengths
        all_tokens: Dict[TokenOrigin, torch.Tensor] = {}
        for sub_lse in sub_lses:
            if len(sub_lse) == 2:
                lse, origins = sub_lse
                transform_meta: Dict[str, Any] = {}
            elif len(sub_lse) == 3:
                lse, origins, transform_meta = sub_lse
            else:
                raise ValueError("Each split LSE entry must have 2 or 3 values")
            if lse is None:
                raise ValueError("return_softmax_lse=True but a split result has no LSE tensor")
            physical_lengths, valid_lengths = cls._case_lengths(origins, transform_meta)
            positions = OriginComparator._positions(physical_lengths, layout_q, valid_lengths)
            for origin, position in zip(origins, positions):
                if origin in all_tokens:
                    raise ValueError(f"Duplicate LSE origin while merging: {origin}")
                all_tokens[origin] = OriginComparator._read_token(lse, position, layout_q, True)

        baseline_origins = cls._baseline_origins(baseline_valid_q_lengths)
        missing = [origin for origin in baseline_origins if origin not in all_tokens]
        if missing:
            raise ValueError(f"Cannot merge incomplete LSE results; missing origins: {missing[:4]}")
        sample = next(iter(all_tokens.values()))
        if layout_q == "BSND":
            merged = torch.zeros(
                (len(baseline_q_lengths), sample.shape[0], max(baseline_q_lengths), sample.shape[1]),
                dtype=sample.dtype,
                device=sample.device,
            )
        else:
            merged = torch.zeros(
                (sum(baseline_q_lengths), sample.shape[0], sample.shape[1]),
                dtype=sample.dtype,
                device=sample.device,
            )
        baseline_positions = OriginComparator._positions(
            baseline_q_lengths,
            layout_q,
            baseline_valid_q_lengths,
        )
        for origin, position in zip(baseline_origins, baseline_positions):
            OriginComparator._write_token(merged, position, all_tokens[origin], layout_q, True)
        return merged, baseline_origins

    @classmethod
    def _case_lengths(
        cls,
        origins: Sequence[TokenOrigin],
        transform_meta: Dict[str, Any],
    ) -> Tuple[List[int], List[int]]:
        valid_lengths = list(
            transform_meta.get("valid_q_lengths", cls._lengths_from_origins(origins))
        )
        physical_lengths = list(transform_meta.get("physical_q_lengths", valid_lengths))
        return physical_lengths, valid_lengths

    @staticmethod
    def _lengths_from_origins(origins: Sequence[TokenOrigin]) -> List[int]:
        lengths = []
        current_batch = object()
        for origin in origins:
            if origin.source_batch != current_batch:
                lengths.append(0)
                current_batch = origin.source_batch
            lengths[-1] += 1
        return lengths


def tensor_md5(tensor: Optional[torch.Tensor]) -> str:
    return TensorDigest.md5(tensor)


def compare_pair(
    expected: torch.Tensor, actual: torch.Tensor, require_bitwise: bool = False
) -> Dict[str, Any]:
    expected_md5 = TensorDigest.md5(expected)
    actual_md5 = TensorDigest.md5(actual)
    bitwise_equal = expected_md5 == actual_md5
    if expected.shape != actual.shape:
        return {
            "pass": False,
            "numeric_pass": False,
            "fulfill_percent": 0.0,
            "error": f"shape mismatch: expected {tuple(expected.shape)}, actual {tuple(actual.shape)}",
            "expected_md5": expected_md5,
            "actual_md5": actual_md5,
            "bitwise_equal": False,
        }
    result = result_compare_method.check_result(expected, actual)
    numeric_pass = result[0].lower() in {"pass", "passed"}
    return {
        "pass": numeric_pass and (bitwise_equal or not require_bitwise),
        "numeric_pass": numeric_pass,
        "fulfill_percent": float(result[1]),
        "expected_md5": expected_md5,
        "actual_md5": actual_md5,
        "bitwise_equal": bitwise_equal,
        "bitwise_required": require_bitwise,
    }


def compare_consistency(
    baseline_output: torch.Tensor,
    baseline_origins: List[TokenOrigin],
    derived_output: torch.Tensor,
    derived_origins: List[TokenOrigin],
    layout_q: str,
    baseline_q_lengths: List[int],
    derived_q_lengths: Optional[List[int]] = None,
    compare_lse: bool = False,
    baseline_lse: Optional[torch.Tensor] = None,
    derived_lse: Optional[torch.Tensor] = None,
    require_bitwise: bool = True,
    baseline_valid_q_lengths: Optional[List[int]] = None,
    derived_valid_q_lengths: Optional[List[int]] = None,
) -> Dict[str, Any]:
    error = OriginComparator.validate_origin_sets(baseline_origins, derived_origins)
    if error:
        return {"pass": False, "error": error}
    derived_q_lengths = derived_q_lengths or baseline_q_lengths
    baseline_valid_q_lengths = baseline_valid_q_lengths or baseline_q_lengths
    derived_valid_q_lengths = derived_valid_q_lengths or derived_q_lengths
    aligned_output = OriginComparator.align_to_template(
        derived_output,
        derived_origins,
        derived_q_lengths,
        baseline_origins,
        baseline_q_lengths,
        baseline_output,
        layout_q,
        source_valid_q_lengths=derived_valid_q_lengths,
        target_valid_q_lengths=baseline_valid_q_lengths,
    )
    expected_output = OriginComparator.pack_valid_tokens(
        baseline_output,
        baseline_q_lengths,
        layout_q,
        baseline_valid_q_lengths,
    )
    actual_output = OriginComparator.pack_valid_tokens(
        aligned_output,
        baseline_q_lengths,
        layout_q,
        baseline_valid_q_lengths,
    )
    output_result = compare_pair(expected_output, actual_output, require_bitwise=require_bitwise)
    result: Dict[str, Any] = {"output": output_result, "pass": output_result["pass"]}
    if compare_lse and (baseline_lse is None or derived_lse is None):
        result["lse"] = {"pass": False, "fulfill_percent": 0.0, "error": "LSE result is missing"}
        result["pass"] = False
    elif compare_lse:
        aligned_lse = OriginComparator.align_to_template(
            derived_lse,
            derived_origins,
            derived_q_lengths,
            baseline_origins,
            baseline_q_lengths,
            baseline_lse,
            layout_q,
            lse=True,
            source_valid_q_lengths=derived_valid_q_lengths,
            target_valid_q_lengths=baseline_valid_q_lengths,
        )
        expected_lse = OriginComparator.pack_valid_tokens(
            baseline_lse,
            baseline_q_lengths,
            layout_q,
            baseline_valid_q_lengths,
            lse=True,
        )
        actual_lse = OriginComparator.pack_valid_tokens(
            aligned_lse,
            baseline_q_lengths,
            layout_q,
            baseline_valid_q_lengths,
            lse=True,
        )
        result["lse"] = compare_pair(expected_lse, actual_lse, require_bitwise=require_bitwise)
        result["pass"] = result["pass"] and result["lse"]["pass"]
    return result


def compare_subset(
    baseline_output: torch.Tensor,
    baseline_origins: List[TokenOrigin],
    baseline_q_lengths: List[int],
    derived_output: torch.Tensor,
    derived_origins: List[TokenOrigin],
    derived_q_lengths: List[int],
    layout_q: str,
    compare_lse: bool = False,
    baseline_lse: Optional[torch.Tensor] = None,
    derived_lse: Optional[torch.Tensor] = None,
    require_bitwise: bool = True,
    baseline_valid_q_lengths: Optional[List[int]] = None,
    derived_valid_q_lengths: Optional[List[int]] = None,
) -> Dict[str, Any]:
    baseline_valid_q_lengths = baseline_valid_q_lengths or baseline_q_lengths
    derived_valid_q_lengths = derived_valid_q_lengths or derived_q_lengths
    try:
        expected_output = OriginComparator.align_to_template(
            baseline_output,
            baseline_origins,
            baseline_q_lengths,
            derived_origins,
            derived_q_lengths,
            derived_output,
            layout_q,
            source_valid_q_lengths=baseline_valid_q_lengths,
            target_valid_q_lengths=derived_valid_q_lengths,
        )
    except ValueError as error:
        return {"pass": False, "error": str(error)}
    expected_packed = OriginComparator.pack_valid_tokens(
        expected_output,
        derived_q_lengths,
        layout_q,
        derived_valid_q_lengths,
    )
    actual_packed = OriginComparator.pack_valid_tokens(
        derived_output,
        derived_q_lengths,
        layout_q,
        derived_valid_q_lengths,
    )
    output_result = compare_pair(expected_packed, actual_packed, require_bitwise=require_bitwise)
    result: Dict[str, Any] = {"output": output_result, "pass": output_result["pass"]}
    if compare_lse and (baseline_lse is None or derived_lse is None):
        result["lse"] = {"pass": False, "fulfill_percent": 0.0, "error": "LSE result is missing"}
        result["pass"] = False
    elif compare_lse:
        expected_lse = OriginComparator.align_to_template(
            baseline_lse,
            baseline_origins,
            baseline_q_lengths,
            derived_origins,
            derived_q_lengths,
            derived_lse,
            layout_q,
            lse=True,
            source_valid_q_lengths=baseline_valid_q_lengths,
            target_valid_q_lengths=derived_valid_q_lengths,
        )
        expected_lse_packed = OriginComparator.pack_valid_tokens(
            expected_lse,
            derived_q_lengths,
            layout_q,
            derived_valid_q_lengths,
            lse=True,
        )
        actual_lse_packed = OriginComparator.pack_valid_tokens(
            derived_lse,
            derived_q_lengths,
            layout_q,
            derived_valid_q_lengths,
            lse=True,
        )
        result["lse"] = compare_pair(
            expected_lse_packed,
            actual_lse_packed,
            require_bitwise=require_bitwise,
        )
        result["pass"] = result["pass"] and result["lse"]["pass"]
    return result


def merge_split_outputs(
    sub_results: List[Tuple[torch.Tensor, List[TokenOrigin], Dict[str, Any]]],
    layout_q: str,
    baseline_q_lengths: List[int],
    baseline_valid_q_lengths: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, List[TokenOrigin]]:
    return SplitMerger.merge_outputs(
        sub_results,
        layout_q,
        baseline_q_lengths,
        baseline_valid_q_lengths,
    )


def merge_split_lse(
    sub_lses: List[Tuple[Any, ...]],
    layout_q: str,
    baseline_q_lengths: List[int],
    baseline_valid_q_lengths: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, List[TokenOrigin]]:
    return SplitMerger.merge_lse(
        sub_lses,
        layout_q,
        baseline_q_lengths,
        baseline_valid_q_lengths,
    )
