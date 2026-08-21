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

"""Origin-aware output comparison for batch-consistency relations."""

import hashlib
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from .model import RunResult, TokenOrigin


class OriginComparator:
    """Align output tokens by their position in the original pytest case."""

    @staticmethod
    def positions(
        physical_lengths: Sequence[int],
        layout_q: str,
        valid_lengths: Optional[Sequence[int]] = None,
    ) -> List[Any]:
        valid_lengths = valid_lengths or physical_lengths
        if len(physical_lengths) != len(valid_lengths):
            raise ValueError("physical and valid Q length counts do not match")
        if any(
            valid < 0 or valid > physical
            for physical, valid in zip(physical_lengths, valid_lengths)
        ):
            raise ValueError(
                f"valid Q lengths {list(valid_lengths)} exceed physical lengths "
                f"{list(physical_lengths)}"
            )
        if layout_q == "BSND":
            return [
                (batch, token)
                for batch, length in enumerate(valid_lengths)
                for token in range(length)
            ]

        positions = []
        offset = 0
        for physical, valid in zip(physical_lengths, valid_lengths):
            positions.extend(range(offset, offset + valid))
            offset += physical
        return positions

    @staticmethod
    def read_token(tensor: torch.Tensor, position: Any, layout_q: str, lse: bool) -> torch.Tensor:
        if layout_q == "TND":
            return tensor[position]
        batch, token = position
        return tensor[batch, :, token, :] if lse else tensor[batch, token, :, :]

    @classmethod
    def pack(
        cls,
        tensor: torch.Tensor,
        physical_lengths: Sequence[int],
        valid_lengths: Sequence[int],
        layout_q: str,
        lse: bool,
    ) -> torch.Tensor:
        values = [
            cls.read_token(tensor, position, layout_q, lse)
            for position in cls.positions(physical_lengths, layout_q, valid_lengths)
        ]
        if values:
            return torch.stack(values).contiguous()
        token_shape = tuple(tensor.shape[1:]) if layout_q == "TND" else tuple(tensor.shape[2:])
        return tensor.new_empty((0,) + token_shape)

    @classmethod
    def align_expected(
        cls,
        expected: torch.Tensor,
        expected_origins: Sequence[TokenOrigin],
        expected_physical: Sequence[int],
        expected_valid: Sequence[int],
        actual_origins: Sequence[TokenOrigin],
        actual_physical: Sequence[int],
        actual_valid: Sequence[int],
        layout_q: str,
        lse: bool,
    ) -> torch.Tensor:
        expected_positions = cls.positions(expected_physical, layout_q, expected_valid)
        actual_positions = cls.positions(actual_physical, layout_q, actual_valid)
        if len(expected_positions) != len(expected_origins):
            raise ValueError("baseline output and token-origin counts differ")
        if len(actual_positions) != len(actual_origins):
            raise ValueError("derived output and token-origin counts differ")

        source = {
            origin: cls.read_token(expected, position, layout_q, lse)
            for origin, position in zip(expected_origins, expected_positions)
        }
        if len(source) != len(expected_origins):
            raise ValueError("baseline token origins contain duplicates")
        try:
            return torch.stack([source[origin] for origin in actual_origins]).contiguous()
        except KeyError as error:
            raise ValueError(f"derived output references unknown token {error.args[0]}") from error


class ResultComparator:
    """Apply the strict bitwise gate and the operator's existing precision gate."""

    def __init__(self, precision_compare: Callable[[torch.Tensor, torch.Tensor], Any]):
        self.precision_compare = precision_compare

    @staticmethod
    def digest(tensor: torch.Tensor) -> str:
        data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.md5(data).hexdigest()

    def compare_tensor(
        self, expected: torch.Tensor, actual: torch.Tensor, require_bitwise: bool
    ) -> Dict[str, Any]:
        expected = expected.detach().cpu()
        actual = actual.detach().cpu()
        expected_md5 = self.digest(expected)
        actual_md5 = self.digest(actual)
        bitwise_equal = expected.shape == actual.shape and expected_md5 == actual_md5
        if expected.shape != actual.shape:
            return {
                "pass": False,
                "error": f"shape mismatch: expected {tuple(expected.shape)}, actual {tuple(actual.shape)}",
                "bitwise_equal": False,
            }

        precision = self.precision_compare(expected, actual)
        numeric_pass = str(precision[0]).lower() in {"pass", "passed"}
        return {
            "pass": numeric_pass and (bitwise_equal or not require_bitwise),
            "numeric_pass": numeric_pass,
            "fulfill_percent": float(precision[1]),
            "expected_md5": expected_md5,
            "actual_md5": actual_md5,
            "bitwise_equal": bitwise_equal,
            "bitwise_required": require_bitwise,
        }

    def compare_results(
        self,
        expected: RunResult,
        expected_origins: Sequence[TokenOrigin],
        expected_physical: Sequence[int],
        expected_valid: Sequence[int],
        actual: RunResult,
        actual_origins: Sequence[TokenOrigin],
        actual_physical: Sequence[int],
        actual_valid: Sequence[int],
        layout_q: str,
        compare_lse: bool,
        require_bitwise: bool,
    ) -> Dict[str, Any]:
        expected_output = OriginComparator.align_expected(
            expected.output,
            expected_origins,
            expected_physical,
            expected_valid,
            actual_origins,
            actual_physical,
            actual_valid,
            layout_q,
            False,
        )
        actual_output = OriginComparator.pack(
            actual.output, actual_physical, actual_valid, layout_q, False
        )
        output = self.compare_tensor(expected_output, actual_output, require_bitwise)
        report = {"pass": output["pass"], "output": output}
        if not compare_lse:
            return report
        if expected.softmax_lse is None or actual.softmax_lse is None:
            report["pass"] = False
            report["softmax_lse"] = {"pass": False, "error": "softmax_lse is missing"}
            return report

        expected_lse = OriginComparator.align_expected(
            expected.softmax_lse,
            expected_origins,
            expected_physical,
            expected_valid,
            actual_origins,
            actual_physical,
            actual_valid,
            layout_q,
            True,
        )
        actual_lse = OriginComparator.pack(
            actual.softmax_lse, actual_physical, actual_valid, layout_q, True
        )
        lse = self.compare_tensor(expected_lse, actual_lse, require_bitwise)
        report["softmax_lse"] = lse
        report["pass"] = report["pass"] and lse["pass"]
        return report
