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

"""Execution orchestration for SMLA batch-consistency modes."""

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch_npu

from . import schema
from .adapter import CaseAdapter
from .model import RunResult


_BATCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BATCH_DIR not in sys.path:
    sys.path.insert(0, _BATCH_DIR)
from batch import sparse_flash_mla_process  # noqa: E402


class DeterministicLevelError(RuntimeError):
    """Raised when deterministic-level state cannot be read, set, or restored."""


class BatchConsistencyUnsupportedError(RuntimeError):
    """Raised when the installed torch_npu/CANN runtime rejects level 3."""


class DeterministicLevelGuard:
    """Sets level 3 for one execution and restores the real prior level."""

    BATCH_CONSISTENCY_LEVEL = 3

    def __init__(self):
        self.previous_level: Optional[int] = None

    @staticmethod
    def current_level() -> int:
        get_level = getattr(torch_npu.npu, "_get_deterministic_level", None)
        if get_level is None:
            raise DeterministicLevelError(
                "torch_npu.npu._get_deterministic_level is unavailable"
            )
        return int(get_level())

    @classmethod
    def set_strict(cls, level: int) -> None:
        torch_npu.npu.set_deterministic_level(level)
        actual = cls.current_level()
        if actual != level:
            raise DeterministicLevelError(
                f"set_deterministic_level({level}) did not take effect; actual={actual}"
            )

    @staticmethod
    def is_level3_unsupported(error: BaseException) -> bool:
        message = str(error).lower()
        return "set_deterministic_level" in message and "only supports configuring 0/1/2" in message

    def __enter__(self) -> "DeterministicLevelGuard":
        self.previous_level = self.current_level()
        try:
            self.set_strict(self.BATCH_CONSISTENCY_LEVEL)
        except BaseException as error:
            self._restore_previous()
            if self.is_level3_unsupported(error):
                raise BatchConsistencyUnsupportedError(
                    "The installed torch_npu/CANN runtime rejects deterministic level 3 "
                    "(only levels 0/1/2 are supported)."
                ) from error
            raise
        return self

    def _restore_previous(self) -> None:
        current = self.current_level()
        if self.previous_level is not None and current != self.previous_level:
            self.set_strict(self.previous_level)

    def __exit__(self, exc_type: Any, error: Optional[BaseException], traceback: Any) -> bool:
        restore_error = None
        try:
            self._restore_previous()
        except BaseException as caught_error:
            restore_error = caught_error

        if restore_error is not None:
            raise DeterministicLevelError(
                f"Failed to restore deterministic level {self.previous_level}: {restore_error}"
            ) from restore_error
        if error is not None and self.is_level3_unsupported(error):
            raise BatchConsistencyUnsupportedError(
                "The installed torch_npu/CANN runtime rejects deterministic level 3 on the first NPU "
                "operation (only levels 0/1/2 are supported)."
            ) from error
        return False


class SmlaExecutor:
    """Thin adapter preserving the legacy SMLA call_npu tuple contract."""

    def __call__(self, input_data: Dict[str, Any]) -> RunResult:
        captured_metadata = []
        output, softmax_lse = sparse_flash_mla_process.call_npu(
            input_data,
            metadata_sink=captured_metadata.append,
        )
        metadata = captured_metadata[0] if captured_metadata else None
        return RunResult(output=output, softmax_lse=softmax_lse, metadata=metadata)


class ConsistencyRunner:
    """Runs transforms with an injectable operator executor."""

    def __init__(self, executor: Optional[Callable[[Dict[str, Any]], RunResult]] = None):
        self.executor = executor or SmlaExecutor()

    @staticmethod
    def _validate_case(input_data: Dict[str, Any]) -> None:
        schema.validate_schema(input_data)
        schema.check_invariants(input_data)

    def execute(self, input_data: Dict[str, Any], validate: bool = True) -> RunResult:
        if validate:
            self._validate_case(input_data)
        with DeterministicLevelGuard():
            return self.executor(input_data)

    def run_baseline(self, input_data: Dict[str, Any], validate: bool = True) -> Tuple[RunResult, List[Any]]:
        if validate:
            self._validate_case(input_data)
        result = self.execute(input_data, validate=False)
        return result, schema.build_baseline_origins(input_data, valid_only=True)

    def run_mode1(
        self,
        input_data: Dict[str, Any],
        order: Optional[List[int]] = None,
        seed: Optional[int] = None,
        validate: bool = True,
    ) -> Tuple[Any, RunResult, RunResult, Dict[str, Any]]:
        from .compare import compare_consistency
        from .transform import transform_mode1_reorder

        if validate:
            self._validate_case(input_data)
        case = transform_mode1_reorder(input_data, order=order, seed=seed)
        baseline_result, baseline_origins = self.run_baseline(input_data, validate=False)
        baseline_lengths = schema.get_q_lengths(input_data)
        baseline_valid_lengths = schema.get_q_lengths(input_data, valid_only=True)
        derived_result = self.execute(case.input_data, validate=False)
        derived_lengths = schema.get_q_lengths(case.input_data)
        derived_valid_lengths = schema.get_q_lengths(case.input_data, valid_only=True)
        compare_lse = bool(input_data["params"].get("return_softmax_lse", False))
        comparison = compare_consistency(
            baseline_result.output,
            baseline_origins,
            derived_result.output,
            case.output_origins,
            CaseAdapter(input_data).get_layout_q(),
            baseline_lengths,
            derived_q_lengths=derived_lengths,
            compare_lse=compare_lse,
            baseline_lse=baseline_result.softmax_lse,
            derived_lse=derived_result.softmax_lse,
            require_bitwise=True,
            baseline_valid_q_lengths=baseline_valid_lengths,
            derived_valid_q_lengths=derived_valid_lengths,
        )
        return case, baseline_result, derived_result, comparison

    def run_mode2(
        self,
        input_data: Dict[str, Any],
        groups: Optional[List[List[int]]] = None,
        seed: Optional[int] = None,
        validate: bool = True,
        randomize: bool = False,
        merge_full_partition: bool = True,
    ) -> Tuple[List[Any], RunResult, List[Any], List[Any], Dict[str, Any]]:
        from .compare import compare_consistency, compare_subset, merge_split_lse, merge_split_outputs
        from .transform import transform_mode2_split

        if validate:
            self._validate_case(input_data)
        cases = transform_mode2_split(input_data, groups=groups, seed=seed, randomize=randomize)
        baseline_result, baseline_origins = self.run_baseline(input_data, validate=False)
        baseline_lengths = schema.get_q_lengths(input_data)
        baseline_valid_lengths = schema.get_q_lengths(input_data, valid_only=True)
        layout_q = CaseAdapter(input_data).get_layout_q()
        compare_lse = bool(input_data["params"].get("return_softmax_lse", False))
        derived_results = []
        sub_comparisons = []
        for case in cases:
            result = self.execute(case.input_data, validate=False)
            derived_results.append((result, case))
            comparison = compare_subset(
                baseline_result.output,
                baseline_origins,
                baseline_lengths,
                result.output,
                case.output_origins,
                schema.get_q_lengths(case.input_data),
                layout_q,
                compare_lse=compare_lse,
                baseline_lse=baseline_result.softmax_lse,
                derived_lse=result.softmax_lse,
                require_bitwise=True,
                baseline_valid_q_lengths=baseline_valid_lengths,
                derived_valid_q_lengths=schema.get_q_lengths(case.input_data, valid_only=True),
            )
            sub_comparisons.append({"case": case.name, "batch_ids": case.transform_meta["batch_ids"], **comparison})

        aggregate: Dict[str, Any] = {
            "pass": all(comparison["pass"] for comparison in sub_comparisons),
            "subcases": sub_comparisons,
            "merged": None,
        }
        complete_partition = bool(cases and cases[0].transform_meta["complete_partition"])
        if merge_full_partition and complete_partition:
            merged_output, merged_origins = merge_split_outputs(
                [(result.output, case.output_origins, case.transform_meta) for result, case in derived_results],
                layout_q,
                baseline_lengths,
                baseline_valid_lengths,
            )
            merged_lse = None
            if compare_lse:
                merged_lse, _ = merge_split_lse(
                    [
                        (result.softmax_lse, case.output_origins, case.transform_meta)
                        for result, case in derived_results
                    ],
                    layout_q,
                    baseline_lengths,
                    baseline_valid_lengths,
                )
            aggregate["merged"] = compare_consistency(
                baseline_result.output,
                baseline_origins,
                merged_output,
                merged_origins,
                layout_q,
                baseline_lengths,
                derived_q_lengths=baseline_lengths,
                compare_lse=compare_lse,
                baseline_lse=baseline_result.softmax_lse,
                derived_lse=merged_lse,
                require_bitwise=True,
                baseline_valid_q_lengths=baseline_valid_lengths,
                derived_valid_q_lengths=baseline_valid_lengths,
            )
            aggregate["pass"] = aggregate["pass"] and aggregate["merged"]["pass"]
        return cases, baseline_result, derived_results, sub_comparisons, aggregate

    def run_mode3(
        self,
        input_data: Dict[str, Any],
        token_ids_by_batch: Optional[List[List[int]]] = None,
        batch_ids: Optional[List[int]] = None,
        validate: bool = True,
    ) -> Tuple[Any, RunResult, RunResult, Dict[str, Any]]:
        from .compare import compare_subset
        from .transform import transform_mode3_token_split

        if validate:
            self._validate_case(input_data)
        case = transform_mode3_token_split(
            input_data,
            token_ids_by_batch=token_ids_by_batch,
            batch_ids=batch_ids,
            validate_semantics=True,
        )
        baseline_result, baseline_origins = self.run_baseline(input_data, validate=False)
        baseline_lengths = schema.get_q_lengths(input_data)
        baseline_valid_lengths = schema.get_q_lengths(input_data, valid_only=True)
        derived_result = self.execute(case.input_data, validate=False)
        compare_lse = bool(input_data["params"].get("return_softmax_lse", False))
        comparison = compare_subset(
            baseline_result.output,
            baseline_origins,
            baseline_lengths,
            derived_result.output,
            case.output_origins,
            schema.get_q_lengths(case.input_data),
            CaseAdapter(input_data).get_layout_q(),
            compare_lse=compare_lse,
            baseline_lse=baseline_result.softmax_lse,
            derived_lse=derived_result.softmax_lse,
            require_bitwise=True,
            baseline_valid_q_lengths=baseline_valid_lengths,
            derived_valid_q_lengths=schema.get_q_lengths(case.input_data, valid_only=True),
        )
        return case, baseline_result, derived_result, comparison


def run_with_batch_consistency(input_data: Dict[str, Any], validate: bool = True) -> RunResult:
    return ConsistencyRunner().execute(input_data, validate=validate)


def run_baseline(input_data: Dict[str, Any], validate: bool = True) -> Tuple[RunResult, List[Any]]:
    return ConsistencyRunner().run_baseline(input_data, validate=validate)


def run_mode1(
    input_data: Dict[str, Any],
    order: Optional[List[int]] = None,
    seed: Optional[int] = None,
    validate: bool = True,
) -> Tuple[Any, RunResult, RunResult, Dict[str, Any]]:
    return ConsistencyRunner().run_mode1(input_data, order=order, seed=seed, validate=validate)


def run_mode2(
    input_data: Dict[str, Any],
    groups: Optional[List[List[int]]] = None,
    seed: Optional[int] = None,
    validate: bool = True,
    randomize: bool = False,
    merge_full_partition: bool = True,
) -> Tuple[List[Any], RunResult, List[Any], List[Any], Dict[str, Any]]:
    return ConsistencyRunner().run_mode2(
        input_data,
        groups=groups,
        seed=seed,
        validate=validate,
        randomize=randomize,
        merge_full_partition=merge_full_partition,
    )


def run_mode3(
    input_data: Dict[str, Any],
    token_ids_by_batch: Optional[List[List[int]]] = None,
    batch_ids: Optional[List[int]] = None,
    validate: bool = True,
) -> Tuple[Any, RunResult, RunResult, Dict[str, Any]]:
    return ConsistencyRunner().run_mode3(
        input_data,
        token_ids_by_batch=token_ids_by_batch,
        batch_ids=batch_ids,
        validate=validate,
    )
