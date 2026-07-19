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

"""Original-pytest-shaped orchestration for consistency modes 1-4."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch_npu

from .consistency_adapter import CaseAdapter
from .consistency_compare import compare_consistency, compare_subset
from .consistency_model import ConsistencyCase, RunResult
from .consistency_schema import build_baseline_origins, get_q_lengths
from .consistency_transform import (
    transform_mode1_reorder,
    transform_mode2_split,
    transform_mode3_token_partition,
    transform_mode4_shape_change,
)
from .metadata_coverage import CoverageContext, MetadataCoverageAnalyzer
from .metadata_layout import MetadataLayout


class DeterministicLevelError(RuntimeError):
    """Raised when deterministic-level state cannot be read, set, or restored."""


class BatchConsistencyUnsupportedError(RuntimeError):
    """Raised when the installed runtime rejects deterministic level 3."""


class DeterministicLevelGuard:
    """Keep level 3 active across each metadata + main-op execution sequence."""

    LEVEL = 3

    def __init__(self):
        self.previous_level: Optional[int] = None

    @staticmethod
    def current_level() -> int:
        getter = getattr(torch_npu.npu, "_get_deterministic_level", None)
        if getter is None:
            raise DeterministicLevelError("torch_npu.npu._get_deterministic_level is unavailable")
        return int(getter())

    @classmethod
    def set_strict(cls, level: int) -> None:
        torch_npu.npu.set_deterministic_level(level)
        actual = cls.current_level()
        if actual != level:
            raise DeterministicLevelError(
                f"set_deterministic_level({level}) did not take effect; actual={actual}"
            )

    @staticmethod
    def is_runtime_rejection(error: BaseException) -> bool:
        message = str(error).lower()
        return "set_deterministic_level" in message and "only supports configuring 0/1/2" in message

    def __enter__(self) -> "DeterministicLevelGuard":
        self.previous_level = self.current_level()
        try:
            self.set_strict(self.LEVEL)
        except BaseException as error:
            self._restore()
            if self.is_runtime_rejection(error):
                raise BatchConsistencyUnsupportedError(
                    "The installed torch_npu/CANN runtime rejects deterministic level 3."
                ) from error
            raise
        return self

    def _restore(self) -> None:
        if self.previous_level is None:
            return
        current = self.current_level()
        if current != self.previous_level:
            self.set_strict(self.previous_level)

    def __exit__(self, exc_type: Any, error: Optional[BaseException], traceback: Any) -> bool:
        try:
            self._restore()
        except BaseException as restore_error:
            raise DeterministicLevelError(
                f"Failed to restore deterministic level {self.previous_level}: {restore_error}"
            ) from restore_error
        if error is not None and self.is_runtime_rejection(error):
            raise BatchConsistencyUnsupportedError(
                "The installed runtime rejected level 3 on the first NPU operation."
            ) from error
        return False


class ConsistencyFlow:
    """Run transformed PT cases through the operator's ordinary pytest executor."""

    def __init__(self, executor: Callable[[Dict[str, Any]], RunResult]):
        self.executor = executor

    @staticmethod
    def _compare_lse(input_data: Dict[str, Any]) -> bool:
        return bool(input_data["params"].get("return_softmax_lse", False))

    @staticmethod
    def _full_comparison(
        expected: RunResult,
        actual: RunResult,
        input_data: Dict[str, Any],
        case: ConsistencyCase,
        require_bitwise: bool,
    ) -> Dict[str, Any]:
        physical = get_q_lengths(input_data)
        valid = get_q_lengths(input_data, valid_only=True)
        return compare_consistency(
            expected.output,
            build_baseline_origins(input_data, valid_only=True),
            actual.output,
            case.output_origins,
            CaseAdapter(input_data).get_layout_q(),
            physical,
            derived_q_lengths=get_q_lengths(case.input_data),
            compare_lse=ConsistencyFlow._compare_lse(input_data),
            baseline_lse=expected.softmax_lse,
            derived_lse=actual.softmax_lse,
            require_bitwise=require_bitwise,
            baseline_valid_q_lengths=valid,
            derived_valid_q_lengths=get_q_lengths(case.input_data, valid_only=True),
        )

    @staticmethod
    def _subset_comparison(
        expected: RunResult,
        actual: RunResult,
        baseline_data: Dict[str, Any],
        case: ConsistencyCase,
        require_bitwise: bool,
        derived_valid_lengths: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        return compare_subset(
            expected.output,
            build_baseline_origins(baseline_data, valid_only=True),
            get_q_lengths(baseline_data),
            actual.output,
            case.output_origins,
            get_q_lengths(case.input_data),
            CaseAdapter(baseline_data).get_layout_q(),
            compare_lse=ConsistencyFlow._compare_lse(baseline_data),
            baseline_lse=expected.softmax_lse,
            derived_lse=actual.softmax_lse,
            require_bitwise=require_bitwise,
            baseline_valid_q_lengths=get_q_lengths(baseline_data, valid_only=True),
            derived_valid_q_lengths=(
                derived_valid_lengths
                if derived_valid_lengths is not None
                else get_q_lengths(case.input_data, valid_only=True)
            ),
        )

    @staticmethod
    def _cpu_result(input_data: Dict[str, Any]) -> RunResult:
        return RunResult(
            output=input_data["cpu_output"],
            softmax_lse=input_data.get("softmax_lse"),
        )

    @staticmethod
    def _metadata_report(result: RunResult, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if result.metadata is None:
            return None
        parsed = MetadataLayout().parse(result.metadata)
        context = CoverageContext.from_input_data(input_data)
        return MetadataCoverageAnalyzer.analyze(parsed, context=context)

    def _run_case(
        self,
        baseline_data: Dict[str, Any],
        baseline: RunResult,
        case: ConsistencyCase,
        subset: bool,
        derived_valid_lengths: Optional[List[int]] = None,
    ) -> Tuple[RunResult, Dict[str, Any]]:
        derived = self.executor(case.input_data)
        compare = self._subset_comparison if subset else self._full_comparison
        if subset:
            consistency = compare(
                baseline,
                derived,
                baseline_data,
                case,
                True,
                derived_valid_lengths,
            )
            precision = compare(
                self._cpu_result(baseline_data),
                derived,
                baseline_data,
                case,
                False,
                derived_valid_lengths,
            )
        else:
            consistency = compare(baseline, derived, baseline_data, case, True)
            precision = compare(self._cpu_result(baseline_data), derived, baseline_data, case, False)
        coverage = {
            "baseline": self._metadata_report(baseline, baseline_data),
            "derived": self._metadata_report(derived, case.input_data),
        }
        coverage_integrity = all(
            item is None or not item.get("errors") for item in coverage.values()
        )
        report = {
            "case": case.name,
            "transform": case.transform_meta,
            "batch_consistency": consistency,
            "precision": precision,
            "metadata_coverage": coverage,
            "metadata_integrity_pass": coverage_integrity,
            "pass": consistency["pass"] and precision["pass"] and coverage_integrity,
        }
        return derived, report

    def run_mode1(
        self,
        input_data: Dict[str, Any],
        order: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        case = transform_mode1_reorder(input_data, order=order, seed=seed)
        with DeterministicLevelGuard():
            baseline = self.executor(input_data)
            _, report = self._run_case(input_data, baseline, case, subset=False)
        return report

    def run_mode2(
        self, input_data: Dict[str, Any], groups: Sequence[Sequence[int]]
    ) -> Dict[str, Any]:
        normalized = [list(group) for group in groups]
        cases = transform_mode2_split(input_data, groups=normalized)
        with DeterministicLevelGuard():
            baseline = self.executor(input_data)
            reports = [self._run_case(input_data, baseline, case, subset=True)[1] for case in cases]
        return {"mode": "split", "subcases": reports, "pass": all(item["pass"] for item in reports)}

    def run_mode3(
        self, input_data: Dict[str, Any], batch_id: int, split_sizes: Sequence[int]
    ) -> Dict[str, Any]:
        cases = transform_mode3_token_partition(input_data, batch_id, split_sizes)
        with DeterministicLevelGuard():
            baseline = self.executor(input_data)
            reports = [self._run_case(input_data, baseline, case, subset=True)[1] for case in cases]
        return {
            "mode": "token-split",
            "split_sizes": list(split_sizes),
            "subcases": reports,
            "pass": all(item["pass"] for item in reports),
        }

    def run_mode4(
        self,
        input_data: Dict[str, Any],
        batch_id: int,
        common_tokens: int,
        derived_extra_tokens: int,
    ) -> Dict[str, Any]:
        case = transform_mode4_shape_change(
            input_data,
            batch_id,
            common_tokens,
            derived_extra_tokens,
        )
        with DeterministicLevelGuard():
            baseline = self.executor(input_data)
            _, report = self._run_case(
                input_data,
                baseline,
                case,
                subset=True,
                derived_valid_lengths=[common_tokens],
            )
        return report
