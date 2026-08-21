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

"""Execute the four black-box relations through an operator's normal pytest path."""

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch_npu

from .adapter import CaseAdapter
from .compare import ResultComparator
from .model import ConsistencyCase, RunResult
from .schema import build_baseline_origins, check_invariants, get_q_lengths, validate_schema
from .transform import (
    transform_mode1_reorder,
    transform_mode2_split,
    transform_mode3_token_partition,
    transform_mode4_shape_change,
)


class BatchConsistencyUnsupportedError(RuntimeError):
    """The installed runtime cannot execute deterministic level 3."""


class DeterministicLevelGuard:
    """Keep level 3 active across metadata and main-operator calls."""

    LEVEL = 3

    def __init__(self):
        self.previous_level: Optional[int] = None

    @staticmethod
    def current_level() -> int:
        getter = getattr(torch_npu.npu, "_get_deterministic_level", None)
        if getter is None:
            raise BatchConsistencyUnsupportedError(
                "UNSUPPORTED: torch_npu does not expose deterministic-level state"
            )
        return int(getter())

    @staticmethod
    def runtime_rejects_level3(error: BaseException) -> bool:
        message = str(error).lower()
        return "deterministic" in message and (
            "0/1/2" in message or "level 3" in message or "level3" in message
        )

    @classmethod
    def set_level(cls, level: int) -> None:
        setter = getattr(torch_npu.npu, "set_deterministic_level", None)
        if setter is None:
            raise BatchConsistencyUnsupportedError(
                "UNSUPPORTED: torch_npu does not expose set_deterministic_level"
            )
        setter(level)
        actual = cls.current_level()
        if actual != level:
            raise RuntimeError(
                f"set_deterministic_level({level}) did not take effect; actual={actual}"
            )

    def restore(self) -> None:
        if self.previous_level is not None and self.current_level() != self.previous_level:
            self.set_level(self.previous_level)

    def __enter__(self):
        self.previous_level = self.current_level()
        try:
            self.set_level(self.LEVEL)
        except BaseException as error:
            self.restore()
            if self.runtime_rejects_level3(error):
                raise BatchConsistencyUnsupportedError(
                    f"UNSUPPORTED: runtime rejected deterministic level 3: {error}"
                ) from error
            raise
        return self

    def __exit__(self, exc_type, error, traceback) -> bool:
        try:
            self.restore()
        except BaseException as restore_error:
            raise RuntimeError(
                f"failed to restore deterministic level {self.previous_level}: {restore_error}"
            ) from restore_error
        if error is not None and self.runtime_rejects_level3(error):
            raise BatchConsistencyUnsupportedError(
                f"UNSUPPORTED: NPU dispatch rejected deterministic level 3: {error}"
            ) from error
        return False


class ConsistencyRunner:
    """Run transformed cases with an injected SMLA or MQSMLA executor."""

    def __init__(
        self,
        executor: Callable[[Dict[str, Any]], RunResult],
        precision_compare: Callable[[Any, Any], Any],
        device_setup: Callable[[], None],
    ):
        self.executor = executor
        self.comparator = ResultComparator(precision_compare)
        self.device_setup = device_setup

    @staticmethod
    def cpu_result(input_data: Dict[str, Any]) -> RunResult:
        softmax_lse = input_data.get("softmax_lse")
        if softmax_lse is None:
            softmax_lse = input_data.get("cpu_lse")
        return RunResult(input_data["cpu_output"], softmax_lse)

    @staticmethod
    def lengths(input_data: Dict[str, Any]) -> tuple[List[int], List[int]]:
        return get_q_lengths(input_data), get_q_lengths(input_data, valid_only=True)

    def compare_case(
        self,
        baseline_data: Dict[str, Any],
        baseline: RunResult,
        baseline_origins: Sequence[Any],
        case: ConsistencyCase,
        derived: RunResult,
        derived_valid_override: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        layout_q = CaseAdapter(baseline_data).get_layout_q()
        baseline_physical, baseline_valid = self.lengths(baseline_data)
        derived_physical, derived_valid = self.lengths(case.input_data)
        if derived_valid_override is not None:
            derived_valid = derived_valid_override
        compare_lse = bool(baseline_data["params"].get("return_softmax_lse", False))
        consistency = self.comparator.compare_results(
            baseline,
            baseline_origins,
            baseline_physical,
            baseline_valid,
            derived,
            case.output_origins,
            derived_physical,
            derived_valid,
            layout_q,
            compare_lse,
            True,
        )
        precision = self.comparator.compare_results(
            self.cpu_result(baseline_data),
            baseline_origins,
            baseline_physical,
            baseline_valid,
            derived,
            case.output_origins,
            derived_physical,
            derived_valid,
            layout_q,
            compare_lse,
            False,
        )
        return {
            "case": case.name,
            "transform": case.transform_meta,
            "batch_consistency": consistency,
            "precision": precision,
            "pass": consistency["pass"] and precision["pass"],
        }

    def run_cases(
        self,
        input_data: Dict[str, Any],
        cases: Sequence[ConsistencyCase],
        derived_valid_overrides: Optional[Sequence[Optional[List[int]]]] = None,
    ) -> Dict[str, Any]:
        validate_schema(input_data)
        check_invariants(input_data)
        overrides = list(derived_valid_overrides or [None] * len(cases))
        if len(overrides) != len(cases):
            raise ValueError("derived comparison scopes do not match transformed cases")

        self.device_setup()
        with DeterministicLevelGuard():
            baseline = self.executor(input_data)
            origins = build_baseline_origins(input_data, valid_only=True)
            reports = [
                self.compare_case(
                    input_data,
                    baseline,
                    origins,
                    case,
                    self.executor(case.input_data),
                    override,
                )
                for case, override in zip(cases, overrides)
            ]
        return {
            "deterministic_level": 3,
            "relations": reports,
            "pass": bool(reports) and all(report["pass"] for report in reports),
        }

    def run_mode1(
        self, input_data: Dict[str, Any], order: Optional[List[int]], seed: int
    ) -> Dict[str, Any]:
        return self.run_cases(input_data, [transform_mode1_reorder(input_data, order, seed)])

    def run_mode2(
        self, input_data: Dict[str, Any], groups: List[List[int]], seed: int
    ) -> Dict[str, Any]:
        cases = transform_mode2_split(input_data, groups, seed, False)
        return self.run_cases(input_data, cases)

    def run_mode3(
        self, input_data: Dict[str, Any], batch_id: int, split_sizes: Sequence[int]
    ) -> Dict[str, Any]:
        return self.run_cases(
            input_data,
            transform_mode3_token_partition(input_data, batch_id, split_sizes),
        )

    def run_mode4(
        self,
        input_data: Dict[str, Any],
        batch_id: int,
        common_tokens: int,
        derived_extra_tokens: int,
    ) -> Dict[str, Any]:
        case = transform_mode4_shape_change(
            input_data, batch_id, common_tokens, derived_extra_tokens
        )
        return self.run_cases(input_data, [case], [[common_tokens]])
