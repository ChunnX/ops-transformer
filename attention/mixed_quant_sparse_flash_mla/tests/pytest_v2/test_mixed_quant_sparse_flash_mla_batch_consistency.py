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

"""Opt-in MQSMLA consistency tests with per-PT config and structured results."""

from functools import lru_cache
import os
from typing import Any, Callable, Dict, List, Optional

import pytest
import torch

from batch.consistency.adapter import CaseAdapter
from batch.consistency.config import ConsistencyMode
from batch.consistency.flow import BatchConsistencyUnsupportedError
from batch.consistency.reporting import attach_provenance
from batch.consistency.result_artifact import (
    ResultArtifactError,
    ResultStatus,
    write_result_artifact,
)
from batch.consistency.runtime_capability import (
    ERROR as CAPABILITY_ERROR,
    UNSUPPORTED as CAPABILITY_UNSUPPORTED,
    RuntimeCapabilityError,
    is_capability_configuration_error,
    load_runtime_capability_evidence,
    require_runtime_capability_evidence,
)
from batch.consistency.schema import get_q_lengths
from batch.consistency.suite_config import (
    ResolvedCaseConfig,
    SuiteConfigError,
    resolve_runtime_case_config,
)
from batch.consistency.transform import (
    InvalidTransformError,
    transform_mode1_reorder,
    transform_mode2_split,
    transform_mode3_token_partition,
    transform_mode4_shape_change,
)


OPERATOR = "mqsmla"
ENV_PREFIX = "MQSMLA_CONSISTENCY"
_CONFIGURED_CAPABILITY = object()


class ConsistencyTestConfig:
    enabled = os.environ.get("MQSMLA_CONSISTENCY_TEST", "0") == "1"
    pt_dir = os.environ.get("MQSMLA_PT_DIR", "mqsmla_testcase")
    single_case = os.environ.get("QSAS_TESTCASE_PATH", "").strip()
    report_dir = os.path.abspath(
        os.environ.get("MQSMLA_CONSISTENCY_REPORT_DIR", "./result/mqsmla_consistency")
    )
    capability_report = os.environ.get("MQSMLA_CONSISTENCY_CAPABILITY_REPORT", "").strip()

    @classmethod
    def collect_pt_files(cls) -> List[str]:
        if cls.single_case:
            return [cls.single_case] if os.path.isfile(cls.single_case) else []
        if not os.path.isdir(cls.pt_dir):
            return []
        return [
            os.path.join(cls.pt_dir, name)
            for name in sorted(os.listdir(cls.pt_dir))
            if name.endswith(".pt")
        ]

    @staticmethod
    def device_id() -> int:
        return int(os.environ.get("MQSMLA_DEVICE_ID", "0"))

    @classmethod
    def set_device(cls) -> None:
        import torch_npu

        torch_npu.npu.set_device(cls.device_id())


TESTCASE_FILES = ConsistencyTestConfig.collect_pt_files()

pytestmark = [
    pytest.mark.consistency,
    pytest.mark.skipif(
        not ConsistencyTestConfig.enabled,
        reason="MQSMLA_CONSISTENCY_TEST is not 1; consistency tests are opt-in",
    ),
    pytest.mark.skipif(
        not TESTCASE_FILES,
        reason=f"No MQSMLA PT cases found in {ConsistencyTestConfig.pt_dir}",
    ),
]


@lru_cache(maxsize=1)
def runtime_capability() -> Optional[Dict]:
    if not ConsistencyTestConfig.capability_report:
        return None
    return load_runtime_capability_evidence(ConsistencyTestConfig.capability_report)


def create_flow():
    from batch.mixed_quant_sparse_flash_mla_consistency import create_consistency_flow

    return create_consistency_flow()


def load_mqsmla_case(path: str) -> Dict:
    data = torch.load(path, map_location="cpu")
    attach_provenance(data, source_pt=path)
    return data


def record_result(
    data: Dict,
    path: str,
    mode: ConsistencyMode,
    status: ResultStatus,
    *,
    config: Optional[ResolvedCaseConfig] = None,
    report: Optional[Dict] = None,
    reason_code: Optional[str] = None,
    reason: Optional[str] = None,
    runtime_capability_override: Any = _CONFIGURED_CAPABILITY,
) -> str:
    capability = (
        runtime_capability()
        if runtime_capability_override is _CONFIGURED_CAPABILITY
        else runtime_capability_override
    )
    return write_result_artifact(
        ConsistencyTestConfig.report_dir,
        operator=OPERATOR,
        input_data=data,
        pt_path=path,
        mode=mode.value,
        status=status,
        config=config,
        report=report,
        reason_code=reason_code,
        reason=reason,
        runtime_capability=capability,
    )


def skip_with_result(
    data: Dict,
    path: str,
    mode: ConsistencyMode,
    status: ResultStatus,
    reason_code: str,
    reason: str,
    config: Optional[ResolvedCaseConfig] = None,
) -> None:
    record_result(
        data,
        path,
        mode,
        status,
        config=config,
        reason_code=reason_code,
        reason=reason,
    )
    pytest.skip(f"{status.value}: {reason}")


def fail_config(
    data: Dict,
    path: str,
    mode: ConsistencyMode,
    reason: str,
    config: Optional[ResolvedCaseConfig] = None,
) -> None:
    record_result(
        data,
        path,
        mode,
        ResultStatus.CONFIG_ERROR,
        config=config,
        reason_code="INVALID_CASE_CONFIG",
        reason=reason,
    )
    pytest.fail(f"CONFIG_ERROR: {reason}")


def prepare_mode(
    path: str, mode: ConsistencyMode
) -> tuple[Dict, ResolvedCaseConfig]:
    data = load_mqsmla_case(path)
    detected = CaseAdapter(data).name
    if detected != OPERATOR:
        skip_with_result(
            data,
            path,
            mode,
            ResultStatus.NOT_APPLICABLE,
            "OPERATOR_MISMATCH",
            f"PT operator is {detected}, expected {OPERATOR}",
        )
    try:
        config = resolve_runtime_case_config(data, path, ENV_PREFIX)
    except SuiteConfigError as error:
        fail_config(data, path, mode, str(error))
    if mode not in config.modes:
        skip_with_result(
            data,
            path,
            mode,
            ResultStatus.NOT_APPLICABLE,
            "MODE_NOT_SELECTED",
            f"{mode.value} is not selected by config {config.config_id}",
            config,
        )
    return data, config


def execute_mode(
    data: Dict,
    path: str,
    mode: ConsistencyMode,
    config: ResolvedCaseConfig,
    precheck: Callable[[], Any],
    run: Callable[[], Dict],
) -> None:
    try:
        precheck()
    except InvalidTransformError as error:
        skip_with_result(
            data,
            path,
            mode,
            ResultStatus.INVALID_TRANSFORM,
            "SEMANTIC_TRANSFORM_REJECTED",
            str(error),
            config,
        )
    except Exception as error:
        record_result(
            data,
            path,
            mode,
            ResultStatus.FAIL,
            config=config,
            reason_code="PRECHECK_EXCEPTION",
            reason=f"{type(error).__name__}: {error}",
        )
        raise
    try:
        capability = require_runtime_capability_evidence(
            ConsistencyTestConfig.capability_report,
            ConsistencyTestConfig.device_id(),
        )
    except RuntimeCapabilityError as error:
        reason_code = (
            "MISSING_RUNTIME_CAPABILITY"
            if not ConsistencyTestConfig.capability_report
            else "INVALID_RUNTIME_CAPABILITY"
        )
        record_result(
            data,
            path,
            mode,
            ResultStatus.CONFIG_ERROR,
            config=config,
            reason_code=reason_code,
            reason=str(error),
            runtime_capability_override=None,
        )
        pytest.fail(f"CONFIG_ERROR: {error}")
    probe = capability["probe"]
    if probe["status"] == CAPABILITY_UNSUPPORTED:
        skip_with_result(
            data,
            path,
            mode,
            ResultStatus.UNSUPPORTED,
            probe["reason_code"],
            probe["reason"],
            config,
        )
    if probe["status"] == CAPABILITY_ERROR:
        result_status = (
            ResultStatus.CONFIG_ERROR
            if is_capability_configuration_error(probe)
            else ResultStatus.FAIL
        )
        record_result(
            data,
            path,
            mode,
            result_status,
            config=config,
            reason_code=probe["reason_code"],
            reason=probe["reason"],
        )
        pytest.fail(
            f"{result_status.value}: runtime capability probe failed: {probe['reason']}"
        )
    try:
        ConsistencyTestConfig.set_device()
        report = run()
    except InvalidTransformError as error:
        skip_with_result(
            data,
            path,
            mode,
            ResultStatus.INVALID_TRANSFORM,
            "SEMANTIC_TRANSFORM_REJECTED",
            str(error),
            config,
        )
    except BatchConsistencyUnsupportedError as error:
        skip_with_result(
            data,
            path,
            mode,
            ResultStatus.UNSUPPORTED,
            error.reason_code,
            str(error),
            config,
        )
    except Exception as error:
        record_result(
            data,
            path,
            mode,
            ResultStatus.FAIL,
            config=config,
            reason_code="EXECUTION_EXCEPTION",
            reason=f"{type(error).__name__}: {error}",
        )
        raise

    print(report)
    passed = isinstance(report, dict) and report.get("pass") is True
    if not passed:
        record_result(
            data,
            path,
            mode,
            ResultStatus.FAIL,
            config=config,
            report=report if isinstance(report, dict) else None,
            reason_code="COMPARISON_FAILED",
            reason="Consistency or CPU precision comparison failed",
        )
        pytest.fail(f"FAIL: {report}")
    try:
        record_result(
            data,
            path,
            mode,
            ResultStatus.PASS,
            config=config,
            report=report,
        )
    except ResultArtifactError as error:
        record_result(
            data,
            path,
            mode,
            ResultStatus.FAIL,
            config=config,
            report=report,
            reason_code="PASS_GATE_REJECTED",
            reason=str(error),
        )
        pytest.fail(f"FAIL: {error}")


@pytest.mark.parametrize("testcase_file", TESTCASE_FILES)
def test_mode1_reorder(testcase_file):
    mode = ConsistencyMode.REORDER
    data, config = prepare_mode(testcase_file, mode)
    if CaseAdapter(data).get_batch_size() < 2:
        skip_with_result(
            data,
            testcase_file,
            mode,
            ResultStatus.NOT_APPLICABLE,
            "BATCH_TOO_SMALL",
            "reorder requires B >= 2",
            config,
        )
    order = list(config.reorder_order) if config.reorder_order is not None else None
    execute_mode(
        data,
        testcase_file,
        mode,
        config,
        lambda: transform_mode1_reorder(data, order=order, seed=config.seed),
        lambda: create_flow().run_mode1(
            data, order=order, seed=config.seed
        ),
    )


@pytest.mark.parametrize("testcase_file", TESTCASE_FILES)
def test_mode2_split(testcase_file):
    mode = ConsistencyMode.SPLIT
    data, config = prepare_mode(testcase_file, mode)
    if CaseAdapter(data).get_batch_size() < 2:
        skip_with_result(
            data,
            testcase_file,
            mode,
            ResultStatus.NOT_APPLICABLE,
            "BATCH_TOO_SMALL",
            "split requires B >= 2",
            config,
        )
    if config.split_groups is None:
        fail_config(data, testcase_file, mode, "split_groups were not resolved", config)
    groups = [list(group) for group in config.split_groups]
    execute_mode(
        data,
        testcase_file,
        mode,
        config,
        lambda: transform_mode2_split(data, groups=groups),
        lambda: create_flow().run_mode2(data, groups),
    )


@pytest.mark.parametrize("testcase_file", TESTCASE_FILES)
def test_mode3_token_split(testcase_file):
    mode = ConsistencyMode.TOKEN_SPLIT
    data, config = prepare_mode(testcase_file, mode)
    token_count = get_q_lengths(data, valid_only=True)[config.mode_batch]
    if token_count < 2:
        skip_with_result(
            data,
            testcase_file,
            mode,
            ResultStatus.NOT_APPLICABLE,
            "QUERY_TOO_SHORT",
            "token-split requires at least two valid Q tokens",
            config,
        )
    if config.token_split_sizes is None:
        fail_config(
            data, testcase_file, mode, "token_split_sizes were not resolved", config
        )
    execute_mode(
        data,
        testcase_file,
        mode,
        config,
        lambda: transform_mode3_token_partition(
            data, config.mode_batch, config.token_split_sizes
        ),
        lambda: create_flow().run_mode3(
            data, config.mode_batch, config.token_split_sizes
        ),
    )


@pytest.mark.parametrize("testcase_file", TESTCASE_FILES)
def test_mode4_shape_change(testcase_file):
    mode = ConsistencyMode.SHAPE_CHANGE
    data, config = prepare_mode(testcase_file, mode)
    token_count = get_q_lengths(data, valid_only=True)[config.mode_batch]
    if token_count < 1:
        skip_with_result(
            data,
            testcase_file,
            mode,
            ResultStatus.NOT_APPLICABLE,
            "EMPTY_QUERY",
            "shape-change requires at least one valid Q token",
            config,
        )
    if config.common_tokens is None or config.derived_extra is None:
        fail_config(
            data,
            testcase_file,
            mode,
            "shape-change values were not resolved",
            config,
        )
    execute_mode(
        data,
        testcase_file,
        mode,
        config,
        lambda: transform_mode4_shape_change(
            data,
            config.mode_batch,
            config.common_tokens,
            config.derived_extra,
        ),
        lambda: create_flow().run_mode4(
            data,
            config.mode_batch,
            config.common_tokens,
            config.derived_extra,
        ),
    )
