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

"""Opt-in SMLA batch-consistency NPU tests."""

import os
from typing import List

import pytest
import torch

from batch_consistency.registry import ConsistencyMode, ModeRegistry
from batch_consistency.adapter import CaseAdapter
from batch_consistency import schema


class ConsistencyTestConfig:
    """Collects environment configuration without module-private helper functions."""

    enabled = os.environ.get("SMLA_CONSISTENCY_TEST", "0") == "1"
    seed = int(os.environ.get("SMLA_CONSISTENCY_SEED", "20260716"))
    randomize = os.environ.get("SMLA_CONSISTENCY_RANDOMIZE", "0") == "1"
    pt_dir = os.getenv("SMLA_PT_LOAD_PATH", "./data")
    single_case_path = os.environ.get("QSAS_TESTCASE_PATH", "").strip()
    modes_value = os.environ.get("SMLA_CONSISTENCY_MODES", "").strip() or None
    profile_value = os.environ.get("SMLA_CONSISTENCY_PROFILE", "").strip() or None

    try:
        modes = ModeRegistry.resolve(
            modes=modes_value,
            profile=profile_value,
            warn_legacy=enabled,
        )
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error

    unsupported_modes = ModeRegistry.unsupported(modes)
    if enabled and unsupported_modes:
        names = ", ".join(mode.value for mode in unsupported_modes)
        raise pytest.UsageError(f"Unsupported consistency modes requested: {names}")

    @classmethod
    def collect_pt_files(cls) -> List[str]:
        files = []
        if cls.single_case_path:
            if os.path.isfile(cls.single_case_path):
                files.append(cls.single_case_path)
            return files
        if not os.path.isdir(cls.pt_dir):
            return files
        return [
            os.path.join(cls.pt_dir, filename)
            for filename in sorted(os.listdir(cls.pt_dir))
            if filename.endswith(".pt")
        ]

    @staticmethod
    def filter_by_batch_size(files: List[str], minimum: int) -> List[str]:
        selected = []
        for path in files:
            try:
                data = torch.load(path, map_location="cpu")
                adapter = CaseAdapter(data)
                if adapter.name == "smla" and adapter.get_batch_size() >= minimum:
                    selected.append(path)
            except Exception as error:
                if ConsistencyTestConfig.enabled:
                    raise pytest.UsageError(f"Cannot inspect PT case {path}: {error}") from error
        return selected

    @staticmethod
    def filter_mode3(files: List[str]) -> List[str]:
        selected = []
        for path in files:
            try:
                data = torch.load(path, map_location="cpu")
                adapter = CaseAdapter(data)
                if adapter.name != "smla":
                    continue
                if max(schema.get_q_lengths(data, valid_only=True), default=0) > 1:
                    selected.append(path)
            except Exception as error:
                if ConsistencyTestConfig.enabled:
                    raise pytest.UsageError(f"Cannot inspect PT case {path}: {error}") from error
        return selected

    @staticmethod
    def set_npu_device() -> None:
        import torch_npu

        torch_npu.npu.set_device(int(os.environ.get("SMLA_DEVICE_ID", "0")))


TESTCASE_FILES = ConsistencyTestConfig.filter_by_batch_size(
    ConsistencyTestConfig.collect_pt_files(),
    1,
)
BATCH_TESTCASE_FILES = ConsistencyTestConfig.filter_by_batch_size(TESTCASE_FILES, 2)
MODE3_TESTCASE_FILES = ConsistencyTestConfig.filter_mode3(TESTCASE_FILES)

pytestmark = [
    pytest.mark.consistency,
    pytest.mark.skipif(
        not ConsistencyTestConfig.enabled,
        reason="SMLA_CONSISTENCY_TEST is not set to 1; consistency tests are opt-in",
    ),
    pytest.mark.skipif(
        not TESTCASE_FILES,
        reason=f"No .pt files found in {ConsistencyTestConfig.pt_dir}; run save mode first",
    ),
]


@pytest.mark.consistency
@pytest.mark.parametrize("testcase_file", BATCH_TESTCASE_FILES)
def test_mode1_batch_reorder(testcase_file):
    if ConsistencyMode.REORDER not in ConsistencyTestConfig.modes:
        pytest.skip("reorder not requested")
    from batch_consistency.runner import BatchConsistencyUnsupportedError, run_mode1

    test_data = torch.load(testcase_file, map_location="cpu")
    ConsistencyTestConfig.set_npu_device()
    try:
        case, _, _, comparison = run_mode1(test_data, seed=ConsistencyTestConfig.seed)
    except BatchConsistencyUnsupportedError as error:
        pytest.skip(str(error))

    print(f"\n[reorder] file={os.path.basename(testcase_file)}, order={case.transform_meta['order']}")
    if not comparison["pass"]:
        pytest.fail(f"reorder failed: {comparison}")


@pytest.mark.consistency
@pytest.mark.parametrize("testcase_file", BATCH_TESTCASE_FILES)
def test_mode2_batch_split(testcase_file):
    if ConsistencyMode.SPLIT not in ConsistencyTestConfig.modes:
        pytest.skip("split not requested")
    from batch_consistency.runner import BatchConsistencyUnsupportedError, run_mode2

    test_data = torch.load(testcase_file, map_location="cpu")
    ConsistencyTestConfig.set_npu_device()
    try:
        cases, _, _, comparisons, aggregate = run_mode2(
            test_data,
            seed=ConsistencyTestConfig.seed,
            randomize=ConsistencyTestConfig.randomize,
        )
    except BatchConsistencyUnsupportedError as error:
        pytest.skip(str(error))

    print(f"\n[split] file={os.path.basename(testcase_file)}")
    for case, comparison in zip(cases, comparisons):
        print(
            f"  batches={case.transform_meta['batch_ids']}, "
            f"bitwise={comparison['output']['bitwise_equal']}, "
            f"numeric={comparison['output']['fulfill_percent']:.2f}%"
        )
    if aggregate["merged"] is not None:
        print(f"  optional full-partition merge pass={aggregate['merged']['pass']}")
    if not aggregate["pass"]:
        pytest.fail(f"split failed: {aggregate}")


@pytest.mark.consistency
@pytest.mark.parametrize("testcase_file", MODE3_TESTCASE_FILES)
def test_mode3_batch_internal_token_split(testcase_file):
    if ConsistencyMode.TOKEN_SPLIT not in ConsistencyTestConfig.modes:
        pytest.skip("token-split not requested")
    from batch_consistency.runner import BatchConsistencyUnsupportedError, run_mode3
    from batch_consistency.transform import InvalidTransformError

    test_data = torch.load(testcase_file, map_location="cpu")
    ConsistencyTestConfig.set_npu_device()
    try:
        case, _, _, comparison = run_mode3(test_data)
    except InvalidTransformError as error:
        pytest.skip(f"INVALID_TRANSFORM: {error}")
    except BatchConsistencyUnsupportedError as error:
        pytest.skip(str(error))

    print(
        f"\n[token-split] file={os.path.basename(testcase_file)}, "
        f"selection={case.transform_meta['token_ids_by_batch']}"
    )
    if not comparison["pass"]:
        pytest.fail(f"token-split failed: {comparison}")
