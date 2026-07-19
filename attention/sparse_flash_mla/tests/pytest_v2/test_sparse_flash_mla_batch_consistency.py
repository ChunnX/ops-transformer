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

"""Opt-in SMLA consistency tests integrated with the original PT batch flow."""

import os
from typing import Dict, List

import pytest
import torch

from batch.consistency_adapter import CaseAdapter
from batch.consistency_config import CaseConfigParser, ConsistencyMode, ModeRegistry
from batch.consistency_flow import BatchConsistencyUnsupportedError
from batch.consistency_schema import get_q_lengths
from batch.consistency_transform import InvalidTransformError, generate_split_groups
from batch.sparse_flash_mla_consistency import create_consistency_flow


class ConsistencyTestConfig:
    enabled = os.environ.get("SMLA_CONSISTENCY_TEST", "0") == "1"
    pt_dir = os.environ.get("SMLA_PT_LOAD_PATH", "./data")
    single_case = os.environ.get("QSAS_TESTCASE_PATH", "").strip()
    seed = int(os.environ.get("SMLA_CONSISTENCY_SEED", "20260716"))
    randomize = os.environ.get("SMLA_CONSISTENCY_RANDOMIZE", "0") == "1"
    modes = ModeRegistry.resolve(
        modes=os.environ.get("SMLA_CONSISTENCY_MODES", "").strip() or None,
        profile=os.environ.get("SMLA_CONSISTENCY_PROFILE", "").strip() or None,
        warn_legacy=enabled,
    )
    split_groups_value = os.environ.get("SMLA_CONSISTENCY_SPLIT_GROUPS", "").strip()
    token_sizes_value = os.environ.get("SMLA_CONSISTENCY_TOKEN_SPLIT_SIZES", "").strip()
    mode3_batch = int(os.environ.get("SMLA_CONSISTENCY_MODE3_BATCH", "0"))
    common_tokens_value = os.environ.get("SMLA_CONSISTENCY_COMMON_TOKENS", "").strip()
    derived_extra_value = os.environ.get("SMLA_CONSISTENCY_DERIVED_EXTRA_TOKENS", "").strip()

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

    @classmethod
    def split_groups(cls, batch_size: int) -> List[List[int]]:
        if cls.split_groups_value:
            return CaseConfigParser.split_groups(cls.split_groups_value)
        return generate_split_groups(batch_size, seed=cls.seed, randomize=cls.randomize)

    @classmethod
    def token_split_sizes(cls, token_count: int) -> List[int]:
        if cls.token_sizes_value:
            return CaseConfigParser.split_sizes(cls.token_sizes_value, "token_split_sizes")
        if token_count == 16 * 1024:
            return [4 * 1024, 12 * 1024]
        first = token_count // 2
        if first == 0:
            raise InvalidTransformError("Mode 3 needs at least two query tokens")
        return [first, token_count - first]

    @classmethod
    def shape_change(cls, token_count: int) -> Dict[str, int]:
        if cls.common_tokens_value:
            common = int(cls.common_tokens_value)
        elif token_count >= 400:
            common = 300
        else:
            common = max(1, token_count // 2)
        if cls.derived_extra_value:
            derived_extra = int(cls.derived_extra_value)
        elif token_count == 400 and common == 300:
            derived_extra = 200
        else:
            derived_extra = token_count - common + 1
        return {"common": common, "derived_extra": derived_extra}

    @staticmethod
    def set_device() -> None:
        import torch_npu

        torch_npu.npu.set_device(int(os.environ.get("SMLA_DEVICE_ID", "0")))


TESTCASE_FILES = ConsistencyTestConfig.collect_pt_files()

pytestmark = [
    pytest.mark.consistency,
    pytest.mark.skipif(
        not ConsistencyTestConfig.enabled,
        reason="SMLA_CONSISTENCY_TEST is not 1; consistency tests are opt-in",
    ),
    pytest.mark.skipif(
        not TESTCASE_FILES,
        reason=f"No SMLA PT cases found in {ConsistencyTestConfig.pt_dir}",
    ),
]


def load_smla_case(path: str) -> Dict:
    data = torch.load(path, map_location="cpu")
    if CaseAdapter(data).name != "smla":
        pytest.skip("NOT_APPLICABLE: PT is not an SMLA case")
    return data


def run_or_skip(callable_):
    try:
        return callable_()
    except InvalidTransformError as error:
        pytest.skip(f"INVALID_TRANSFORM: {error}")
    except BatchConsistencyUnsupportedError as error:
        pytest.skip(f"UNSUPPORTED: {error}")


@pytest.mark.parametrize("testcase_file", TESTCASE_FILES)
def test_mode1_reorder(testcase_file):
    if ConsistencyMode.REORDER not in ConsistencyTestConfig.modes:
        pytest.skip("NOT_APPLICABLE: reorder not selected")
    data = load_smla_case(testcase_file)
    if CaseAdapter(data).get_batch_size() < 2:
        pytest.skip("NOT_APPLICABLE: reorder requires B >= 2")
    ConsistencyTestConfig.set_device()
    report = run_or_skip(
        lambda: create_consistency_flow().run_mode1(data, seed=ConsistencyTestConfig.seed)
    )
    print(report)
    assert report["pass"], report


@pytest.mark.parametrize("testcase_file", TESTCASE_FILES)
def test_mode2_split(testcase_file):
    if ConsistencyMode.SPLIT not in ConsistencyTestConfig.modes:
        pytest.skip("NOT_APPLICABLE: split not selected")
    data = load_smla_case(testcase_file)
    batch_size = CaseAdapter(data).get_batch_size()
    if batch_size < 2:
        pytest.skip("NOT_APPLICABLE: split requires B >= 2")
    groups = ConsistencyTestConfig.split_groups(batch_size)
    ConsistencyTestConfig.set_device()
    report = run_or_skip(lambda: create_consistency_flow().run_mode2(data, groups))
    print(report)
    assert report["pass"], report


@pytest.mark.parametrize("testcase_file", TESTCASE_FILES)
def test_mode3_token_split(testcase_file):
    if ConsistencyMode.TOKEN_SPLIT not in ConsistencyTestConfig.modes:
        pytest.skip("NOT_APPLICABLE: token-split not selected")
    data = load_smla_case(testcase_file)
    lengths = get_q_lengths(data, valid_only=True)
    batch_id = ConsistencyTestConfig.mode3_batch
    if batch_id >= len(lengths):
        pytest.skip(f"NOT_APPLICABLE: mode3 batch {batch_id} is outside B={len(lengths)}")
    split_sizes = ConsistencyTestConfig.token_split_sizes(lengths[batch_id])
    ConsistencyTestConfig.set_device()
    report = run_or_skip(
        lambda: create_consistency_flow().run_mode3(data, batch_id, split_sizes)
    )
    print(report)
    assert report["pass"], report


@pytest.mark.parametrize("testcase_file", TESTCASE_FILES)
def test_mode4_shape_change(testcase_file):
    if ConsistencyMode.SHAPE_CHANGE not in ConsistencyTestConfig.modes:
        pytest.skip("NOT_APPLICABLE: shape-change not selected")
    data = load_smla_case(testcase_file)
    lengths = get_q_lengths(data, valid_only=True)
    batch_id = ConsistencyTestConfig.mode3_batch
    if batch_id >= len(lengths):
        pytest.skip(f"NOT_APPLICABLE: mode4 batch {batch_id} is outside B={len(lengths)}")
    config = ConsistencyTestConfig.shape_change(lengths[batch_id])
    ConsistencyTestConfig.set_device()
    report = run_or_skip(
        lambda: create_consistency_flow().run_mode4(
            data,
            batch_id,
            config["common"],
            config["derived_extra"],
        )
    )
    print(report)
    assert report["pass"], report
