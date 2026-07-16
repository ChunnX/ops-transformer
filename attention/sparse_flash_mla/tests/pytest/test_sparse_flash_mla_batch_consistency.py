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

"""Batch consistency NPU test entry point.

**Default: OFF.**  Set ``SMLA_CONSISTENCY_TEST=1`` to enable.

This module is completely isolated from the existing ``single``, ``save``,
``load`` and ``load_graph`` entry points.  It does not modify any existing
file and only reads from the PT files produced by ``save``.

Usage::

    SMLA_CONSISTENCY_TEST=1 SMLA_CONSISTENCY_MODES=1,2 \\
        python3 -m pytest -s test_sparse_flash_mla_batch_consistency.py -v -m consistency

Or via test_run.sh::

    bash test_run.sh consistency -P ./data -M 1,2 --seed 20260716
"""

import os
import pytest
import torch

# ------------------------------------------------------------------ #
#  Switch: default OFF
# ------------------------------------------------------------------ #

_CONSISTENCY_ENABLED = os.environ.get("SMLA_CONSISTENCY_TEST", "0") == "1"


def _parse_modes(value):
    try:
        return {int(mode.strip()) for mode in value.split(",") if mode.strip()}
    except ValueError as error:
        raise pytest.UsageError(f"Invalid SMLA_CONSISTENCY_MODES value: {value!r}") from error


_CONSISTENCY_MODES = _parse_modes(os.environ.get("SMLA_CONSISTENCY_MODES", "1,2"))
_CONSISTENCY_SEED = int(os.environ.get("SMLA_CONSISTENCY_SEED", "20260716"))

if _CONSISTENCY_ENABLED:
    unsupported = _CONSISTENCY_MODES & {3, 4}
    invalid = _CONSISTENCY_MODES - {1, 2, 3, 4}
    if unsupported:
        reasons = {
            3: "requires CPU-golden query-position/mask equivalence validation",
            4: "changes query position, masks, prefix sums, and sparse indices",
        }
        detail = "; ".join(f"mode {mode}: {reasons[mode]}" for mode in sorted(unsupported))
        raise pytest.UsageError(f"Unsupported consistency mode requested: {detail}")
    if invalid or not _CONSISTENCY_MODES:
        raise pytest.UsageError(
            f"SMLA_CONSISTENCY_MODES must contain mode 1 and/or 2, got {sorted(_CONSISTENCY_MODES)}"
        )

pt_dir = os.getenv("SMLA_PT_LOAD_PATH", "./data")
_single_case_path = os.environ.get("QSAS_TESTCASE_PATH", "").strip()

# Collect PT files (same logic as test_sparse_flash_mla_batch.py but isolated)
_testcase_files: list = []
if _single_case_path:
    if os.path.isfile(_single_case_path):
        _testcase_files.append(_single_case_path)
elif os.path.isdir(pt_dir):
    for f in sorted(os.listdir(pt_dir)):
        if f.endswith(".pt"):
            _testcase_files.append(os.path.join(pt_dir, f))

# Skip entire module if consistency is not enabled
pytestmark = [
    pytest.mark.consistency,
    pytest.mark.skipif(
        not _CONSISTENCY_ENABLED,
        reason="SMLA_CONSISTENCY_TEST is not set to 1; consistency tests are opt-in",
    ),
    pytest.mark.skipif(
        len(_testcase_files) == 0,
        reason=f"No .pt files found in {pt_dir}; run 'save' mode first",
    ),
]


def _filter_by_b(testcase_files, min_b=2):
    """Filter PT files to those with B >= min_b."""
    filtered = []
    for f in testcase_files:
        try:
            data = torch.load(f, map_location="cpu")
            if int(data.get("B", 0)) >= min_b:
                filtered.append(f)
        except Exception:
            pass
    return filtered


_consistency_files = _filter_by_b(_testcase_files, min_b=2)


def _set_npu_device():
    import torch_npu

    device_id = int(os.environ.get("SMLA_DEVICE_ID", "0"))
    torch_npu.npu.set_device(device_id)


@pytest.mark.consistency
@pytest.mark.parametrize("testcase_file", _consistency_files)
def test_mode1_batch_reorder(testcase_file):
    """Mode 1: reorder batches, verify output consistency by token origin."""
    if 1 not in _CONSISTENCY_MODES:
        pytest.skip("Mode 1 not in SMLA_CONSISTENCY_MODES")

    from batch_consistency.runner import BatchConsistencyUnsupportedError, run_mode1

    test_data = torch.load(testcase_file, map_location="cpu")
    _set_npu_device()
    B = int(test_data["B"])
    assert B >= 2, f"Mode 1 requires B >= 2, got B={B}"

    try:
        case, baseline_res, derived_res, comparison = run_mode1(
            test_data, seed=_CONSISTENCY_SEED
        )
    except BatchConsistencyUnsupportedError as error:
        pytest.skip(str(error))

    print(f"\n[Mode1] file={os.path.basename(testcase_file)}")
    print(f"  order={case.transform_meta['order']}")
    print(f"  layout_q={case.transform_meta['layout_q']}, layout_kv={case.transform_meta['layout_kv']}")
    print(f"  output: pass={comparison['pass']}, fulfill={comparison['output']['fulfill_percent']:.2f}%")
    if comparison["output"]["bitwise_equal"]:
        print("  [diagnostic] output bitwise equal (md5 match)")
    if "lse" in comparison:
        print(f"  lse: pass={comparison['lse']['pass']}, fulfill={comparison['lse']['fulfill_percent']:.2f}%")

    if not comparison["pass"]:
        first_fail = "output"
        if "error" in comparison:
            pytest.fail(f"Origin alignment error: {comparison['error']}")
        if "lse" in comparison and not comparison["lse"]["pass"]:
            first_fail = "lse"
        pytest.fail(
            f"Mode 1 consistency failed: {first_fail} "
            f"fulfill={comparison[first_fail]['fulfill_percent']:.2f}%, "
            f"order={case.transform_meta['order']}, seed={_CONSISTENCY_SEED}"
        )


@pytest.mark.consistency
@pytest.mark.parametrize("testcase_file", _consistency_files)
def test_mode2_batch_split(testcase_file):
    """Mode 2: split batches into sub-cases, verify merged output consistency."""
    if 2 not in _CONSISTENCY_MODES:
        pytest.skip("Mode 2 not in SMLA_CONSISTENCY_MODES")

    from batch_consistency.runner import BatchConsistencyUnsupportedError, run_mode2

    test_data = torch.load(testcase_file, map_location="cpu")
    _set_npu_device()
    B = int(test_data["B"])
    assert B >= 2, f"Mode 2 requires B >= 2, got B={B}"

    try:
        cases, baseline_res, derived_results, _, comparison = run_mode2(
            test_data, seed=_CONSISTENCY_SEED
        )
    except BatchConsistencyUnsupportedError as error:
        pytest.skip(str(error))

    print(f"\n[Mode2] file={os.path.basename(testcase_file)}")
    for i, case in enumerate(cases):
        print(f"  sub-case {i}: batches={case.transform_meta['batch_ids']}, "
              f"B={len(case.transform_meta['batch_ids'])}")
    print(f"  merged: pass={comparison['pass']}, "
          f"fulfill={comparison['output']['fulfill_percent']:.2f}%")
    if comparison["output"]["bitwise_equal"]:
        print("  [diagnostic] merged output bitwise equal (md5 match)")
    if "lse" in comparison:
        print(f"  lse: pass={comparison['lse']['pass']}, "
              f"fulfill={comparison['lse']['fulfill_percent']:.2f}%")

    if not comparison["pass"]:
        failed_component = "lse" if "lse" in comparison and not comparison["lse"]["pass"] else "output"
        pytest.fail(
            f"Mode 2 consistency failed: component={failed_component}, "
            f"fulfill={comparison[failed_component]['fulfill_percent']:.2f}%, "
            f"groups={[c.transform_meta['batch_ids'] for c in cases]}, "
            f"seed={_CONSISTENCY_SEED}"
        )
