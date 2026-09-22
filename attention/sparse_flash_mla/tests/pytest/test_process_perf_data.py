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

import os

import pandas as pd

from process_perf_data import process_profiler_data


def _write_profile(root, folder_name, rows, timestamp):
    output = root / folder_name / "mindstudio_profiler_output"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "op_summary_0.csv", index=False)
    # Keep the collection order deterministic without depending on folder names.
    os.utime(root / folder_name, (timestamp, timestamp))


def test_process_profiler_data_merges_multiple_profile_folders(tmp_path):
    result_path = tmp_path / "smla_result.xlsx"
    pd.DataFrame(
        [
            {"testcase_name": "case_1", "result": "Passed"},
            {"testcase_name": "bad_case", "result": "NPU ERROR"},
            {"testcase_name": "case_2", "result": "Failed"},
        ]
    ).to_excel(result_path, index=False)

    _write_profile(
        tmp_path,
        "PROF_first",
        [
            {"Op Name": "UnrelatedOp", "Task Duration(us)": 999},
            {"Op Name": "SparseFlashMlaMetadata", "Task Duration(us)": 11},
            {"Op Name": "SparseFlashMla", "Task Duration(us)": 101},
        ],
        100,
    )
    _write_profile(
        tmp_path,
        "PROF_second",
        [
            {"Op Name": "sparse_flash_mla_metadata", "Task Duration(us)": 22},
            {"Op Name": "sparse_flash_mla", "Task Duration(us)": 202},
        ],
        200,
    )

    output_path = process_profiler_data(result_path, profile_root=tmp_path)
    merged = pd.read_excel(output_path)

    assert output_path == tmp_path / "smla_result_perf.xlsx"
    assert merged["testcase_name"].tolist() == ["case_1", "bad_case", "case_2"]
    assert merged.loc[0, "metadata_duration"] == 11
    assert merged.loc[0, "smla_duration"] == 101
    assert pd.isna(merged.loc[1, "smla_duration"])
    assert merged.loc[2, "metadata_duration"] == 22
    assert merged.loc[2, "smla_duration"] == 202
    assert merged.loc[0, "profile_folder"] == "PROF_first"
    assert merged.loc[2, "profile_folder"] == "PROF_second"

    # Reprocessing the same folders rebuilds the table and does not duplicate rows.
    process_profiler_data(result_path, profile_root=tmp_path)
    assert len(pd.read_excel(output_path)) == 3


def test_process_profiler_data_retains_extra_measurements(tmp_path):
    result_path = tmp_path / "result.xlsx"
    pd.DataFrame([{"testcase_name": "case_1", "result": "Passed"}]).to_excel(
        result_path, index=False
    )
    _write_profile(
        tmp_path,
        "PROF_all_runs",
        [
            {"Op Name": "SparseFlashMlaMetadata", "Task Duration(us)": 1},
            {"Op Name": "SparseFlashMla", "Task Duration(us)": 10},
            {"Op Name": "SparseFlashMlaMetadata", "Task Duration(us)": 2},
            {"Op Name": "SparseFlashMla", "Task Duration(us)": 20},
        ],
        100,
    )

    output_path = process_profiler_data(result_path, profile_root=tmp_path)
    merged = pd.read_excel(output_path)

    assert len(merged) == 2
    assert merged["smla_duration"].tolist() == [10, 20]
    assert merged["profile_run"].tolist() == [1, 2]
    assert pd.isna(merged.loc[1, "testcase_name"])


def test_process_profiler_data_preserves_runs_when_profile_is_reused(tmp_path):
    result_path = tmp_path / "result.xlsx"
    pd.DataFrame([{"testcase_name": "case_1", "result": "Passed"}]).to_excel(
        result_path, index=False
    )
    _write_profile(
        tmp_path,
        "PROF_reused",
        [
            {"Op Name": "SparseFlashMlaMetadata", "Task Duration(us)": 3},
            {"Op Name": "SparseFlashMla", "Task Duration(us)": 30},
        ],
        100,
    )
    output_path = process_profiler_data(result_path, profile_root=tmp_path)

    pd.DataFrame(
        [
            {"testcase_name": "case_1", "result": "Passed"},
            {"testcase_name": "case_2", "result": "Passed"},
        ]
    ).to_excel(result_path, index=False)
    _write_profile(
        tmp_path,
        "PROF_reused",
        [
            {"Op Name": "SparseFlashMlaMetadata", "Task Duration(us)": 4},
            {"Op Name": "SparseFlashMla", "Task Duration(us)": 40},
        ],
        200,
    )

    process_profiler_data(result_path, profile_root=tmp_path)
    merged = pd.read_excel(output_path)

    assert merged["testcase_name"].tolist() == ["case_1", "case_2"]
    assert merged["metadata_duration"].tolist() == [3, 4]
    assert merged["smla_duration"].tolist() == [30, 40]
