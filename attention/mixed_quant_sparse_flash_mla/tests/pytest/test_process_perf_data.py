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

"""Regression checks for MQSMLA profiling reports (no NPU required)."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from process_perf_data import collect_operator_timings, process_profiler_data


METADATA = "MixedQuantSparseFlashMlaMetadata"
ATTENTION = "MixedQuantSparseFlashMla"


def summary(rows):
    return pd.DataFrame(rows, columns=["OP Type", "Task Duration(us)"])


def test_orphan_metadata_and_unrelated_ops_do_not_shift_timings():
    data = summary([(METADATA, 10), (ATTENTION, 20), (METADATA, 30),
                    ("Cast", 1), (METADATA, 40), (ATTENTION, 50)])
    assert collect_operator_timings(data, 2) == [(10, 20), (40, 50)]


@pytest.mark.parametrize("rows,count", [
    ([(ATTENTION, 20)], 1),
    ([(METADATA, 10)], 1),
    ([(METADATA, 10), (ATTENTION, float("nan"))], 1),
    ([(METADATA, 10), (ATTENTION, 20)], 2),
])
def test_incomplete_or_invalid_timings_are_rejected(rows, count):
    with pytest.raises(ValueError):
        collect_operator_timings(summary(rows), count)


def test_all_failed_cases_do_not_produce_an_empty_performance_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "PROF_test" / "mindstudio_profiler_output"
    folder.mkdir(parents=True)
    summary([(METADATA, 10)]).to_csv(folder / "op_summary.csv", index=False)
    pd.DataFrame({"result": ["NPU ERROR"]}).to_excel("result.xlsx", index=False)
    args = SimpleNamespace(test_result_path="result.xlsx", is_compare=False,
                           delete_perf_file=True)
    assert process_profiler_data(args) is None
    assert not Path("result_perf.xlsx").exists()
    assert folder.exists()


@pytest.mark.parametrize("failures", [0, 1])
def test_perf_script_preserves_pytest_status_when_msprof_returns_zero(tmp_path, failures):
    source = Path(__file__).parent
    for name in ("perf_run.sh", "process_perf_data.py"):
        shutil.copy2(source / name, tmp_path / name)
    binary = tmp_path / "bin"
    binary.mkdir()
    # Simulate msprof returning success regardless of the application's exit status.
    profiler = binary / "msprof"
    profiler.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\nimport sys\n"
        "path = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--junitxml='))\n"
        f"Path(path).write_text('<testsuites><testsuite tests=\"2\" failures=\"{failures}\" errors=\"0\" skipped=\"0\"/></testsuites>')\n"
        "Path('results.xlsx').write_bytes(Path('fixture.xlsx').read_bytes())\n"
    )
    profiler.chmod(0o755)
    folder = tmp_path / "PROF_test" / "mindstudio_profiler_output"
    folder.mkdir(parents=True)
    summary([(METADATA, 10), (ATTENTION, 20)]).to_csv(folder / "op_summary.csv", index=False)
    pd.DataFrame({"Testcase_Name": ["case"], "result": ["Executed (accuracy skipped)"]}).to_excel(
        tmp_path / "fixture.xlsx", index=False
    )
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
               MQSMLA_RESULT_SAVE_PATH="results.xlsx", MQSMLA_KEEP_PROF="0")
    result = subprocess.run(["bash", "perf_run.sh"], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == failures, result.stdout + result.stderr
    report = pd.read_excel(tmp_path / "results_perf.xlsx")
    assert report["metadata_duration"].tolist() == [10]
    assert report["qsmla_duration"].tolist() == [20]
    assert folder.exists() == bool(failures)
