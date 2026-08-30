# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Directed ST materialization and execution-harness regression tests."""

from __future__ import annotations

import ast
import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


TESTS_DIR = Path(__file__).resolve().parent.parent
ST_DIR = TESTS_DIR / "st"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


materializer = load_module(
    "moe_tutel_dispatch_materializer", ST_DIR / "design/materialize_ttk_cases.py"
)
manual_data = load_module(
    "moe_tutel_dispatch_manual_data", ST_DIR / "design/prepare_manual_data.py"
)
harness = load_module("moe_tutel_dispatch_harness", ST_DIR / "run_contract_harness.py")


def test_materialized_csv_uses_lf_line_endings(tmp_path):
    output = tmp_path / "cases.csv"

    materializer.write_rows(output, ["case_id"], [{"case_id": "MTD"}])

    assert output.read_bytes() == b"case_id\nMTD\n"


def test_p0_descriptor_cases_are_real_geir_targets():
    source = ST_DIR / "testcases/MoeTutelDispatch_required_test_cases.csv"
    _, geir_rows, manifest = materializer.required_ttk_rows(source)
    by_id = {row["case_id"]: row for row in manifest}

    assert {row["testcase_name"] for row in geir_rows} >= {
        "MTD_REQ_031",
        "MTD_REQ_032",
        "MTD_REQ_033",
    }
    assert by_id["MTD_REQ_031"]["harness"] == "ttk_geir"
    assert by_id["MTD_REQ_032"]["harness"] == "ttk_geir"
    assert by_id["MTD_REQ_033"]["dynamic"] == "1"

    dynamic_case = next(
        row for row in geir_rows if row["testcase_name"] == "MTD_REQ_033"
    )
    assert ast.literal_eval(dynamic_case["input_ori_shapes"]) == (
        (4, 8),
        (3, 4),
        (3, 4),
        (3, 4),
    )
    assert ast.literal_eval(dynamic_case["dyn_input_shapes"]) == (
        (-1, -1),
        (3, -1),
        (3, -1),
        (3, -1),
    )


def test_ttk_command_honors_repeat_dynamic_binary_and_manual_data():
    row = {
        "harness": "ttk_geir",
        "target": "testcases/MoeTutelDispatch_required_geir.csv#MTD_REQ_033",
        "repeats": "20",
        "parallelism": "1",
        "dynamic": "1",
    }
    args = SimpleNamespace(
        ttk_root=Path("/tmp/ttk"),
        manual_data_dir=Path("/tmp/manual"),
        plugin=Path("/tmp/golden.py"),
    )

    command, cwd = harness.command_for(row, args, ST_DIR)

    assert cwd == args.ttk_root
    assert "--pc=1" in command
    assert "--run=20" in command
    assert "--dl=1" in command
    assert "--compare=binary" in command
    assert command[command.index("--warmup") + 1] == "false"
    assert command[command.index("--manual-data-dirs") + 1] == str(args.manual_data_dir)
    assert "-c=false" in command and "-d=true" in command
    assert "-b=release" not in command


def test_kernel_command_is_binary_only_and_skips_warmup():
    row = {
        "harness": "ttk_kernel",
        "target": "testcases/MoeTutelDispatch_required_kernel.csv#MTD_REQ_010",
        "repeats": "1",
        "parallelism": "1",
        "dynamic": "0",
    }
    args = SimpleNamespace(
        ttk_root=Path("/tmp/ttk"),
        manual_data_dir=Path("/tmp/manual"),
        plugin=Path("/tmp/golden.py"),
    )

    command, cwd = harness.command_for(row, args, ST_DIR)

    assert cwd == args.ttk_root
    assert command[command.index("--warmup") + 1] == "false"
    assert "-d=false" in command
    assert "-b=release" in command
    assert "-d=true" not in command


def test_csv_target_is_resolved_inside_st_directory():
    csv_path, case_name = harness.split_csv_target(
        ST_DIR,
        "testcases/MoeTutelDispatch_required_geir.csv#MTD_REQ_033",
    )

    assert (
        csv_path == (ST_DIR / "testcases/MoeTutelDispatch_required_geir.csv").resolve()
    )
    assert case_name == "MTD_REQ_033"


def test_csv_target_rejects_directory_escape(tmp_path):
    outside = tmp_path / "outside.csv"
    outside.touch()
    st_dir = tmp_path / "st"
    st_dir.mkdir()

    with pytest.raises(RuntimeError, match="escapes the ST directory"):
        harness.split_csv_target(st_dir, "../outside.csv#case")


def test_parallelism_launches_two_subprocesses_concurrently(monkeypatch):
    barrier = threading.Barrier(2)
    thread_ids = []

    def fake_run(*_args, **_kwargs):
        thread_ids.append(threading.get_ident())
        barrier.wait(timeout=2)

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    row = {
        "case_id": "MTD_REQ_035",
        "harness": "host_ut",
        "target": "MoeTutelDispatchTiling.DynamicRank",
        "repeats": "1",
        "parallelism": "2",
        "dynamic": "0",
    }
    args = SimpleNamespace(host_ut_binary=Path("/tmp/host_ut"))

    harness.execute_row(row, args, ST_DIR)

    assert len(thread_ids) == 2
    assert len(set(thread_ids)) == 2


def test_duplicate_destination_manual_case_has_writer_independent_bits():
    row = {
        "case_id": "MTD_REQ_027",
        "dtype": "float16",
        "S": "2",
        "H": "8",
        "K": "2",
        "E": "3",
        "capacity": "4",
        "route_pattern": "duplicate_destination",
        "gate_pattern": "all_one",
        "x_pattern": "random_bits",
    }

    _, inputs, golden = manual_data.prepare_case(row, lambda values: values)
    x_raw = inputs[0].view(np.uint16)
    y_raw = golden.view(np.uint16)

    np.testing.assert_array_equal(x_raw[0], x_raw[1])
    np.testing.assert_array_equal(y_raw[0, 0], x_raw[0])
