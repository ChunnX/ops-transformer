# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host code-generation regressions; no Torch/NPU or profiler execution.

Run: python -m pytest --confcutdir=host_tests host_tests/test_perf_threshold.py
from attention/flash_attn/tests/pytests after installing pytest.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "perf_runner.py"
_SPEC = importlib.util.spec_from_file_location("perf_threshold_runner", _MODULE_PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)

THRESHOLDS = [0, 8, 16, 32, 64, 10000]
CASES = [
    {"S1": 8, "N1": 1, "D": 64, "marker": "eight"},
    {"S1": 16, "N1": 1, "D": 64, "marker": "sixteen"},
    {"S1": 32, "N1": 1, "D": 64, "marker": "thirtytwo"},
    {"S1": 64, "N1": 1, "D": 64, "marker": "sixtyfour"},
    {"N1": 1, "D": 64, "marker": "missing"},
]


def assigned_literals(script):
    tree = ast.parse(script)
    compile(tree, "generated-perf-script", "exec")
    fields = {"_hot_cases", "_cold_cases", "_hot_names", "_cold_names"}
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in fields
    }


@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("order", [range(5), range(4, -1, -1), [2, 0, 4, 1, 3]])
def test_batch_cases_and_names_use_the_same_threshold(threshold, order):
    cases = [CASES[i] for i in order]
    names = [case["marker"] for case in cases]
    script = runner.build_batch_script(cases, 3, threshold, case_names=names)
    values = assigned_literals(script)
    hot = [case for case in cases if case.get("S1", 9999) > threshold]
    cold = [case for case in cases if case.get("S1", 9999) <= threshold]
    assert values["_hot_cases"] == hot
    assert values["_cold_cases"] == cold
    assert values["_hot_names"] == [case["marker"] for case in hot]
    assert values["_cold_names"] == [case["marker"] for case in cold]
    for mode in ["hot", "cold"]:
        generated_cases = values[f"_{mode}_cases"]
        generated_names = values[f"_{mode}_names"]
        assert len(generated_cases) == len(generated_names)
        for case, name in zip(generated_cases, generated_names, strict=True):
            assert case["marker"] == name
        # Every emitted case index must be valid for its generated literal list.
        indices = [
            node.value.slice.value
            for node in ast.parse(script).body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == f"_{mode}_cases"
        ]
        assert indices == list(range(len(generated_cases)))


@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("case", CASES)
def test_single_case_mode_and_memory_path_use_the_threshold(threshold, case):
    script = runner.build_case_script(case, 3, "fixture", cold_thr=threshold)
    compile(script, "generated-single-perf-script", "exec")
    hot = case.get("S1", 9999) > threshold
    mode = "hot" if hot else "cold"
    assert f"CASE fixture  {mode}  runs=3" in script
    assert ("torch.npu.empty_cache()" in script) is not hot
    assert ("_f1 = torch.randn" in script) is not hot
    assert ("_f2 = torch.randn" in script) is not hot


@pytest.mark.parametrize("case", CASES)
def test_existing_single_case_call_defaults_to_sixteen(case):
    default = runner.build_case_script(case, 2, "fixture", 1)
    mode = "hot" if case.get("S1", 9999) > 16 else "cold"
    assert f"CASE fixture  {mode}  runs=2" in default
    assert "torch.npu.set_device(1)" in default
