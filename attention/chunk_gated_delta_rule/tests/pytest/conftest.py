# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)
import csv
import os
import pytest
import torch
import torch_npu  # noqa: F401

_RESULT_ROWS = []
_CURRENT_SEED = 0


@pytest.fixture(autouse=True)
def _set_random_seed():
    global _CURRENT_SEED
    fix_seed = os.environ.get("TORCH_SEED", "")
    if fix_seed:
        _CURRENT_SEED = int(fix_seed)
        torch.manual_seed(_CURRENT_SEED)
        torch.npu.manual_seed(_CURRENT_SEED)
    else:
        _CURRENT_SEED = torch.seed()
        torch.npu.manual_seed(_CURRENT_SEED)
    logger.info(f"[seed] {_CURRENT_SEED}")
    yield


def _get_model_name():
    if os.environ.get("USE_GRAPH", "false").lower() == "true":
        return "aclgraph"
    return "torch直调"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when != "call" and not (report.when == "setup" and (report.failed or report.skipped)):
        return

    params = item.funcargs.get("param_combinations")
    if params is None:
        # 泛化测试以case字典/独立参数组织；保留旧RDV列格式，并用nodeid确保失败可定位。
        case = item.funcargs.get("case", {})
        lengths = case.get("lengths", (65,)) if case else None
        params = {
            "_name": item.nodeid,
            "model": "ACLNN直调",
            "B": len(lengths) if lengths is not None else "",
            "seqlen": lengths if lengths is not None else "",
            "nk": case.get("nk", 2) if case else "",
            "nv": case.get("nv", 4) if case else "",
            "dk": case.get("dk", 128) if case else item.funcargs.get("dk", ""),
            "dv": case.get("dv", 128) if case else item.funcargs.get("dv", ""),
            "chunk_size": 64,
            "data_type": "bfloat16",
            "state_data_type": case.get("state_dtype", item.funcargs.get("dtype", "")),
            "has_g": case.get("gate", "random") != "none" if case else "",
            "is_contiguous": not bool(case.get("layouts")) if case else "",
            "seed": case.get("seed", 20260908),
        }

    if report.passed:
        status = "PASSED"
    elif report.failed:
        status = "FAILED"
    elif report.skipped:
        status = "SKIPPED"
    else:
        status = "UNKNOWN"

    error = ""
    if status in ("FAILED", "ERROR") and report.longrepr:
        error = str(report.longreprtext).replace("\n", " | ")[:2000]

    row = {
        "seed": params.get("seed", _CURRENT_SEED),
        "test_name": params.get("_name", ""),
        "model": params.get("model", _get_model_name()),
        "status": status,
        "B": params.get("B", ""),
        "seqlen": params.get("seqlen", ""),
        "nk": params.get("nk", ""),
        "nv": params.get("nv", ""),
        "dk": params.get("dk", ""),
        "dv": params.get("dv", ""),
        "chunk_size": params.get("chunk_size", ""),
        "data_type": str(params.get("data_type", "")),
        "state_data_type": str(params.get("state_data_type", "")),
        "has_g": params.get("has_g", ""),
        "is_continue": params.get("is_contiguous", ""),
        "errmsg": error,
        "durations": "",
    }
    _RESULT_ROWS.append(row)


def pytest_runtest_logfinish(nodeid, location):
    logger.info("")


def pytest_sessionfinish(session, exitstatus):
    csv_file = os.environ.get("CSV_FILE", "")
    if not csv_file or not _RESULT_ROWS:
        return

    fields = [
        "seed",
        "test_name",
        "model",
        "status",
        "B",
        "seqlen",
        "nk",
        "nv",
        "dk",
        "dv",
        "chunk_size",
        "data_type",
        "state_data_type",
        "has_g",
        "is_continue",
        "errmsg",
        "durations",
    ]
    with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_RESULT_ROWS)

    total = len(_RESULT_ROWS)
    passed = sum(1 for r in _RESULT_ROWS if r["status"] == "PASSED")
    failed = sum(1 for r in _RESULT_ROWS if r["status"] in ("FAILED", "ERROR"))
    logger.info(f"\nCSV generated: {csv_file} (total {total}, passed{passed}, failed{failed})")
