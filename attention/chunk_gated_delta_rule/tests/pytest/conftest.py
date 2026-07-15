import csv
import os

import pytest

_results = []

CSV_COLUMNS = [
    "test_name", "status",
    "B", "seqlen", "nk", "nv", "dk", "dv",
    "chunk_size", "data_type", "state_data_type",
    "has_g", "is_continue", "error", "durations",
]


def _get_params(item):
    if hasattr(item, "callspec"):
        pv = item.callspec.params.get("param_combinations", {})
        if isinstance(pv, dict):
            return {k: str(v) for k, v in pv.items()}
    return {}


def _add_result(item, report, status):
    params = _get_params(item)
    error = ""
    if report.failed and report.longrepr:
        error = str(report.longrepr)
    _results.append({
        "test_name": item.name,
        "status": status,
        "B": params.get("B", ""),
        "seqlen": params.get("seqlen", ""),
        "nk": params.get("nk", ""),
        "nv": params.get("nv", ""),
        "dk": params.get("dk", ""),
        "dv": params.get("dv", ""),
        "chunk_size": params.get("chunk_size", ""),
        "data_type": params.get("data_type", ""),
        "state_data_type": params.get("state_data_type", ""),
        "has_g": params.get("has_g", ""),
        "is_continue": params.get("is_continue", ""),
        "error": error,
        "durations": "",
    })


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "setup" and report.failed:
        _add_result(item, report, "ERROR")
    elif report.when == "call":
        _add_result(item, report, "PASS" if report.passed else "FAIL")


def pytest_sessionfinish(session, exitstatus):
    csv_path = os.environ.get("CSV_REPORT_PATH")
    if not csv_path or not _results:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(_results)
    print(f"\nCSV 结果已保存至: {csv_path}")
