# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "mqfa_ci_log_monitor",
    Path(__file__).resolve().parents[1] / "tools/monitor_ci_log.py",
)
monitor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monitor)


@pytest.mark.parametrize(
    "details, expected",
    [
        ("FailElems: 0/100\nFailElems: 0/30", True),
        ("FailElems: 0/100\nFailElems: 7/30", False),
        ("FailElems: 0/100\nFailElems: invalid", False),
        ("no precision data", False),
    ],
)
def test_strict_report_checks_every_output(monkeypatch, details, expected):
    parent = "build_verify_logs/run"
    paths = [f"{parent}/SUMMARY.txt", f"{parent}/verify.log"]
    blobs = {
        paths[0]: "case=b270_sink_n1_6\ncommit=abc123\nresult=PASS",
        paths[1]: details,
    }
    monkeypatch.setattr(monitor, "summary_paths", lambda *args: paths[:1])
    monkeypatch.setattr(monitor, "blob", lambda repo, ref, path: blobs[path])
    monkeypatch.setattr(
        monitor,
        "git",
        lambda repo, *args: "\n".join(paths) if args[0] == "ls-tree" else "log123",
    )
    assert (
        monitor.report(
            Path.cwd(), "origin/log", monitor.DEFAULT_CASE, "abc", strict=True
        )
        is expected
    )
    assert (
        monitor.report(
            Path.cwd(), "origin/log", monitor.DEFAULT_CASE, "other", strict=True
        )
        is None
    )


def test_restarted_strict_monitor_checks_already_seen_commit(monkeypatch, tmp_path):
    state = tmp_path / "last-seen"
    state.write_text("log123\n", encoding="utf-8")

    def git(repo, *args):
        if args == ("rev-parse", "--git-path", "mqfa-ci-log-monitor/last-seen"):
            return str(state)
        if args == ("rev-parse", "origin/log"):
            return "log123"
        assert args == ("fetch", "--quiet", "origin", "log")
        return ""

    monkeypatch.setattr(monitor, "git", git)
    monkeypatch.setattr(monitor, "report", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        sys, "argv", ["monitor", "--repo", str(tmp_path), "--commit", "abc", "--check"]
    )
    assert monitor.main() == 0
