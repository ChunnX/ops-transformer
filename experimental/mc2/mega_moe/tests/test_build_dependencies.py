# Copyright (c) 2026, edconeone
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate toolchain/dependency mismatches before native compilation starts."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def checker(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "common/deepep/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "mc2_check_dependencies", scripts / "check_dependencies.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "__file__", str(tmp_path / "scripts/check_dependencies.py")
    )
    lock = {
        "source": {"revision": "source"},
        "cann_version": "9.2.0",
        "megamoe": {"catlass_revision": "a" * 40},
    }
    (tmp_path / "dependencies.lock.json").write_text(json.dumps(lock))
    (tmp_path / "cann/compiler").mkdir(parents=True)
    (tmp_path / "cann/compiler/version.info").write_text("Version=9.2.0\n")
    (tmp_path / "catlass/include/catlass").mkdir(parents=True)
    (tmp_path / "catlass/include/catlass/catlass.hpp").write_text("// fixture\n")
    manifest = tmp_path / "sdk.json"
    manifest.write_text('{"revision": "sdk"}')
    monkeypatch.setattr(module, "check", lambda _: manifest)
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda args, **kw: "a" * 40 if "rev-parse" in args else "",
    )
    return module, tmp_path


def test_records_pinned_component_identity(checker):
    module, root = checker
    info = module.validate(root / "cann", root / "shmem", root / "catlass", True)
    assert info["megamoe"] == {"enabled": True, "timer": True, "catlass": "a" * 40}
    assert info["shmem"]["revision"] == "sdk"


def test_wrong_cann_rejected(checker):
    module, root = checker
    (root / "cann/compiler/version.info").write_text("Version=9.0.0\n")
    with pytest.raises(ValueError, match="CANN 9.2.0"):
        module.validate(root / "cann", root / "shmem")


def test_wrong_catlass_rejected(checker, monkeypatch):
    module, root = checker
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kw: "b" * 40)
    with pytest.raises(ValueError, match="revision"):
        module.validate(root / "cann", root / "shmem", root / "catlass")


def test_modified_catlass_rejected(checker, monkeypatch):
    module, root = checker
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda args, **kw: "a" * 40
        if "rev-parse" in args
        else " M include/catlass/catlass.hpp",
    )
    with pytest.raises(ValueError, match="local changes"):
        module.validate(root / "cann", root / "shmem", root / "catlass")
