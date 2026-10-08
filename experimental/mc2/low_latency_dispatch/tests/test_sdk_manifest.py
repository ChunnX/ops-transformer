# Copyright (c) 2026, edconeone
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SDK identity checks must reject stale or modified dependency artifacts."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    script = (
        Path(__file__).resolve().parents[2]
        / "common/deepep/scripts/record_shmem_sdk.py"
    )
    spec = importlib.util.spec_from_file_location("mc2_record_sdk", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "__file__", str(tmp_path / "scripts/record_shmem_sdk.py")
    )
    root = tmp_path / "sdk"
    (root / "include/host").mkdir(parents=True)
    (root / "lib").mkdir()
    header = b"// SDK fixture\n"
    (root / "include/host/header.h").write_bytes(header)
    (root / "lib/libshmem.so").write_bytes(b"fixture, not a binary")
    lock = {
        "shmem": {
            "revision": "a" * 40,
            "header_sha256_lf": {"host/header.h": hashlib.sha256(header).hexdigest()},
        }
    }
    (tmp_path / "dependencies.lock.json").write_text(json.dumps(lock))
    return module, root


def test_record_and_verify(sdk):
    module, root = sdk
    manifest = module.record(root, "a" * 40)
    assert module.check(root) == manifest
    with pytest.raises(FileExistsError):
        module.record(root, "a" * 40)


def test_wrong_revision_rejected(sdk):
    module, root = sdk
    with pytest.raises(ValueError, match="revision"):
        module.record(root, "b" * 40)
    assert not (root / "deepep-sdk.json").exists()


@pytest.mark.parametrize("path", ["include/host/header.h", "lib/libshmem.so"])
def test_changed_sdk_is_rejected(sdk, path):
    module, root = sdk
    module.record(root, "a" * 40)
    (root / path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        module.check(root)
