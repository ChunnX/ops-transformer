"""Cached stages must be intact and must not invoke Native compilation."""
from pathlib import Path
import runpy
import sys

import opkit
from opkit.integrity import write_manifest
import pytest


@pytest.mark.parametrize("damage", [None, "changed", "missing"])
def test_verify_cached_stage(tmp_path, monkeypatch, damage):
    source = tmp_path / "operator.py"
    source.write_text("# Cached input\n")
    stage = tmp_path / "stage"
    stage.mkdir()
    provider = stage / "provider.so"
    provider.write_bytes(b"cached provider")
    write_manifest(stage, {"kind": "opkit"})
    if damage == "changed":
        provider.write_bytes(b"changed provider")
    elif damage == "missing":
        provider.unlink()
    monkeypatch.setattr(opkit, "build", lambda **kwargs: pytest.fail("cache hit recompiled Native"))
    builder = Path(__file__).parents[3] / "scripts/build_cannbotdsl.py"
    monkeypatch.setattr(sys, "argv", [str(builder), "--source", str(source),
                                     "--output", str(stage), "--verify-only"])
    if damage:
        with pytest.raises(ValueError, match="manifest"):
            runpy.run_path(str(builder), run_name="__main__")
    else:
        runpy.run_path(str(builder), run_name="__main__")
