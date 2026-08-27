from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


_VERIFIER_PATH = Path(__file__).with_name("verify_flash_kda_aclnn_dump.py")
_SPEC = importlib.util.spec_from_file_location(
    "flash_kda_dump_verifier", _VERIFIER_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load verifier: {_VERIFIER_PATH}")
verifier = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = verifier
_SPEC.loader.exec_module(verifier)


def test_metadata_drives_dynamic_dump_shape(tmp_path):
    metadata = tmp_path / "metadata.txt"
    metadata.write_text(
        "schema=1\n"
        "batch=2\n"
        "heads=96\n"
        "sequence=10240\n"
        "head_dim=128\n"
        "scale=0.0883883476\n"
        "lower_bound=-1\n",
        encoding="utf-8",
    )

    assert verifier._load_metadata(metadata) == {
        "schema": 1,
        "batch": 2,
        "heads": 96,
        "sequence": 10240,
        "head_dim": 128,
        "scale": 0.0883883476,
        "lower_bound": -1.0,
    }


def test_metadata_rejects_non_finite_scalar(tmp_path):
    metadata = tmp_path / "metadata.txt"
    metadata.write_text(
        "schema=1\n"
        "batch=1\n"
        "heads=3\n"
        "sequence=8192\n"
        "head_dim=128\n"
        "scale=nan\n"
        "lower_bound=-1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported"):
        verifier._load_metadata(metadata)


def test_memmap_rejects_truncated_dump(tmp_path):
    dump = tmp_path / "tensor.bf16"
    dump.write_bytes(b"\x00\x00")

    with pytest.raises(ValueError, match="expected 4 bytes"):
        verifier._checked_memmap(dump, np.uint16, (2,))


def test_streaming_finite_checks_cover_bf16_and_float(tmp_path):
    bf16_path = tmp_path / "out.bf16"
    np.array([0x3F80, 0x7F80], dtype=np.uint16).tofile(bf16_path)
    float_path = tmp_path / "state.f32"
    np.array([0.0, np.inf], dtype=np.float32).tofile(float_path)

    bf16 = verifier._checked_memmap(bf16_path, np.uint16, (2,))
    state = verifier._checked_memmap(float_path, np.float32, (2,))
    assert not verifier._all_finite_bf16(bf16)
    assert not verifier._all_finite_float(state)


@pytest.mark.parametrize("text, expected", [("0:0", (0, 0)), ("1:95", (1, 95))])
def test_sample_parser(text, expected):
    assert verifier._sample(text) == expected


def test_all_golden_samples_cover_every_batch_and_head():
    assert verifier._resolve_samples(2, 3, [], True) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]


def test_explicit_and_all_samples_are_exclusive():
    with pytest.raises(ValueError, match="exclusive"):
        verifier._resolve_samples(1, 1, [(0, 0)], True)
