from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest


_RUNNER_PATH = Path(__file__).with_name("run_flash_kda_acceptance.py")
_SPEC = importlib.util.spec_from_file_location("flash_kda_acceptance", _RUNNER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load runner: {_RUNNER_PATH}")
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


def _smoke_output(*, streams: int, waves: int, workspace_size: int = 1024) -> str:
    lines = []
    for wave in range(waves):
        for stream in range(streams):
            lines.extend(
                [
                    f"wave={wave} stream={stream} workspace_size={workspace_size} "
                    "get_workspace_status=0",
                    f"wave={wave} stream={stream} execute_status=0",
                    f"wave={wave} stream={stream} output_finite=true "
                    "final_state_finite=true",
                ]
            )
    lines.append(
        f"timing streams={streams} waves={waves} "
        "launch_sync_ms_min=0.500 avg=0.750 max=1.000"
    )
    return "\n".join(lines)


def test_smoke_parser_proves_every_invocation_and_stable_workspace():
    assert runner._parse_smoke(
        _smoke_output(streams=2, waves=20),
        expected_streams=2,
        expected_waves=20,
    ) == {
        "invocations": 40,
        "workspace_size": 1024,
        "timing_ms": {"min": 0.5, "avg": 0.75, "max": 1.0},
    }


def test_smoke_parser_rejects_missing_invocation():
    output = _smoke_output(streams=2, waves=2).replace(
        "wave=1 stream=1 execute_status=0\n", ""
    )
    with pytest.raises(AssertionError, match="invocation count"):
        runner._parse_smoke(output, expected_streams=2, expected_waves=2)


def test_golden_parser_requires_full_batch_head_coverage():
    output = (
        "validated_samples=192; output max_abs=2.5e-4 max_rel=1.0e-2; "
        "state max_abs=1.0e-3 max_rel=5.0e-3"
    )
    result = runner._parse_golden(output, expected_samples=192)
    assert result["samples"] == 192
    with pytest.raises(AssertionError, match="does not cover"):
        runner._parse_golden(output, expected_samples=191)


def test_native_parser_requires_bitwise_acl_nn_parity():
    output = (
        "sample=0:0 native/golden output max_abs=2e-4; state max_abs=3e-4; "
        "aclnn/native output max_abs=0.0; state max_abs=0.0\n"
        "sample=1:95 native/golden output max_abs=3e-4; state max_abs=4e-4; "
        "aclnn/native output max_abs=0.0; state max_abs=0.0\n"
    )
    result = runner._parse_native(output, expected_samples=["0:0", "1:95"])
    assert result["bitwise_equal"] is True

    drifted = output.replace(
        "output max_abs=0.0; state", "output max_abs=1e-4; state", 1
    )
    with pytest.raises(AssertionError, match="not bitwise"):
        runner._parse_native(drifted, expected_samples=["0:0", "1:95"])


def test_negative_matrix_requires_all_ten_fail_closed_cases():
    labels = [
        "lower_bound_below",
        "lower_bound_above",
        "lower_bound_nan",
        "scale_nan",
        "scale_float_overflow",
        "null_k",
        "invalid_k_contract",
        "unsupported_variant",
    ]
    output = "\n".join(
        f"negative_case={label} status=161002 workspace_size=0 executor=(nil)"
        for label in labels
    )
    output += (
        "\nnegative_case=null_outputs workspace_status=161002 "
        "executor_status=161002 workspace_size=0 executor=(nil)"
    )
    statuses = runner._parse_negative_matrix(output)
    assert len(statuses) == 10
    assert set(statuses.values()) == {161002}


def test_representative_samples_deduplicate_single_head_shape():
    assert runner._representative_samples(1, 1) == ["0:0"]
    assert runner._representative_samples(2, 96) == ["0:0", "1:95"]


def test_runtime_environment_prioritizes_extracted_opapi(tmp_path, monkeypatch):
    opapi = tmp_path / "package/op_api/lib/libcust_opapi.so"
    opbase = tmp_path / "opbase/libnnopbase.so"
    extra = tmp_path / "extra"
    runtime = tmp_path / "runtime"
    for path in (opapi.parent, opbase.parent, extra, runtime):
        path.mkdir(parents=True, exist_ok=True)
    opapi.touch()
    opbase.touch()
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    monkeypatch.setenv("PYTHONPATH", "/existing-python")

    environment = runner._runtime_environment(
        opapi=opapi,
        opbase=opbase,
        library_dirs=[extra],
        runtime_root=runtime,
    )

    assert environment["LD_LIBRARY_PATH"].split(":")[:3] == [
        str(opapi.parent.resolve()),
        str(opbase.parent.resolve()),
        str(extra.resolve()),
    ]
    assert environment["PYTHONPATH"].split(":") == [
        str(runtime.resolve()),
        "/existing-python",
    ]
    assert os.environ["LD_LIBRARY_PATH"] == "/existing"


def test_runtime_linkage_rejects_any_unresolved_library(tmp_path, monkeypatch):
    smoke = tmp_path / "smoke"
    opapi = tmp_path / "libcust_opapi.so"
    opbase = tmp_path / "libnnopbase.so"
    for path in (smoke, opapi, opbase):
        path.touch()
    monkeypatch.setattr(
        runner,
        "_run",
        lambda *args, **kwargs: (
            f"libcust_opapi.so => {opapi}\n"
            f"libnnopbase.so => {opbase}\n"
            "libopapi_math.so => not found\n"
        ),
    )

    with pytest.raises(AssertionError, match="libopapi_math"):
        runner._verify_runtime_linkage(smoke, opapi, opbase, {})
