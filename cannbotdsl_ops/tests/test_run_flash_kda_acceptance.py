from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from cannbotdsl.package.errors import PackageIntegrityError


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


def _rule(
    *,
    case_id,
    batch_slot,
    head_slot,
    sequence_slot,
    variant_id="auto_grouped",
    content_key="c" * 64,
):
    static_values = {"dv_base": 128, "group": 4}
    return {
        "variant_id": variant_id,
        "physical_key": "1" * 64,
        "variant_key": dict(static_values),
        "case_id": case_id,
        "predicate": {
            "tensors": {
                "V": {
                    "dtype": "float16",
                    "shape": [
                        batch_slot,
                        head_slot,
                        sequence_slot,
                        {"equal": 128},
                    ],
                }
            },
            "attrs": {},
        },
        "static_values": dict(static_values),
        "content_key": content_key,
    }


def test_delivery_regions_consume_canonical_rules_without_reopening_json(monkeypatch):
    rules = (
        _rule(
            case_id="case_z",
            batch_slot={"equal": 1},
            head_slot={"min": 3, "max": 8, "multiple_of": 4},
            sequence_slot={"equal": 16},
        ),
        _rule(
            case_id="case_a",
            batch_slot={"min": 3, "max": 8, "multiple_of": 2},
            head_slot={"equal": 32},
            sequence_slot={"min": 17, "max": 64, "multiple_of": 16},
        ),
    )

    def forbidden_json_load(*args, **kwargs):
        raise AssertionError("acceptance must not load raw manifest JSON")

    monkeypatch.setattr(runner.json, "loads", forbidden_json_load)

    regions = runner._delivery_regions(rules)

    assert [(item.variant_id, item.case_id) for item in regions] == [
        ("auto_grouped", "case_a"),
        ("auto_grouped", "case_z"),
    ]
    assert [(item.batch, item.heads, item.sequence_min) for item in regions] == [
        (4, 32, 32),
        (1, 4, 16),
    ]
    assert all(item.content_key == "c" * 64 for item in regions)
    assert all(
        item.static_values == (("dv_base", 128), ("group", 4))
        for item in regions
    )
    assert len({item.variant_id for item in regions}) == 1
    assert len(regions) == 2


def test_delivery_dump_paths_encode_variant_and_case_without_separator_collisions(
    tmp_path,
):
    regions = runner._delivery_regions(
        (
            _rule(
                variant_id="auto-a",
                case_id="case",
                batch_slot={"equal": 1},
                head_slot={"equal": 1},
                sequence_slot={"equal": 16},
            ),
            _rule(
                variant_id="auto",
                case_id="a-case",
                batch_slot={"equal": 2},
                head_slot={"equal": 1},
                sequence_slot={"equal": 16},
            ),
        )
    )

    paths = [runner._delivery_dump_path(tmp_path, region) for region in regions]

    assert [path.relative_to(tmp_path).parts for path in paths] == [
        ("dump-auto", "a-case"),
        ("dump-auto-a", "case"),
    ]
    assert len(set(paths)) == 2


@pytest.mark.parametrize(
    ("variant_id", "case_id"),
    [
        (".", "case"),
        ("..", "case"),
        ("auto", "."),
        ("auto", ".."),
        ("auto/escape", "case"),
        ("auto", "case/escape"),
    ],
)
def test_delivery_dump_path_rejects_path_control_components(
    tmp_path, variant_id, case_id
):
    region = runner.DeliveryRegion(
        variant_id=variant_id,
        case_id=case_id,
        content_key="c" * 64,
        static_values=(("dv_base", 128), ("group", 4)),
        batch=1,
        heads=1,
        sequence_min=16,
    )

    with pytest.raises(ValueError, match="path component"):
        runner._delivery_dump_path(tmp_path, region)


def test_delivery_dump_path_stays_two_levels_below_resolved_work_root(tmp_path):
    region = runner.DeliveryRegion(
        variant_id="auto_safe",
        case_id="case_safe",
        content_key="c" * 64,
        static_values=(("dv_base", 128), ("group", 4)),
        batch=1,
        heads=1,
        sequence_min=16,
    )

    result = runner._delivery_dump_path(tmp_path, region)

    assert result.is_absolute()
    assert result.relative_to(tmp_path.resolve()).parts == (
        "dump-auto_safe",
        "case_safe",
    )


def test_delivery_dump_path_rejects_existing_symlink_escape(tmp_path):
    outside = tmp_path.parent / "task5-outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "dump-auto_safe").symlink_to(outside, target_is_directory=True)
    region = runner.DeliveryRegion(
        variant_id="auto_safe",
        case_id="case_safe",
        content_key="c" * 64,
        static_values=(("dv_base", 128), ("group", 4)),
        batch=1,
        heads=1,
        sequence_min=16,
    )

    with pytest.raises(ValueError, match="escapes work root"):
        runner._delivery_dump_path(tmp_path, region)


def test_acceptance_stops_at_shared_verifier_failure_before_device_work(
    tmp_path, monkeypatch
):
    run_file = tmp_path / "flash-kda.run"
    smoke = tmp_path / "smoke"
    opbase = tmp_path / "libnnopbase.so"
    report = tmp_path / "report.json"
    runtime_root = tmp_path / "runtime"
    (runtime_root / "cannbotdsl/package").mkdir(parents=True)
    (runtime_root / "cannbotdsl/package/cli.py").touch()
    for path in (run_file, smoke, opbase):
        path.touch()

    def reject_corruption(*args, **kwargs):
        raise PackageIntegrityError("VariantManifest predicate overlap")

    def forbidden_device_work(*args, **kwargs):
        raise AssertionError("device work started after package rejection")

    monkeypatch.setattr(runner, "verify_run_package", reject_corruption)
    monkeypatch.setattr(runner, "_run", forbidden_device_work)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(runner._RUNNER_PATH) if hasattr(runner, "_RUNNER_PATH") else "accept",
            "--run",
            str(run_file),
            "--smoke",
            str(smoke),
            "--opbase-library",
            str(opbase),
            "--cannbotdsl-source",
            str(tmp_path / "cannbotdsl-source"),
            "--runtime-root",
            str(runtime_root),
            "--report",
            str(report),
        ],
    )

    assert runner.main() == 1
    assert "VariantManifest predicate overlap" in json.loads(
        report.read_text(encoding="utf-8")
    )["error"]


def test_acceptance_report_and_dump_paths_retain_case_and_content_identity(
    tmp_path, monkeypatch
):
    run_file = tmp_path / "flash-kda.run"
    smoke = tmp_path / "smoke"
    opbase = tmp_path / "libnnopbase.so"
    opapi = tmp_path / "libcust_opapi.so"
    report = tmp_path / "report.json"
    runtime_root = tmp_path / "runtime"
    (runtime_root / "cannbotdsl/package").mkdir(parents=True)
    (runtime_root / "cannbotdsl/package/cli.py").touch()
    for path in (run_file, smoke, opbase, opapi):
        path.touch()
    rules = (
        _rule(
            case_id="case_a",
            batch_slot={"equal": 1},
            head_slot={"equal": 1},
            sequence_slot={"equal": 16},
        ),
        _rule(
            case_id="case_b",
            batch_slot={"equal": 1},
            head_slot={"equal": 1},
            sequence_slot={"equal": 32},
        ),
    )
    verified = SimpleNamespace(
        report={"opapi": "package/op_api/lib/libcust_opapi.so"},
        vendor_root=tmp_path / "vendor",
        opapi=opapi,
        variant_rules=rules,
    )
    commands = []

    negative_labels = [
        "lower_bound_below",
        "lower_bound_above",
        "lower_bound_nan",
        "scale_nan",
        "scale_float_overflow",
        "null_k",
        "invalid_k_contract",
        "unsupported_variant",
    ]

    def fake_run(command, *, environment, description):
        commands.append((command, description))
        if description.startswith("full CPU golden"):
            return (
                "validated_samples=1; output max_abs=0 max_rel=0; "
                "state max_abs=0 max_rel=0"
            )
        if description.startswith("representative native parity"):
            return (
                "sample=0:0 native/golden output max_abs=0; state max_abs=0; "
                "aclnn/native output max_abs=0; state max_abs=0"
            )
        if description.startswith("two-stream soak"):
            negative = "\n".join(
                f"negative_case={label} status=161002 workspace_size=0 "
                "executor=(nil)"
                for label in negative_labels
            )
            negative += (
                "\nnegative_case=null_outputs workspace_status=161002 "
                "executor_status=161002 workspace_size=0 executor=(nil)"
            )
            return _smoke_output(streams=2, waves=1) + "\n" + negative
        return _smoke_output(streams=1, waves=1)

    monkeypatch.setattr(runner, "verify_run_package", lambda *args, **kwargs: verified)
    monkeypatch.setattr(runner, "_runtime_environment", lambda **kwargs: {})
    monkeypatch.setattr(runner, "_verify_runtime_linkage", lambda *args: {})
    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "_sha256", lambda path: "d" * 64)
    monkeypatch.setattr(runner, "_git_head", lambda path: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accept",
            "--run",
            str(run_file),
            "--smoke",
            str(smoke),
            "--opbase-library",
            str(opbase),
            "--cannbotdsl-source",
            str(tmp_path / "cannbotdsl-source"),
            "--runtime-root",
            str(runtime_root),
            "--report",
            str(report),
            "--timing-waves",
            "1",
            "--soak-waves",
            "1",
        ],
    )

    assert runner.main() == 0

    result = json.loads(report.read_text(encoding="utf-8"))
    assert [item["case_id"] for item in result["variants"]] == [
        "case_a",
        "case_b",
    ]
    assert {item["content_key"] for item in result["variants"]} == {"c" * 64}
    assert all(
        item["generated_values"] == {"dv_base": 128, "group": 4}
        for item in result["variants"]
    )
    dump_paths = {
        Path(command[command.index("--dump-dir") + 1]).parts[-2:]
        for command, _ in commands
        if "--dump-dir" in command
    }
    assert dump_paths == {
        ("dump-auto_grouped", "case_a"),
        ("dump-auto_grouped", "case_b"),
    }
