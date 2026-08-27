#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Run the complete FlashKDA mixed-package device acceptance matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


_TEST_DIR = Path(__file__).resolve().parent
_POLICY_DIR = _TEST_DIR.parent / "attention" / "flash_kda"
sys.path.insert(0, str(_TEST_DIR))
sys.path.insert(0, str(_POLICY_DIR))

from flash_kda_variant_policy import FLASH_KDA_VARIANT_POLICIES  # noqa: E402
from verify_mixed_run import verify_run_package  # noqa: E402


_TIMING_PATTERN = re.compile(
    r"timing streams=(?P<streams>\d+) waves=(?P<waves>\d+) "
    r"launch_sync_ms_min=(?P<minimum>[0-9.]+) "
    r"avg=(?P<average>[0-9.]+) max=(?P<maximum>[0-9.]+)"
)
_GOLDEN_PATTERN = re.compile(
    r"validated_samples=(?P<samples>\d+); "
    r"output max_abs=(?P<output_abs>[0-9.eE+-]+) "
    r"max_rel=(?P<output_rel>[0-9.eE+-]+); "
    r"state max_abs=(?P<state_abs>[0-9.eE+-]+) "
    r"max_rel=(?P<state_rel>[0-9.eE+-]+)"
)
_NATIVE_PATTERN = re.compile(
    r"sample=(?P<sample>\d+:\d+) native/golden output max_abs="
    r"(?P<native_output>[0-9.eE+-]+); state max_abs="
    r"(?P<native_state>[0-9.eE+-]+); aclnn/native output max_abs="
    r"(?P<aclnn_output>[0-9.eE+-]+); state max_abs="
    r"(?P<aclnn_state>[0-9.eE+-]+)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, description: str) -> Path:
    result = path.resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{description} does not exist: {result}")
    return result


def _run(command: list[str], *, environment: dict[str, str], description: str) -> str:
    print(f"[acceptance] {description}", flush=True)
    result = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code {result.returncode}")
    return result.stdout


def _parse_smoke(
    output: str, *, expected_streams: int, expected_waves: int
) -> dict[str, Any]:
    timing_matches = list(_TIMING_PATTERN.finditer(output))
    if len(timing_matches) != 1:
        raise AssertionError("smoke output must contain exactly one timing summary")
    timing = timing_matches[0].groupdict()
    if int(timing["streams"]) != expected_streams:
        raise AssertionError("smoke timing stream count drifted")
    if int(timing["waves"]) != expected_waves:
        raise AssertionError("smoke timing wave count drifted")

    expected_invocations = expected_streams * expected_waves
    workspace_matches = re.findall(
        r"workspace_size=(\d+) get_workspace_status=(\d+)", output
    )
    execution_statuses = re.findall(r"execute_status=(\d+)", output)
    finite_matches = re.findall(
        r"output_finite=(true|false) final_state_finite=(true|false)", output
    )
    if (
        len(workspace_matches) != expected_invocations
        or len(execution_statuses) != expected_invocations
        or len(finite_matches) != expected_invocations
    ):
        raise AssertionError("smoke invocation count differs from streams × waves")
    if any(status != "0" for _, status in workspace_matches):
        raise AssertionError("GetWorkspace returned a non-zero status")
    if any(status != "0" for status in execution_statuses):
        raise AssertionError("execution returned a non-zero status")
    if any(values != ("true", "true") for values in finite_matches):
        raise AssertionError("smoke produced a non-finite output")
    workspace_sizes = {int(size) for size, _ in workspace_matches}
    if len(workspace_sizes) != 1:
        raise AssertionError("workspace size changed across repeated invocations")
    return {
        "invocations": expected_invocations,
        "workspace_size": workspace_sizes.pop(),
        "timing_ms": {
            "min": float(timing["minimum"]),
            "avg": float(timing["average"]),
            "max": float(timing["maximum"]),
        },
    }


def _parse_golden(output: str, *, expected_samples: int) -> dict[str, Any]:
    matches = list(_GOLDEN_PATTERN.finditer(output))
    if len(matches) != 1:
        raise AssertionError("golden output must contain exactly one summary")
    values = matches[0].groupdict()
    if int(values["samples"]) != expected_samples:
        raise AssertionError("golden sample count does not cover B × H")
    return {
        "samples": int(values["samples"]),
        "output_max_abs": float(values["output_abs"]),
        "output_max_rel": float(values["output_rel"]),
        "state_max_abs": float(values["state_abs"]),
        "state_max_rel": float(values["state_rel"]),
    }


def _parse_native(output: str, *, expected_samples: list[str]) -> dict[str, Any]:
    matches = [match.groupdict() for match in _NATIVE_PATTERN.finditer(output)]
    if [match["sample"] for match in matches] != expected_samples:
        raise AssertionError("native verification did not cover the requested samples")
    for match in matches:
        if float(match["aclnn_output"]) != 0.0 or float(match["aclnn_state"]) != 0.0:
            raise AssertionError("ACLNN and native outputs are not bitwise identical")
    return {
        "samples": expected_samples,
        "bitwise_equal": True,
        "native_output_max_abs": max(float(item["native_output"]) for item in matches),
        "native_state_max_abs": max(float(item["native_state"]) for item in matches),
    }


def _parse_negative_matrix(output: str) -> dict[str, int]:
    statuses = {
        label: int(status)
        for label, status in re.findall(
            r"negative_case=(\S+) status=(\d+) workspace_size=0 executor=\(nil\)",
            output,
        )
    }
    null_outputs = re.search(
        r"negative_case=null_outputs workspace_status=(\d+) executor_status=(\d+) "
        r"workspace_size=0 executor=\(nil\)",
        output,
    )
    if null_outputs is None:
        raise AssertionError("null output-pointer error cases are missing")
    statuses["null_workspace_size_output"] = int(null_outputs.group(1))
    statuses["null_executor_output"] = int(null_outputs.group(2))
    if len(statuses) != 10 or set(statuses.values()) != {161002}:
        raise AssertionError("GetWorkspace error matrix is incomplete or drifted")
    return statuses


def _representative_samples(batch: int, heads: int) -> list[str]:
    samples = ["0:0", f"{batch - 1}:{heads - 1}"]
    return list(dict.fromkeys(samples))


def _runtime_environment(
    *, opapi: Path, opbase: Path, library_dirs: list[Path], runtime_root: Path
) -> dict[str, str]:
    environment = os.environ.copy()
    search_paths = [opapi.parent, opbase.parent, *library_dirs]
    existing = environment.get("LD_LIBRARY_PATH")
    if existing:
        search_paths.extend(Path(path) for path in existing.split(":") if path)
    unique_paths = list(dict.fromkeys(str(path.resolve()) for path in search_paths))
    environment["LD_LIBRARY_PATH"] = ":".join(unique_paths)
    python_paths = [str(runtime_root.resolve())]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = ":".join(python_paths)
    return environment


def _verify_runtime_linkage(
    smoke: Path, opapi: Path, opbase: Path, environment: dict[str, str]
) -> dict[str, str]:
    output = _run(
        ["ldd", str(smoke)],
        environment=environment,
        description="verify smoke dynamic-library resolution",
    )
    unresolved = re.findall(r"^\s*(\S+) => not found$", output, re.MULTILINE)
    if unresolved:
        raise AssertionError(
            f"smoke has unresolved dynamic libraries: {sorted(unresolved)}"
        )
    expected = {
        "libcust_opapi.so": opapi.resolve(),
        "libnnopbase.so": opbase.resolve(),
    }
    loaded = {}
    for library, expected_path in expected.items():
        match = re.search(rf"{re.escape(library)} => (\S+)", output)
        if match is None:
            raise AssertionError(f"ldd did not resolve {library}")
        actual_path = Path(match.group(1)).resolve()
        if actual_path != expected_path:
            raise AssertionError(
                f"smoke resolves {actual_path}, expected {expected_path}"
            )
        loaded[library] = str(actual_path)
    return loaded


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the FlashKDA mixed-package device acceptance matrix"
    )
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--smoke", required=True, type=Path)
    parser.add_argument("--opbase-library", required=True, type=Path)
    parser.add_argument("--cannbotdsl-source", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--library-dir", action="append", type=Path, default=[])
    parser.add_argument("--timing-waves", type=int, default=3)
    parser.add_argument("--soak-waves", type=int, default=20)
    parser.add_argument("--expected-traditional-kernel-count", type=int, default=8)
    arguments = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": 1,
        "passed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if arguments.timing_waves <= 0 or arguments.soak_waves <= 0:
            raise ValueError("timing and soak waves must be positive")
        run_file = _require_file(arguments.run, "mixed .run package")
        smoke = _require_file(arguments.smoke, "FlashKDA smoke executable")
        opbase = _require_file(arguments.opbase_library, "Provider ABI libnnopbase")
        verifier = _require_file(
            _TEST_DIR / "verify_flash_kda_aclnn_dump.py", "golden verifier"
        )
        runtime_root = arguments.runtime_root.resolve()
        if not (runtime_root / "cannbotdsl/package/cli.py").is_file():
            raise FileNotFoundError(f"invalid CANNBotDSL runtime root: {runtime_root}")
        cannbotdsl_source = arguments.cannbotdsl_source.resolve()

        with tempfile.TemporaryDirectory(prefix="flash-kda-acceptance-") as tmp:
            work_root = Path(tmp)
            extract_root = work_root / "extracted"
            extract_root.mkdir()
            verified_run = verify_run_package(
                run_file,
                extract_root,
                expected_provider_count=len(FLASH_KDA_VARIANT_POLICIES),
                expected_traditional_kernel_count=(
                    arguments.expected_traditional_kernel_count
                ),
            )
            opapi = verified_run.opapi
            environment = _runtime_environment(
                opapi=opapi,
                opbase=opbase,
                library_dirs=arguments.library_dir,
                runtime_root=runtime_root,
            )
            _verify_runtime_linkage(smoke, opapi, opbase, environment)
            runtime_linkage = {
                "libcust_opapi.so": {
                    "source": "extracted_run_package",
                    "path": verified_run.report["opapi"],
                    "exact_resolution_verified": True,
                },
                "libnnopbase.so": {
                    "source": "explicit_provider_abi_v1",
                    "path": str(opbase),
                    "exact_resolution_verified": True,
                },
            }

            standalone_l0_output = _run(
                [
                    str(smoke),
                    "--api",
                    "standalone-l0",
                    "--waves",
                    "1",
                ],
                environment=environment,
                description="standalone FlashKDA L0 ACLNN regression",
            )
            standalone_l0_result = _parse_smoke(
                standalone_l0_output,
                expected_streams=1,
                expected_waves=1,
            )

            variants = []
            for policy in FLASH_KDA_VARIANT_POLICIES:
                dump_dir = work_root / f"dump-{policy.variant_id}"
                dump_dir.mkdir()
                smoke_output = _run(
                    [
                        str(smoke),
                        "--api",
                        "custom-l2",
                        "--batch",
                        str(policy.batch),
                        "--heads",
                        str(policy.heads),
                        "--sequence",
                        str(policy.sequence_min),
                        "--waves",
                        str(arguments.timing_waves),
                        "--dump-dir",
                        str(dump_dir),
                    ],
                    environment=environment,
                    description=f"device smoke {policy.variant_id}",
                )
                smoke_result = _parse_smoke(
                    smoke_output,
                    expected_streams=1,
                    expected_waves=arguments.timing_waves,
                )
                golden_output = _run(
                    [
                        sys.executable,
                        str(verifier),
                        "--dump-dir",
                        str(dump_dir),
                        "--cannbotdsl-source",
                        str(cannbotdsl_source),
                        "--all-golden-samples",
                        "--summary-only",
                    ],
                    environment=environment,
                    description=f"full CPU golden {policy.variant_id}",
                )
                golden_result = _parse_golden(
                    golden_output, expected_samples=policy.batch * policy.heads
                )
                native_samples = _representative_samples(policy.batch, policy.heads)
                native_command = [
                    sys.executable,
                    str(verifier),
                    "--dump-dir",
                    str(dump_dir),
                    "--cannbotdsl-source",
                    str(cannbotdsl_source),
                    "--verify-native",
                    "--summary-only",
                ]
                for sample in native_samples:
                    native_command.extend(["--golden-sample", sample])
                native_output = _run(
                    native_command,
                    environment=environment,
                    description=f"representative native parity {policy.variant_id}",
                )
                native_result = _parse_native(
                    native_output, expected_samples=native_samples
                )
                variants.append(
                    {
                        "variant_id": policy.variant_id,
                        "shape": [
                            policy.batch,
                            policy.heads,
                            policy.sequence_min,
                            128,
                        ],
                        "smoke": smoke_result,
                        "golden": golden_result,
                        "native": native_result,
                    }
                )

            soak_output = _run(
                [
                    str(smoke),
                    "--api",
                    "custom-l2",
                    "--stress",
                    "--negative",
                    "--waves",
                    str(arguments.soak_waves),
                ],
                environment=environment,
                description="two-stream soak and GetWorkspace error matrix",
            )
            soak_result = _parse_smoke(
                soak_output,
                expected_streams=2,
                expected_waves=arguments.soak_waves,
            )
            soak_result["negative_statuses"] = _parse_negative_matrix(soak_output)

            report.update(
                passed=True,
                package=verified_run.report,
                binaries={
                    "run_sha256": _sha256(run_file),
                    "smoke_sha256": _sha256(smoke),
                    "opapi_sha256": _sha256(opapi),
                    "opbase_sha256": _sha256(opbase),
                },
                runtime_linkage=runtime_linkage,
                sources={
                    "ops_transformer_commit": _git_head(_TEST_DIR.parents[1]),
                    "cannbotdsl_commit": _git_head(cannbotdsl_source),
                },
                configuration={
                    "delivery_api": "aclnnCustomFlashKda",
                    "timing_waves": arguments.timing_waves,
                    "soak_streams": 2,
                    "soak_waves": arguments.soak_waves,
                },
                standalone_l0_regression=standalone_l0_result,
                variants=variants,
                soak=soak_result,
            )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        _write_report(arguments.report, report)
        print(report["error"], file=sys.stderr)
        return 1

    _write_report(arguments.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
