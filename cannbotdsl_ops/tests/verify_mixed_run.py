#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Verify a mixed traditional AscendC and CANNBotDSL Provider run package."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import tempfile
from typing import Any


PUBLIC_PARAMETERS = [
    "K",
    "V",
    "Q",
    "beta",
    "g",
    "A_log",
    "dt_bias",
    "st_in",
    "scale",
    "lower_bound",
    "o",
    "st_out",
]
INTERNAL_PARAMETERS = ["group", "dv_base", "block_num"]
OUTPUT_PARAMETERS = ["o", "st_out"]
REQUIRED_SYMBOLS = [
    "aclnnChunkKdaFwd",
    "aclnnChunkKdaFwdGetWorkspaceSize",
    "aclnnFlashKda",
    "aclnnFlashKdaGetWorkspaceSize",
    "l0op::FlashKda(",
]


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path}: expected a JSON object")
    return value


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_relative(root: pathlib.Path, relative: str) -> pathlib.Path:
    result = (root / relative).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise AssertionError(f"path escapes package root: {relative!r}") from error
    return result


def _verify_checksums(package_root: pathlib.Path) -> int:
    checksum_path = package_root / "checksums.sha256"
    checked = 0
    for line_number, raw_line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            expected, relative = raw_line.split(maxsplit=1)
        except ValueError as error:
            raise AssertionError(
                f"{checksum_path}:{line_number}: malformed checksum"
            ) from error
        relative = relative.lstrip("* ")
        target = _resolve_relative(package_root, relative)
        if not target.is_file():
            raise AssertionError(f"checksum target does not exist: {relative}")
        actual = _sha256(target)
        if actual != expected:
            raise AssertionError(
                f"checksum mismatch for {relative}: {actual} != {expected}"
            )
        checked += 1
    if checked == 0:
        raise AssertionError(f"{checksum_path}: no checksum entries")
    return checked


def _find_vendor_root(extract_root: pathlib.Path) -> pathlib.Path:
    vendors_root = extract_root / "packages/vendors"
    candidates = sorted(
        path for path in vendors_root.glob("*_transformer") if path.is_dir()
    )
    if len(candidates) != 1:
        raise AssertionError(
            f"expected exactly one transformer vendor, found {len(candidates)}"
        )
    return candidates[0]


def _verify_provider_package(
    vendor_root: pathlib.Path,
    package_name: str,
    expected_provider_count: int,
) -> tuple[int, int]:
    cannbotdsl_root = vendor_root / "op_impl/cannbotdsl"
    registry_path = cannbotdsl_root / "registry.json"
    registry = _load_json(registry_path)
    if registry.get("schema_version") != 1:
        raise AssertionError("registry schema_version must be 1")
    packages = registry.get("packages")
    if not isinstance(packages, dict) or package_name not in packages:
        raise AssertionError(f"registry does not contain {package_name!r}")
    package_record = packages[package_name]
    active = package_record.get("active")
    versions = package_record.get("versions")
    if not isinstance(active, str) or not isinstance(versions, dict):
        raise AssertionError(f"invalid registry record for {package_name!r}")
    version_record = versions.get(active)
    if not isinstance(version_record, dict):
        raise AssertionError(f"active package version {active!r} is missing")
    manifest_path = _resolve_relative(cannbotdsl_root, version_record["manifest"])
    manifest = _load_json(manifest_path)
    manifest_digest = _canonical_digest(manifest)
    if manifest_digest != version_record.get("manifest_digest"):
        raise AssertionError("registry manifest digest does not match manifest.json")

    package = manifest.get("package")
    if not isinstance(package, dict) or package.get("name") != package_name:
        raise AssertionError("manifest package identity does not match registry")
    if package.get("version") != active:
        raise AssertionError("manifest version does not match active registry version")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != expected_provider_count:
        raise AssertionError(
            f"expected {expected_provider_count} Provider artifacts, "
            f"found {len(artifacts) if isinstance(artifacts, list) else 'invalid'}"
        )

    provider_binaries: set[pathlib.Path] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise AssertionError(f"artifact {index} is not an object")
        entry = artifact.get("entry")
        if not isinstance(entry, dict) or entry.get("schema_version") != 3:
            raise AssertionError(f"artifact {index} does not carry entry schema v3")
        if entry.get("public_parameters") != PUBLIC_PARAMETERS:
            raise AssertionError(f"artifact {index} public L0 parameters drifted")
        if entry.get("internal_parameters") != INTERNAL_PARAMETERS:
            raise AssertionError(f"artifact {index} internal parameters drifted")
        if entry.get("output_parameters") != OUTPUT_PARAMETERS:
            raise AssertionError(f"artifact {index} output parameters drifted")
        binary = _resolve_relative(manifest_path.parent, artifact["binary"])
        sidecar = _resolve_relative(manifest_path.parent, artifact["sidecar"])
        if _sha256(binary) != artifact.get("binary_sha256"):
            raise AssertionError(f"artifact {index} binary digest mismatch")
        if _sha256(sidecar) != artifact.get("sidecar_sha256"):
            raise AssertionError(f"artifact {index} sidecar digest mismatch")
        provider_binaries.add(binary)
    if len(provider_binaries) != expected_provider_count:
        raise AssertionError("Provider artifacts do not name unique binaries")

    checked_files = _verify_checksums(manifest_path.parent)
    return len(provider_binaries), checked_files


def _verify_mixed_contents(
    vendor_root: pathlib.Path,
    expected_traditional_kernel_count: int,
) -> tuple[int, pathlib.Path]:
    traditional_root = vendor_root / "op_impl/ai_core/tbe/kernel"
    traditional_kernels = sorted(traditional_root.glob("**/*.o"))
    if len(traditional_kernels) != expected_traditional_kernel_count:
        raise AssertionError(
            f"expected {expected_traditional_kernel_count} traditional kernel objects, "
            f"found {len(traditional_kernels)}"
        )
    opapi = vendor_root / "op_api/lib/libcust_opapi.so"
    if not opapi.is_file():
        raise AssertionError(f"missing mixed opapi library: {opapi}")
    symbols = subprocess.run(
        ["nm", "-D", "-C", str(opapi)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for symbol in REQUIRED_SYMBOLS:
        if symbol not in symbols:
            raise AssertionError(f"libcust_opapi.so does not export {symbol!r}")
    return len(traditional_kernels), opapi


def _extract_run(run_file: pathlib.Path, extract_root: pathlib.Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(run_file),
            "--check",
            "--noexec",
            f"--extract={extract_root}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "run package integrity check/extraction failed\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    combined = result.stdout + result.stderr
    if "SHA256 checksums are OK" not in combined:
        raise AssertionError("makeself did not report a successful SHA256 check")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a mixed AscendC and CANNBotDSL Provider .run package"
    )
    parser.add_argument("--run", required=True, type=pathlib.Path)
    parser.add_argument("--package-name", default="ops-transformer-flash-kda")
    parser.add_argument("--expected-provider-count", type=int, default=6)
    parser.add_argument("--expected-traditional-kernel-count", type=int, default=2)
    arguments = parser.parse_args()

    run_file = arguments.run.resolve()
    if not run_file.is_file():
        raise FileNotFoundError(run_file)
    if arguments.expected_provider_count <= 0:
        raise ValueError("expected Provider count must be positive")
    if arguments.expected_traditional_kernel_count <= 0:
        raise ValueError("expected traditional kernel count must be positive")

    with tempfile.TemporaryDirectory(prefix="ops-transformer-mixed-run-") as tmp:
        extract_root = pathlib.Path(tmp)
        _extract_run(run_file, extract_root)
        vendor_root = _find_vendor_root(extract_root)
        provider_count, checked_files = _verify_provider_package(
            vendor_root,
            arguments.package_name,
            arguments.expected_provider_count,
        )
        traditional_count, opapi = _verify_mixed_contents(
            vendor_root, arguments.expected_traditional_kernel_count
        )
        report = {
            "archive_sha256": _sha256(run_file),
            "archive_size": run_file.stat().st_size,
            "checked_package_files": checked_files,
            "opapi": str(opapi.relative_to(extract_root)),
            "provider_binaries": provider_count,
            "traditional_kernel_objects": traditional_count,
            "vendor": vendor_root.name,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
