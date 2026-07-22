#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Executable level-3 capability probe and reproducible runtime evidence."""

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

import torch
import torch_npu


SCHEMA_VERSION = 1
KIND = "smla_mqsmla_level3_runtime_capability"
REQUESTED_LEVEL = 3
SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
ERROR = "ERROR"
MAX_ATTEMPTS = 100


class RuntimeCapabilityError(ValueError):
    """Raised when a capability report is invalid or cannot be loaded."""


def is_level3_runtime_rejection(error: BaseException) -> bool:
    """Recognize the installed runtime's delayed level-3 rejection."""
    message = str(error).lower()
    return (
        "only supports configuring 0/1/2" in message
        or (
            "deterministic level" in message
            and "level 3" in message
            and ("not support" in message or "unsupported" in message)
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_evidence(path: Path, include_text: bool = False) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
    if not evidence["exists"]:
        return evidence
    try:
        evidence["size_bytes"] = path.stat().st_size
        evidence["sha256"] = _sha256_file(path)
        if include_text:
            evidence["content"] = path.read_text(encoding="utf-8", errors="replace")[:65536]
    except OSError as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
    return evidence


def _tree_evidence(root: Path) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {
        "root": str(root.resolve()),
        "exists": root.is_dir(),
        "source_commit_claim": None,
    }
    if not evidence["exists"]:
        evidence["identity_status"] = "ROOT_NOT_FOUND"
        return evidence

    manifest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    errors: List[Dict[str, str]] = []
    selected = {
        "version.info",
        "op_api/lib/libcust_opapi.so",
        "op_impl/cpu/aicpu_kernel/impl/libtransformer_aicpu_kernels.so",
        "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so",
        (
            "op_impl/ai_core/tbe/custom_transformer_impl/ascendc/"
            "mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla.cpp"
        ),
        (
            "op_impl/ai_core/tbe/custom_transformer_impl/ascendc/"
            "mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla_template_tiling_key.h"
        ),
        (
            "op_impl/ai_core/tbe/custom_transformer_impl/ascendc/"
            "sparse_flash_mla/sparse_flash_mla.cpp"
        ),
    }
    selected_evidence = []
    try:
        paths = sorted(
            (path for path in root.rglob("*") if path.is_file() or path.is_symlink()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    except OSError as error:
        evidence["identity_status"] = "TREE_ENUMERATION_FAILED"
        evidence["errors"] = [{"path": str(root), "error": f"{type(error).__name__}: {error}"}]
        return evidence

    for path in paths:
        relative = path.relative_to(root).as_posix()
        try:
            if path.is_symlink():
                target = os.readlink(path)
                record = f"L\0{relative}\0{target}\n".encode("utf-8")
                manifest.update(record)
                continue
            size = path.stat().st_size
            file_hash = _sha256_file(path)
            manifest.update(f"F\0{relative}\0{size}\0{file_hash}\n".encode("utf-8"))
            file_count += 1
            total_bytes += size
            if relative in selected:
                selected_evidence.append(
                    {"relative_path": relative, "size_bytes": size, "sha256": file_hash}
                )
        except OSError as error:
            errors.append({"path": relative, "error": f"{type(error).__name__}: {error}"})

    version = _file_evidence(root / "version.info", include_text=True)
    content = str(version.get("content") or "")
    for line in content.splitlines():
        key, separator, value = line.partition("=")
        if separator and "commit" in key.lower() and value.strip():
            evidence["source_commit_claim"] = value.strip()
            break
    evidence.update(
        {
            "identity_status": (
                "CONTENT_HASHED_SOURCE_COMMIT_RECORDED"
                if evidence["source_commit_claim"]
                else "CONTENT_HASHED_SOURCE_COMMIT_UNPROVEN"
            ),
            "content_manifest_sha256": manifest.hexdigest(),
            "file_count": file_count,
            "total_bytes": total_bytes,
            "version_info": version,
            "selected_files": selected_evidence,
            "errors": errors,
        }
    )
    return evidence


def _version_files() -> List[Dict[str, Any]]:
    candidates = []
    ascend_home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if ascend_home:
        root = Path(ascend_home)
        candidates.extend(
            [root / "version.info", root / "compiler/version.info", root / "opp/version.info"]
        )
    driver_root = Path(os.environ.get("ASCEND_DRIVER_PATH", "/usr/local/Ascend/driver"))
    candidates.append(driver_root / "version.info")
    unique = []
    seen = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        item = _file_evidence(path, include_text=True)
        if item["exists"]:
            unique.append(item)
    return unique


def _custom_opp_evidence() -> List[Dict[str, Any]]:
    roots = []
    seen = set()
    for raw in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
        if not raw:
            continue
        root = Path(raw).resolve()
        if str(root) in seen:
            continue
        seen.add(str(root))
        roots.append(_tree_evidence(root))
    return roots


def collect_environment_evidence(device_id: int) -> Dict[str, Any]:
    """Collect version and package content evidence without claiming a source commit."""
    device: Dict[str, Any] = {"requested_device_id": device_id}
    try:
        device["visible_device_count"] = int(torch_npu.npu.device_count())
        device["name"] = str(torch_npu.npu.get_device_name(device_id))
        properties = torch_npu.npu.get_device_properties(device_id)
        for name in (
            "total_memory",
            "cube_core_num",
            "vector_core_num",
            "L2_cache_size",
        ):
            if hasattr(properties, name):
                device[name] = int(getattr(properties, name))
    except BaseException as error:
        device["error"] = f"{type(error).__name__}: {error}"

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_npu_version": str(torch_npu.__version__),
        "device": device,
        "paths": {
            name: os.environ.get(name)
            for name in (
                "ASCEND_HOME_PATH",
                "ASCEND_TOOLKIT_HOME",
                "ASCEND_OPP_PATH",
                "ASCEND_CUSTOM_OPP_PATH",
            )
        },
        "runtime_version_files": _version_files(),
        "custom_opp_packages": _custom_opp_evidence(),
    }


class RuntimeCapabilityProbe:
    """Prove level 3 by set, readback, real NPU dispatch, and restoration."""

    def __init__(
        self,
        device_id: int = 0,
        *,
        npu: Any = None,
        smoke: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        self.device_id = device_id
        self.npu = npu if npu is not None else torch_npu.npu
        self.smoke = smoke if smoke is not None else self._default_smoke

    def _current_level(self) -> int:
        getter = getattr(self.npu, "_get_deterministic_level", None)
        if getter is None:
            raise RuntimeCapabilityError("deterministic-level getter is unavailable")
        return int(getter())

    def _set_strict(self, level: int) -> int:
        self.npu.set_deterministic_level(level)
        actual = self._current_level()
        if actual != level:
            raise RuntimeCapabilityError(
                f"set_deterministic_level({level}) read back {actual}"
            )
        return actual

    def _default_smoke(self) -> Dict[str, Any]:
        tensor = torch.empty((1,), dtype=torch.float32).npu()
        self.npu.synchronize()
        return {"op": "torch.empty((1,), float32).npu + synchronize", "shape": list(tensor.shape)}

    @staticmethod
    def _reason_code(stage: str, known_rejection: bool) -> str:
        if stage == "set_level" and known_rejection:
            return "LEVEL3_SET_REJECTED"
        if stage == "npu_smoke" and known_rejection:
            return "LEVEL3_ACCEPTED_BUT_NPU_REJECTED"
        return "LEVEL3_CAPABILITY_PROBE_ERROR"

    def run_attempt(self, index: int = 1) -> Dict[str, Any]:
        started = time.monotonic()
        result: Dict[str, Any] = {
            "attempt": index,
            "requested_level": REQUESTED_LEVEL,
            "set_call_returned": False,
            "readback_matches": False,
            "npu_smoke_passed": False,
            "levels": {"before": None, "observed": None, "restored": None},
            "restoration": {"attempted": False, "status": "NOT_REQUIRED"},
        }
        previous: Optional[int] = None
        stage = "read_initial_level"
        try:
            previous = self._current_level()
            result["levels"]["before"] = previous
            stage = "set_level"
            self.npu.set_deterministic_level(REQUESTED_LEVEL)
            result["set_call_returned"] = True
            stage = "readback_level"
            observed = self._current_level()
            result["levels"]["observed"] = observed
            if observed != REQUESTED_LEVEL:
                raise RuntimeCapabilityError(
                    f"requested level {REQUESTED_LEVEL}, observed level {observed}"
                )
            result["readback_matches"] = True
            stage = "npu_smoke"
            result["npu_smoke"] = self.smoke()
            result["npu_smoke_passed"] = True
            result.update(
                {"status": SUPPORTED, "failure_stage": None, "reason_code": None, "reason": None}
            )
        except BaseException as error:
            known_rejection = is_level3_runtime_rejection(error)
            result.update(
                {
                    "status": UNSUPPORTED if known_rejection else ERROR,
                    "failure_stage": stage,
                    "reason_code": self._reason_code(stage, known_rejection),
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            if previous is not None:
                result["restoration"] = {"attempted": True, "status": "ERROR"}
                try:
                    restored = self._set_strict(previous)
                    result["levels"]["restored"] = restored
                    result["restoration"]["status"] = "PASS"
                except BaseException as restore_error:
                    result["restoration"]["reason"] = (
                        f"{type(restore_error).__name__}: {restore_error}"
                    )
                    result.update(
                        {
                            "status": ERROR,
                            "failure_stage": "restore_level",
                            "reason_code": "LEVEL_RESTORE_FAILED",
                            "reason": result["restoration"]["reason"],
                        }
                    )
            result["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
        return result

    def run(self, attempts: int = 1) -> Dict[str, Any]:
        if attempts < 1 or attempts > MAX_ATTEMPTS:
            raise ValueError(f"attempts must be in [1, {MAX_ATTEMPTS}]")
        device_init: Dict[str, Any] = {"device_id": self.device_id}
        records = []
        try:
            self.npu.set_device(self.device_id)
            device_init["status"] = "PASS"
            records = [self.run_attempt(index) for index in range(1, attempts + 1)]
        except BaseException as error:
            device_init.update(
                {"status": "ERROR", "reason": f"{type(error).__name__}: {error}"}
            )

        statuses = [record["status"] for record in records]
        if records and all(status == SUPPORTED for status in statuses):
            status = SUPPORTED
            failure_stage = reason_code = reason = None
        elif records and all(status == UNSUPPORTED for status in statuses):
            status = UNSUPPORTED
            first = records[0]
            same_reason = all(
                record["reason_code"] == first["reason_code"] for record in records
            )
            failure_stage = first["failure_stage"] if same_reason else "mixed"
            reason_code = first["reason_code"] if same_reason else "LEVEL3_RUNTIME_REJECTED"
            reason = first["reason"]
        else:
            status = ERROR
            first = next((record for record in records if record["status"] == ERROR), None)
            failure_stage = first["failure_stage"] if first else "device_init"
            reason_code = first["reason_code"] if first else "DEVICE_INIT_FAILED"
            reason = first["reason"] if first else device_init.get("reason")

        return {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "claim_scope": (
                "Runtime level-3 dispatch capability only; operator smoke and consistency remain separate gates"
            ),
            "requested_level": REQUESTED_LEVEL,
            "status": status,
            "failure_stage": failure_stage,
            "reason_code": reason_code,
            "reason": reason,
            "attempt_count_requested": attempts,
            "attempt_count_executed": len(records),
            "restoration_pass": bool(records) and all(
                record["restoration"]["status"] == "PASS" for record in records
            ),
            "device_init": device_init,
            "attempts": records,
            "environment": collect_environment_evidence(self.device_id),
        }


def validate_runtime_capability_report(report: Dict[str, Any]) -> None:
    if not isinstance(report, dict):
        raise RuntimeCapabilityError("Runtime capability report root must be an object")
    if report.get("schema_version") != SCHEMA_VERSION or report.get("kind") != KIND:
        raise RuntimeCapabilityError("Unexpected runtime capability schema or kind")
    if report.get("requested_level") != REQUESTED_LEVEL:
        raise RuntimeCapabilityError("Runtime capability report did not request level 3")
    if report.get("status") not in (SUPPORTED, UNSUPPORTED, ERROR):
        raise RuntimeCapabilityError("Runtime capability status is invalid")
    attempts = report.get("attempts")
    if not isinstance(attempts, list):
        raise RuntimeCapabilityError("Runtime capability attempts must be a list")
    if report.get("attempt_count_executed") != len(attempts):
        raise RuntimeCapabilityError("Runtime capability attempt count is inconsistent")
    if not isinstance(report.get("environment"), dict):
        raise RuntimeCapabilityError("Runtime capability environment evidence is missing")
    if report["status"] == SUPPORTED:
        if not attempts or not report.get("restoration_pass"):
            raise RuntimeCapabilityError("SUPPORTED requires an executed and restored probe")
        if any(
            attempt.get("status") != SUPPORTED
            or not attempt.get("readback_matches")
            or not attempt.get("npu_smoke_passed")
            for attempt in attempts
        ):
            raise RuntimeCapabilityError("SUPPORTED requires every NPU smoke to pass")
    else:
        for field in ("failure_stage", "reason_code", "reason"):
            if not isinstance(report.get(field), str) or not report[field]:
                raise RuntimeCapabilityError(f"{report['status']} requires {field}")


def _atomic_json_dump(path: str, value: Dict[str, Any]) -> None:
    absolute = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(absolute)}.", dir=os.path.dirname(absolute)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary, absolute)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


@lru_cache(maxsize=8)
def load_runtime_capability_evidence(path: str) -> Dict[str, Any]:
    absolute = os.path.abspath(path)
    try:
        with open(absolute, "r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeCapabilityError(f"Cannot load capability report {absolute}: {error}") from error
    validate_runtime_capability_report(report)
    canonical = json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "source_report": {
            "path": absolute,
            "sha256": _sha256_file(Path(absolute)),
        },
        "probe_sha256": hashlib.sha256(canonical).hexdigest(),
        "probe": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe executable deterministic level-3 support")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=1)
    args = parser.parse_args()
    report = RuntimeCapabilityProbe(args.device_id).run(args.attempts)
    validate_runtime_capability_report(report)
    _atomic_json_dump(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "reason_code": report["reason_code"],
                "failure_stage": report["failure_stage"],
                "attempts": report["attempt_count_executed"],
                "restoration_pass": report["restoration_pass"],
                "report": os.path.abspath(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if report["status"] == SUPPORTED:
        return 0
    if report["status"] == UNSUPPORTED:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
