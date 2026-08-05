#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# python3 mc2/mega_moe/tools/critical_path_trace/run_critical_path_trace.py \
#   --trace-profile a8w4 \
#   --test-script /path/to/moe_test/mega_moe/mega_moe.py \
#   --artifact-dir /tmp/mega-moe-a8w4-check \
#   --allow-compatible-test-script \
#   --check-only
# python3 mc2/mega_moe/tools/critical_path_trace/run_critical_path_trace.py \
#   --trace-profile a8w4 \
#   --test-script /path/to/moe_test/mega_moe/mega_moe.py \
#   --artifact-dir /tmp/mega-moe-a8w4-trace \
#   --allow-compatible-test-script \
#   --iterations 100

"""Adapt the current moe_test runner, capture raw y bytes, validate, and render."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

from mega_moe_trace_contract import (
    ContractError,
    validate_capture_contract,
    validate_timestamp_order,
)


ADAPTER_VERSION = 4
TESTED_MOE_TEST_COMMIT = "90319bbd05993d2fbe7674d693395d5520732553"
TESTED_SCRIPT_SHA256 = (
    "35f600ca443782fd8f210a33af879af651a9dea011f32afa51191ae2bf9ee43e"
)
TESTED_GRAPH_SHA256 = "f050d695c5d4bde708e71f501ee587268c05ccbfef934b61f95dcd0b7e1fa13f"
TRACE_ENV = "MEGA_MOE_CRITICAL_PATH_TRACE_CAPTURE"
TRACE_PROFILE_ENV = "MEGA_MOE_CRITICAL_PATH_TRACE_PROFILE"
ARTIFACT_ENV = "MEGA_MOE_CRITICAL_PATH_TRACE_ARTIFACT_DIR"
SOURCE_SHA_ENV = "MEGA_MOE_CRITICAL_PATH_TRACE_SOURCE_SHA256"
ORIGINAL_DIR_ENV = "MEGA_MOE_CRITICAL_PATH_TRACE_ORIGINAL_SCRIPT_DIR"

USER_OVERRIDABLE = {
    "A8W8_is_NZ",
    "bs",
    "comm_quant_mode",
    "core_num",
    "graph_type",
    "h",
    "is_full_core",
    "is_interleaved",
    "k",
    "local_moe_expert_num",
    "n",
    "quant_dtype",
    "quant_mode",
    "rank_per_dev",
    "server_num",
    "shared_expert_num",
    "swiglu_quant_dtype",
    "topk_weights_type",
    "tp_world_size",
    "weight_dtype",
}
DEFAULT_TRACE_ITERATIONS = 100
FORCED_OVERRIDES = {
    "is_performance": 0,
    "loop_acc_time": DEFAULT_TRACE_ITERATIONS,
}
TRACE_SWITCH_DEFAULTS = {
    "MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE": 1,
    "MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE": 1,
}

HELPER_INSERT_ANCHOR = "\ndef get_cascade_graph_model_class():\n"
SCRIPT_DIR_ANCHOR = "script_dir = os.path.dirname(os.path.abspath(__file__))"
CACHE_DIR_ANCHOR = 'cache_dir = os.path.join(script_dir, "torch_extensions_cache")'
GOLDEN_CACHE_DIR_ANCHOR = 'golden_cache_dir = os.path.join(script_dir, "golden_cache")'
FINAL_SYNC_ANCHOR = (
    "    torch.npu.synchronize()\n"
    "    print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \\n')"
)
MAIN_COMPARE_ANCHOR = "    result_mega_moe = {}"

CAPTURE_HELPER = r"""

_critical_path_trace_capacity = {}


def _critical_path_trace_validate_capacity():
    global _critical_path_trace_capacity
    if out_dtype != torch.bfloat16:
        raise RuntimeError(
            f"critical-path trace requires BF16 y carrier, got out_dtype={out_dtype}"
        )
    expert_count = int(local_moe_expert_num)
    aic_count = int(aic_core_num)
    output_bytes = int(bs) * int(h) * torch.empty((), dtype=out_dtype).element_size()
    trace_profile = os.environ["MEGA_MOE_CRITICAL_PATH_TRACE_PROFILE"]
    if trace_profile == "a8w4":
        protocol_version = 1
        stage_count = 3
    elif trace_profile == "a8w8-wave":
        protocol_version = 7
        stage_count = 2
    else:
        raise RuntimeError(f"unsupported critical-path trace profile: {trace_profile}")
    record_words = 2 + 2 * stage_count * expert_count
    required_y_bytes = 4 * (30 + 3 * aic_count * record_words)
    scratch_bytes = 4 * 3 * aic_count * stage_count * 3 * expert_count
    scratch_reserve_bytes = 50 * 1024 * 1024
    _critical_path_trace_capacity = {
        "trace_profile": trace_profile,
        "protocol_version": protocol_version,
        "formula": "4 * (30 + 3 * aic_core_num * (2 + 2 * stage_count * local_moe_expert_num))",
        "stage_count": stage_count,
        "bs": int(bs),
        "h": int(h),
        "aic_core_num": aic_count,
        "local_moe_expert_num": expert_count,
        "record_words": record_words,
        "required_y_bytes": required_y_bytes,
        "available_y_bytes": output_bytes,
        "scratch_required_bytes": scratch_bytes,
        "scratch_reserve_bytes": scratch_reserve_bytes,
    }
    if expert_count <= 0 or aic_count <= 0:
        raise RuntimeError(
            "critical-path trace requires positive local_moe_expert_num and aic_core_num"
        )
    if required_y_bytes > output_bytes:
        raise RuntimeError(
            "critical-path trace y capacity is insufficient: "
            f"required={required_y_bytes} bytes, available={output_bytes} bytes; "
            f"record_words=2+2*{stage_count}*{expert_count}={record_words}, AIC={aic_count}"
        )
    if scratch_bytes > scratch_reserve_bytes:
        raise RuntimeError(
            "critical-path trace workspace scratch is insufficient: "
            f"required={scratch_bytes} bytes, reserve={scratch_reserve_bytes} bytes"
        )
    print(
        "[TRACE] capacity preflight passed: "
        f"y={required_y_bytes}/{output_bytes} bytes, "
        f"workspace-scratch={scratch_bytes}/{scratch_reserve_bytes} bytes, "
        f"experts={expert_count}, AIC={aic_count}"
    )

def _critical_path_trace_atomic_write(path, data, binary=False):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    mode = "wb" if binary else "w"
    kwargs = {} if binary else {"encoding": "utf-8"}
    with open(tmp_path, mode, **kwargs) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp_path, path)


def _critical_path_trace_capture_y(y, rank):
    artifact_dir = os.environ["MEGA_MOE_CRITICAL_PATH_TRACE_ARTIFACT_DIR"]
    os.makedirs(artifact_dir, exist_ok=True)
    y_cpu = y.detach().cpu().contiguous()
    raw_bytes = y_cpu.view(torch.uint8).numpy().tobytes(order="C")
    raw_path = os.path.join(artifact_dir, f"rank{rank}.raw")
    metadata_path = os.path.join(artifact_dir, f"rank{rank}.runtime.json")
    metadata = {
        "adapter_version": 4,
        "capture_policy": "last_iteration_only",
        "executed_iterations": int(loop_acc_time),
        "captured_iteration": int(loop_acc_time),
        "rank": int(rank),
        "bs": int(bs),
        "h": int(h),
        "n": int(n),
        "k": int(k),
        "local_moe_expert_num": int(local_moe_expert_num),
        "shared_expert_num": int(shared_expert_num),
        "world_size": int(world_size),
        "aic_core_num": int(aic_core_num),
        "dispatch_quant_out_type": int(dispatch_quant_out_type),
        "comm_quant_mode": int(comm_quant_mode),
        "topk_weights_type": bool(topk_weights_type),
        "is_interleaved": bool(is_interleaved),
        "quant_mode": int(quant_mode),
        "quant_dtype": str(quant_dtype),
        "swiglu_quant_dtype": str(swiglu_quant_dtype),
        "weight_dtype": str(weight_dtype),
        "graph_type": int(graph_type),
        "output_shape": list(y_cpu.shape),
        "output_dtype": str(y_cpu.dtype),
        "output_nbytes": len(raw_bytes),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "test_source_sha256": os.environ["MEGA_MOE_CRITICAL_PATH_TRACE_SOURCE_SHA256"],
        "capacity_preflight": _critical_path_trace_capacity,
    }
    _critical_path_trace_atomic_write(raw_path, raw_bytes, binary=True)
    _critical_path_trace_atomic_write(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(f"[TRACE] rank {rank}: captured {len(raw_bytes)} raw y bytes to {raw_path}")
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_checked(
    command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    print("[RUN] " + " ".join(command), flush=True)
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    except FileNotFoundError as exc:
        raise ContractError(f"command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise ContractError(
            f"command failed with exit code {exc.returncode}: {' '.join(command)}"
        ) from exc


def git_output(path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def parse_override(text: str) -> tuple[str, bool | int | str]:
    name, separator, value_text = text.partition("=")
    name = name.strip()
    value_text = value_text.strip()
    if not separator or not name or not value_text:
        raise ContractError(f"--set requires NAME=VALUE, got {text!r}")
    if name not in USER_OVERRIDABLE:
        allowed = ", ".join(sorted(USER_OVERRIDABLE))
        raise ContractError(f"--set cannot change {name!r}; allowed names: {allowed}")
    normalized = {"true": "True", "false": "False"}.get(value_text.lower(), value_text)
    try:
        value = ast.literal_eval(normalized)
    except (SyntaxError, ValueError):
        if re.fullmatch(r"[A-Za-z0-9_.-]+", value_text):
            value = value_text
        else:
            raise ContractError(f"unsafe or unsupported --set value: {value_text!r}")
    if isinstance(value, bool):
        return name, value
    if isinstance(value, int) and not isinstance(value, bool):
        return name, value
    if isinstance(value, str):
        return name, value
    raise ContractError(
        f"--set supports only bool, int, or string values, got {value!r}"
    )


def replace_top_level_assignments(
    source: str, overrides: dict[str, bool | int | str]
) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ContractError(f"moe_test source is not valid Python: {exc}") from exc
    assignments: dict[str, ast.Assign] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in overrides:
            if target.id in assignments:
                raise ContractError(f"top-level assignment {target.id!r} is not unique")
            assignments[target.id] = node
    missing = sorted(set(overrides) - set(assignments))
    if missing:
        raise ContractError(
            f"moe_test source lacks required assignments: {', '.join(missing)}"
        )

    lines = source.splitlines(keepends=True)
    replacements: list[tuple[int, int, str]] = []
    for name, value in overrides.items():
        node = assignments[name]
        if node.end_lineno is None:
            raise ContractError(f"cannot resolve assignment extent for {name}")
        replacement = f"{name} = {value!r}  # critical-path trace adapter\n"
        replacements.append((node.lineno - 1, node.end_lineno, replacement))
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    return "".join(lines)


def replace_once(source: str, anchor: str, replacement: str, description: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise ContractError(
            f"{description} anchor must occur exactly once, found {count}: {anchor!r}"
        )
    return source.replace(anchor, replacement, 1)


def insert_after_top_level_assignment(source: str, name: str, insertion: str) -> str:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(matches) != 1 or matches[0].end_lineno is None:
        raise ContractError(
            f"top-level assignment {name!r} must occur exactly once for capacity preflight"
        )
    lines = source.splitlines(keepends=True)
    lines[matches[0].end_lineno : matches[0].end_lineno] = [insertion]
    return "".join(lines)


def adapt_test_source(
    source: str,
    *,
    original_script_dir: Path,
    overrides: dict[str, bool | int | str],
) -> str:
    transformed = replace_top_level_assignments(source, overrides)
    transformed = replace_once(
        transformed,
        SCRIPT_DIR_ANCHOR,
        f"script_dir = os.environ[{ORIGINAL_DIR_ENV!r}]",
        "script directory",
    )
    transformed = replace_once(
        transformed,
        CACHE_DIR_ANCHOR,
        f"cache_dir = os.path.join(os.environ[{ARTIFACT_ENV!r}], 'runtime-cache', 'torch_extensions')",
        "Torch extension cache directory",
    )
    transformed = replace_once(
        transformed,
        GOLDEN_CACHE_DIR_ANCHOR,
        f"golden_cache_dir = os.path.join(os.environ[{ARTIFACT_ENV!r}], 'runtime-cache', 'golden_cache')",
        "golden cache directory",
    )
    transformed = replace_once(
        transformed,
        HELPER_INSERT_ANCHOR,
        CAPTURE_HELPER + HELPER_INSERT_ANCHOR,
        "capture helper insertion",
    )
    transformed = insert_after_top_level_assignment(
        transformed,
        "out_dtype",
        f"\nif os.environ.get({TRACE_ENV!r}) == '1':\n"
        "    _critical_path_trace_validate_capacity()\n",
    )
    capture_and_return = (
        "    torch.npu.synchronize()\n"
        f"    if os.environ.get({TRACE_ENV!r}) == '1':\n"
        "        _critical_path_trace_capture_y(y, rank)\n"
        "        queue.put([rank, [None, None, None, None]])\n"
        "        return\n"
        "    print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \\n')"
    )
    transformed = replace_once(
        transformed,
        FINAL_SYNC_ANCHOR,
        capture_and_return,
        "post-MegaMoe synchronize/capture",
    )
    diagnostic_exit = (
        f"    if os.environ.get({TRACE_ENV!r}) == '1':\n"
        "        print('[TRACE] all local rank captures completed; accuracy comparison is intentionally skipped')\n"
        "        raise SystemExit(0)\n\n" + MAIN_COMPARE_ANCHOR
    )
    transformed = replace_once(
        transformed,
        MAIN_COMPARE_ANCHOR,
        diagnostic_exit,
        "accuracy-comparison bypass",
    )
    try:
        compile(transformed, "adapted_mega_moe.py", "exec")
    except SyntaxError as exc:
        raise ContractError(
            f"adapted moe_test source is invalid Python: {exc}"
        ) from exc
    return transformed


def validate_test_provenance(script: Path, *, allow_compatible: bool) -> dict[str, Any]:
    if not script.is_file():
        raise ContractError(f"moe_test script does not exist: {script}")
    script_sha = sha256_file(script)
    graph_path = script.with_name("graph.py")
    graph_sha = sha256_file(graph_path) if graph_path.is_file() else None
    git_root_text = git_output(script.parent, "rev-parse", "--show-toplevel")
    git_root = Path(git_root_text) if git_root_text else None
    git_head = git_output(script.parent, "rev-parse", "HEAD")
    relative_script = (
        str(script.resolve().relative_to(git_root.resolve()))
        if git_root
        else str(script)
    )
    tracked_status = (
        git_output(git_root, "status", "--short", "--", relative_script)
        if git_root
        else None
    )
    exact = (
        script_sha == TESTED_SCRIPT_SHA256
        and graph_sha == TESTED_GRAPH_SHA256
        and git_head == TESTED_MOE_TEST_COMMIT
        and not tracked_status
    )
    if not exact and not allow_compatible:
        raise ContractError(
            "moe_test provenance differs from the audited snapshot; expected commit "
            f"{TESTED_MOE_TEST_COMMIT}, mega_moe.py {TESTED_SCRIPT_SHA256}, graph.py "
            f"{TESTED_GRAPH_SHA256}, clean tracked script; got commit {git_head}, "
            f"mega_moe.py {script_sha}, graph.py {graph_sha}, status {tracked_status!r}. "
            "Use --allow-compatible-test-script only after reviewing adapter anchors and current call ABI."
        )
    return {
        "exact_audited_snapshot": exact,
        "script": str(script.resolve()),
        "script_sha256": script_sha,
        "graph_sha256": graph_sha,
        "git_root": str(git_root) if git_root else None,
        "git_head": git_head,
        "tracked_script_status": tracked_status,
        "audited_commit": TESTED_MOE_TEST_COMMIT,
        "audited_script_sha256": TESTED_SCRIPT_SHA256,
        "audited_graph_sha256": TESTED_GRAPH_SHA256,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_artifact_destination(
    artifact_dir: Path, repo_root: Path, *, decode_only: bool
) -> None:
    if decode_only:
        return
    try:
        artifact_dir.relative_to(repo_root)
    except ValueError:
        return
    raise ContractError(
        f"artifact directory must be outside the source repository: {artifact_dir}. "
        "Run the adapter from a separate trace workspace or pass an external "
        "--artifact-dir."
    )


def validate_trace_source_defaults(repo_root: Path) -> dict[str, Any]:
    header = repo_root / "mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h"
    if not header.is_file():
        raise ContractError(f"trace source header is missing: {header}")
    source = header.read_text(encoding="utf-8")
    for name, expected in TRACE_SWITCH_DEFAULTS.items():
        pattern = re.compile(
            rf"^#ifndef {re.escape(name)}\s*$\n"
            rf"^#define {re.escape(name)} {expected}\s*$",
            re.MULTILINE,
        )
        matches = pattern.findall(source)
        if len(matches) != 1:
            raise ContractError(
                f"trace source must define {name}={expected} exactly once behind its #ifndef; "
                f"found {len(matches)} matches in {header}"
            )
    return {
        "header": str(header),
        "header_sha256": sha256_file(header),
        "defaults": TRACE_SWITCH_DEFAULTS,
        "note": "runtime header remains authoritative for the actually installed kernel build",
    }


def select_capture_spec(
    raw_path: Path,
    candidates: list[tuple[Path, Path]],
) -> tuple[Path, Path]:
    raw_prefix = raw_path.read_bytes()[:12]
    if len(raw_prefix) < 12:
        raise ContractError(f"{raw_path} is too short to identify its trace protocol")
    runtime_version = int.from_bytes(raw_prefix[4:8], "little")
    runtime_schema = int.from_bytes(raw_prefix[8:12], "little")
    for spec_path, audit_path in candidates:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        protocol = spec["trace"]["protocol"]
        if (
            int(protocol["version"]) == runtime_version
            and int(protocol["schema_id"], 0) == runtime_schema
        ):
            return spec_path, audit_path
    expected = ", ".join(
        f"v{json.loads(spec_path.read_text(encoding='utf-8'))['trace']['protocol']['version']}/"
        f"{json.loads(spec_path.read_text(encoding='utf-8'))['trace']['protocol']['schema_id']}"
        for spec_path, _ in candidates
    )
    raise ContractError(
        f"runtime header/config mismatch version={runtime_version}, "
        f"schema_id={runtime_schema} (0x{runtime_schema:08x}); expected one of {expected}"
    )


def decode_artifacts(
    *,
    artifact_dir: Path,
    spec_path: Path,
    audit_path: Path,
    trace_tool: Path,
    python: str,
    allow_truncated: bool,
    expected_source_sha: str | None,
    coarse_spec_path: Path | None = None,
    coarse_audit_path: Path | None = None,
    a8w4_spec_path: Path | None = None,
    a8w4_audit_path: Path | None = None,
) -> list[dict[str, Any]]:
    raw_paths = sorted(
        artifact_dir.glob("rank*.raw"),
        key=lambda path: int(re.fullmatch(r"rank(\d+)\.raw", path.name).group(1))
        if re.fullmatch(r"rank(\d+)\.raw", path.name)
        else 1 << 30,
    )
    if not raw_paths:
        raise ContractError(f"no rank*.raw captures found in {artifact_dir}")
    results: list[dict[str, Any]] = []
    for raw_path in raw_paths:
        match = re.fullmatch(r"rank(\d+)\.raw", raw_path.name)
        if not match:
            continue
        rank = int(match.group(1))
        candidates = [(spec_path, audit_path)]
        if coarse_spec_path is not None and coarse_audit_path is not None:
            candidates.append((coarse_spec_path, coarse_audit_path))
        if a8w4_spec_path is not None and a8w4_audit_path is not None:
            candidates.append((a8w4_spec_path, a8w4_audit_path))
        active_spec_path, active_audit_path = select_capture_spec(raw_path, candidates)
        runtime_path = artifact_dir / f"rank{rank}.runtime.json"
        contract, expected = validate_capture_contract(
            raw_path=raw_path,
            metadata_path=runtime_path,
            spec_path=active_spec_path,
            allow_truncated=allow_truncated,
        )
        if expected_source_sha is not None:
            actual_source_sha = contract["metadata"].get("test_source_sha256")
            if actual_source_sha != expected_source_sha:
                raise ContractError(
                    f"rank {rank} metadata source SHA {actual_source_sha!r} does not match "
                    f"adapted source input {expected_source_sha}"
                )
        contract_path = artifact_dir / f"rank{rank}.contract-check.json"
        write_json(contract_path, {"status": "passed", **contract})

        normalized_path = artifact_dir / f"rank{rank}.normalized.json"
        html_path = artifact_dir / f"rank{rank}.html"
        active_spec = json.loads(active_spec_path.read_text(encoding="utf-8"))
        process_spans = [
            span
            for span in active_spec["spans"]
            if span.get("stage") == "process" or span["id"].endswith("process")
        ]
        if len(process_spans) != 1:
            raise ContractError("active trace spec must define exactly one process span")
        decode_command = [
            python,
            str(trace_tool),
            "decode",
            "--spec",
            str(active_spec_path),
            "--raw",
            str(raw_path),
            "--output",
            str(normalized_path),
            "--reference-role",
            "AIC",
            "--reference-core",
            "0",
            "--reference-point",
            process_spans[0]["begin"],
            "--require-flags-mask",
            f"0x{expected['flags']:x}",
            "--require-flags-value",
            f"0x{expected['flags']:x}",
        ]
        for name, value in expected.items():
            if name in {
                "aic_count",
                "magic",
                "version",
                "schema_id",
                "header_words",
                "record_words",
            }:
                continue
            decode_command.extend(["--expect-header", f"{name}={value}"])
        for role in ("AIC", "AIV0", "AIV1"):
            decode_command.extend(
                ["--expect-role-count", f"{role}={expected['aic_count']}"]
            )
        if allow_truncated:
            decode_command.append("--allow-truncated")
        run_checked(decode_command)
        order_check = validate_timestamp_order(
            normalized_path, active_spec_path, allow_truncated=allow_truncated
        )
        order_path = artifact_dir / f"rank{rank}.order-check.json"
        write_json(order_path, order_check)
        render_command = [
            python,
            str(trace_tool),
            "render",
            "--spec",
            str(active_spec_path),
            "--trace",
            str(normalized_path),
            "--audit",
            str(active_audit_path),
            "--output",
            str(html_path),
        ]
        run_checked(render_command)
        results.append(
            {
                "rank": rank,
                "raw": raw_path.name,
                "runtime_metadata": runtime_path.name,
                "contract_check": contract_path.name,
                "normalized": normalized_path.name,
                "order_check": order_path.name,
                "html": html_path.name,
                "raw_sha256": contract["raw_sha256"],
                "required_trace_bytes": contract["required_trace_bytes"],
                "available_carrier_bytes": contract["available_carrier_bytes"],
                "protocol_version": expected["version"],
                "schema_id": f"0x{expected['schema_id']:08x}",
                "trace_spec": active_spec_path.name,
            }
        )
    return results


def write_index(
    artifact_dir: Path, results: list[dict[str, Any]], manifest: dict[str, Any]
) -> None:
    rows = []
    for result in results:
        rows.append(
            "<tr>"
            f"<td>{result['rank']}</td>"
            f'<td><a href="{html.escape(result["html"])}">时间轴</a></td>'
            f'<td><a href="{html.escape(result["normalized"])}">标准化 JSON</a></td>'
            f'<td><a href="{html.escape(result["contract_check"])}">协议检查</a></td>'
            f'<td><a href="{html.escape(result["order_check"])}">顺序检查</a></td>'
            f"<td>{result['required_trace_bytes']} / {result['available_carrier_bytes']}</td>"
            "</tr>"
        )
    provenance = manifest.get("test_provenance", {})
    capture_kind = manifest.get("capture_kind", "device runtime capture")
    document = (
        '<!doctype html><html><head><meta charset="utf-8"><title>MegaMoe 关键路径 tracing</title>'
        "<style>body{font-family:system-ui;margin:24px;color:#0f172a}table{border-collapse:collapse}"
        "th,td{border:1px solid #cbd5e1;padding:7px;text-align:left}th{background:#f1f5f9}"
        "code{white-space:pre-wrap}</style></head><body>"
        "<h1>MegaMoe 关键路径 tracing 结果</h1>"
        f"<p><strong>采集类型：</strong>{html.escape(str(capture_kind))}</p>"
        "<p>诊断采集会覆盖 <code>y</code>，因此有意跳过精度比较。</p>"
        f"<p>moe_test commit：<code>{html.escape(str(provenance.get('git_head')))}</code>；"
        f"源码 SHA-256：<code>{html.escape(str(provenance.get('script_sha256')))}</code>。</p>"
        "<table><thead><tr><th>Rank</th><th>HTML</th><th>Trace JSON</th>"
        "<th>Header/配置</th><th>顺序/边界</th><th>所需/载体字节</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )
    (artifact_dir / "index.html").write_text(document, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[4]
    default_test = Path("/home/mace/workspace/moe_test/mega_moe/mega_moe.py")
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_artifact_dir = Path.cwd() / "artifacts" / timestamp
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed adapter for the audited moe_test MegaMoe runner. It captures raw y "
            "before masking/comparison, validates the runtime header/config, and renders HTML."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--test-script", type=Path, default=default_test)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=default_artifact_dir,
        help=(
            "external output directory; defaults to "
            f"{default_artifact_dir} based on the current working directory"
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--trace-profile",
        choices=("a8w8-wave", "a8w4"),
        default="a8w8-wave",
        help="select the kernel trace path and its capacity/protocol contract",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override one audited top-level moe_test case variable; repeat as needed",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_TRACE_ITERATIONS,
        help=(
            "run this many synchronized MegaMoe iterations and capture only the final "
            f"iteration (default: {DEFAULT_TRACE_ITERATIONS})"
        ),
    )
    parser.add_argument(
        "--allow-compatible-test-script",
        action="store_true",
        help="accept a non-identical moe_test snapshot only if all fail-closed anchors still match",
    )
    parser.add_argument(
        "--allow-truncated",
        action="store_true",
        help="accept intentional partial capture from the legacy coarse v5 protocol",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate provenance and adapter anchors, write the adapted script, but do not execute",
    )
    parser.add_argument(
        "--decode-only",
        action="store_true",
        help="decode existing rank*.raw/rank*.runtime.json files without running moe_test",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    validate_artifact_destination(artifact_dir, repo_root, decode_only=args.decode_only)
    spec_path = repo_root / "mc2/mega_moe/tools/critical_path_trace/trace-spec.json"
    a8w4_spec_path = repo_root / "mc2/mega_moe/tools/critical_path_trace/trace-spec-a8w4.json"
    coarse_spec_path = (
        repo_root / "mc2/mega_moe/tools/critical_path_trace/trace-spec-coarse-v5.json"
    )
    trace_tool = repo_root / "mc2/mega_moe/tools/critical_path_trace/trace_tool.py"
    audit_path = (
        repo_root
        / "mc2/mega_moe/tools/critical_path_trace/audit-implemented/trace_point_audit.json"
    )
    coarse_audit_path = (
        repo_root
        / "mc2/mega_moe/tools/critical_path_trace/audit-implemented-coarse-v5/trace_point_audit.json"
    )
    a8w4_audit_path = (
        repo_root
        / "mc2/mega_moe/tools/critical_path_trace/audit-implemented-a8w4/trace_point_audit.json"
    )
    for required in (
        spec_path,
        coarse_spec_path,
        a8w4_spec_path,
        trace_tool,
        audit_path,
        coarse_audit_path,
        a8w4_audit_path,
    ):
        if not required.is_file():
            raise ContractError(f"required repository artifact is missing: {required}")
    source_defaults = validate_trace_source_defaults(repo_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "adapter_version": ADAPTER_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "repo_commit": git_output(repo_root, "rev-parse", "HEAD"),
        "artifact_dir": str(artifact_dir),
        "trace_spec": str(spec_path),
        "trace_spec_sha256": sha256_file(spec_path),
        "coarse_trace_spec": str(coarse_spec_path),
        "coarse_trace_spec_sha256": sha256_file(coarse_spec_path),
        "a8w4_trace_spec": str(a8w4_spec_path),
        "a8w4_trace_spec_sha256": sha256_file(a8w4_spec_path),
        "trace_tool_sha256": sha256_file(trace_tool),
        "trace_source_defaults": source_defaults,
        "forced_overrides": FORCED_OVERRIDES,
        "user_overrides": {},
        "mode": "decode-only"
        if args.decode_only
        else "check-only"
        if args.check_only
        else "run",
        "trace_profile": args.trace_profile,
    }
    expected_source_sha: str | None = None

    if not args.decode_only:
        if args.iterations < 1:
            raise ContractError(f"--iterations must be >= 1, got {args.iterations}")
        script = args.test_script.expanduser().resolve()
        provenance = validate_test_provenance(
            script, allow_compatible=args.allow_compatible_test_script
        )
        manifest["test_provenance"] = provenance
        expected_source_sha = provenance["script_sha256"]
        forced_overrides = {
            **FORCED_OVERRIDES,
            "loop_acc_time": args.iterations,
        }
        manifest["forced_overrides"] = forced_overrides
        manifest["capture_policy"] = {
            "executed_iterations": args.iterations,
            "captured_iteration": args.iterations,
            "selection": "last_iteration_only",
            "per_iteration_host_sync": False,
        }
        overrides: dict[str, bool | int | str] = dict(forced_overrides)
        for item in args.set:
            name, value = parse_override(item)
            if name in overrides:
                raise ContractError(
                    f"--set repeats or conflicts with forced variable {name!r}"
                )
            overrides[name] = value
            manifest["user_overrides"][name] = value
        source = script.read_text(encoding="utf-8")
        transformed = adapt_test_source(
            source,
            original_script_dir=script.parent,
            overrides=overrides,
        )
        adapted_path = artifact_dir / "adapted_mega_moe.py"
        adapted_path.write_text(transformed, encoding="utf-8")
        manifest["adapted_script"] = str(adapted_path)
        manifest["adapted_script_sha256"] = sha256_file(adapted_path)
        write_json(artifact_dir / "run-manifest.json", manifest)
        print(
            f"[OK] moe_test adapter anchors/provenance passed; adapted source: {adapted_path}"
        )
        if args.check_only:
            return 0

        env = os.environ.copy()
        env[TRACE_ENV] = "1"
        env[TRACE_PROFILE_ENV] = args.trace_profile
        env[ARTIFACT_ENV] = str(artifact_dir)
        env[SOURCE_SHA_ENV] = expected_source_sha
        env[ORIGINAL_DIR_ENV] = str(script.parent)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(script.parent)
            if not existing_pythonpath
            else str(script.parent) + os.pathsep + existing_pythonpath
        )
        run_checked([args.python, str(adapted_path)], cwd=artifact_dir, env=env)

    results = decode_artifacts(
        artifact_dir=artifact_dir,
        spec_path=spec_path,
        audit_path=audit_path,
        trace_tool=trace_tool,
        python=args.python,
        allow_truncated=args.allow_truncated,
        expected_source_sha=expected_source_sha,
        coarse_spec_path=coarse_spec_path,
        coarse_audit_path=coarse_audit_path,
        a8w4_spec_path=a8w4_spec_path,
        a8w4_audit_path=a8w4_audit_path,
    )
    manifest["captures"] = results
    manifest["status"] = "passed"
    write_json(artifact_dir / "run-manifest.json", manifest)
    write_index(artifact_dir, results, manifest)
    print(
        f"[OK] validated and rendered {len(results)} rank capture(s); open {artifact_dir / 'index.html'}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
