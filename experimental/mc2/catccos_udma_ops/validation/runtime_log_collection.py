# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Incrementally collect SHMEM and CANN runtime logs for validation launches."""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

try:
    from .common import write_json
except ImportError:
    from common import write_json


def _path_is_within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _split_log_path_list(value):
    if not value:
        return []
    return [item for item in value.split(os.pathsep) if item]


def runtime_log_roots(environment, shmem_log_dirs=(), plog_dirs=()):
    """Return de-duplicated SHMEM/CANN log roots for a validation run."""
    home = Path(environment.get("HOME") or Path.home())
    candidates = []

    def add(kind, raw_path, origin):
        if not raw_path:
            return
        expanded = os.path.expandvars(os.path.expanduser(str(raw_path)))
        path = Path(expanded).resolve()
        candidate = {"kind": kind, "path": str(path), "origin": origin}
        for existing in list(candidates):
            if existing["kind"] != kind:
                continue
            existing_path = Path(existing["path"])
            if path == existing_path or _path_is_within(path, existing_path):
                return
            if _path_is_within(existing_path, path):
                candidates.remove(existing)
        candidates.append(candidate)

    for index, path in enumerate(shmem_log_dirs):
        add("shmem", path, "--shmem-log-dir[{}]".format(index))
    for variable in (
        "CATCCOS_UDMA_SHMEM_LOG_DIRS",
        "CATCCOS_UDMA_SYNC_SHMEM_LOG_DIRS",
        "ACLSHMEM_LOG_PATH",
        "SHMEM_LOG_PATH",
    ):
        for path in _split_log_path_list(environment.get(variable)):
            add("shmem", path, variable)
    add("shmem", home / "shmem" / "log", "default_home")

    for index, path in enumerate(plog_dirs):
        add("plog", path, "--plog-dir[{}]".format(index))
    for variable in (
        "CATCCOS_UDMA_PLOG_DIRS",
        "CATCCOS_UDMA_SYNC_PLOG_DIRS",
        "ASCEND_PROCESS_LOG_PATH",
        "ASCEND_LOG_PATH",
    ):
        for path in _split_log_path_list(environment.get(variable)):
            add("plog", path, variable)
    add("plog", home / "ascend" / "log" / "plog", "default_home")
    add("plog", "/var/log/npu/slog/host-0", "default_system")

    return [dict(item, root_index=index) for index, item in enumerate(candidates)]


def snapshot_runtime_logs(log_roots):
    """Record file identities and sizes without copying existing log content."""
    captured_at_ns = time.time_ns()
    files = {}
    roots = []

    def record_file(path, root, root_record):
        try:
            metadata = path.stat()
        except OSError as error:
            root_record["errors"].append(
                {"path": str(path), "error": str(error)}
            )
            return
        if not stat.S_ISREG(metadata.st_mode):
            return
        try:
            relative_path = str(path.relative_to(Path(root["path"])))
        except ValueError:
            relative_path = path.name
        if relative_path in ("", "."):
            relative_path = path.name
        resolved = str(path.resolve())
        files.setdefault(
            resolved,
            {
                "kind": root["kind"],
                "root_index": root["root_index"],
                "root_path": root["path"],
                "relative_path": relative_path,
                "size": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
            },
        )

    for root in log_roots:
        path = Path(root["path"])
        root_record = dict(root, exists=path.exists(), errors=[])
        roots.append(root_record)
        if not root_record["exists"]:
            continue
        if path.is_file():
            record_file(path, root, root_record)
            continue
        if not path.is_dir():
            root_record["errors"].append(
                {"path": str(path), "error": "not a regular file or directory"}
            )
            continue

        def on_walk_error(error):
            root_record["errors"].append(
                {"path": getattr(error, "filename", str(path)), "error": str(error)}
            )

        for directory, _, names in os.walk(path, onerror=on_walk_error):
            for name in names:
                record_file(Path(directory) / name, root, root_record)

    return {
        "captured_at_ns": captured_at_ns,
        "roots": roots,
        "files": files,
    }


def _copy_log_range(source, destination, start, end):
    destination.parent.mkdir(parents=True, exist_ok=True)
    remaining = max(end - start, 0)
    copied = 0
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        input_stream.seek(start)
        while remaining:
            block = input_stream.read(min(remaining, 1 << 20))
            if not block:
                break
            output_stream.write(block)
            copied += len(block)
            remaining -= len(block)
    return copied


def collect_runtime_logs(before, log_roots, output_dir):
    """Copy files or byte ranges created since ``before`` into ``output_dir``."""
    after = snapshot_runtime_logs(log_roots)
    runtime_root = Path(output_dir)
    entries = []
    error_count = sum(
        len(root["errors"]) for root in before["roots"] + after["roots"]
    )
    captured_bytes = 0

    for source_text, current in sorted(after["files"].items()):
        previous = before["files"].get(source_text)
        if previous is None:
            reason = "created"
            byte_start = 0
        else:
            same_identity = (
                current["device"] == previous["device"]
                and current["inode"] == previous["inode"]
            )
            if (
                same_identity
                and current["size"] == previous["size"]
                and current["mtime_ns"] == previous["mtime_ns"]
            ):
                continue
            if same_identity and current["size"] > previous["size"]:
                reason = "appended"
                byte_start = previous["size"]
            elif same_identity and current["size"] == previous["size"]:
                reason = "rewritten_same_size"
                byte_start = 0
            elif same_identity:
                reason = "truncated_or_rewritten"
                byte_start = 0
            else:
                reason = "replaced"
                byte_start = 0

        relative = Path(current["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            relative = Path(Path(source_text).name)
        destination = (
            runtime_root
            / current["kind"]
            / "root_{:02d}".format(current["root_index"])
            / relative
        )
        entry = {
            "kind": current["kind"],
            "source": source_text,
            "destination": str(destination),
            "reason": reason,
            "source_size": current["size"],
            "byte_start": byte_start,
            "byte_end": current["size"],
            "captured_bytes": 0,
            "copied": False,
        }
        try:
            entry["captured_bytes"] = _copy_log_range(
                Path(source_text), destination, byte_start, current["size"]
            )
            entry["copied"] = True
            captured_bytes += entry["captured_bytes"]
        except OSError as error:
            entry["error"] = str(error)
            error_count += 1
        entries.append(entry)

    manifest = {
        "collection_started_at_ns": before["captured_at_ns"],
        "collection_finished_at_ns": after["captured_at_ns"],
        "roots_before": before["roots"],
        "roots_after": after["roots"],
        "captured_file_count": sum(item["copied"] for item in entries),
        "captured_bytes": captured_bytes,
        "error_count": error_count,
        "files": entries,
    }
    manifest_path = runtime_root / "manifest.json"
    write_json(manifest_path, manifest)
    return {
        "manifest": str(manifest_path),
        "captured_file_count": manifest["captured_file_count"],
        "captured_bytes": captured_bytes,
        "error_count": error_count,
    }
