# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Shared helpers for validation launchers and workers."""

from __future__ import annotations

import csv
import datetime
import json
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union


_MIB = 1 << 20
_DEFAULT_MTE_HEAP_BYTES = 1 << 30
_MTE_KERNEL_WORKSPACE_BYTES = 400 * _MIB + (128 << 10)
_MTE_HEAP_HEADROOM_BYTES = 64 * _MIB


def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def write_json(path: Union[Path, str], payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(
    path: Union[Path, str], rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mte_init_method_from_env() -> str:
    """Build the numeric ``tcp://IPv4:port`` URI accepted by SHMEM."""
    host = os.environ.get("MTE_MASTER_ADDR", "127.0.0.1")
    try:
        address = socket.gethostbyname(host)
    except OSError as error:
        raise RuntimeError(
            "Unable to resolve MTE_MASTER_ADDR={!r} to IPv4".format(host)
        ) from error
    port = int(
        os.environ.get(
            "MTE_MASTER_PORT", str(int(os.environ.get("MASTER_PORT", "29500")) + 1)
        )
    )
    if port <= 0 or port > 65535:
        raise RuntimeError("MTE_MASTER_PORT must be in [1, 65535]")
    return "tcp://{}:{}".format(address, port)


def validation_mte_heap_size() -> int:
    """Size SHMEM for the fixed remote-receive workspace and synchronization state."""
    minimum = _MTE_KERNEL_WORKSPACE_BYTES + _MTE_HEAP_HEADROOM_BYTES

    configured = os.environ.get("MTE_HEAP_SIZE")
    if configured is None:
        return _DEFAULT_MTE_HEAP_BYTES
    try:
        requested = int(configured)
    except ValueError as error:
        raise RuntimeError("MTE_HEAP_SIZE must be an integer number of bytes") from error
    if requested < minimum:
        raise RuntimeError(
            "MTE_HEAP_SIZE={} is too small for the MTE remote-receive workspace; "
            "at least {} bytes are required (default size: {})".format(
                requested, minimum, _DEFAULT_MTE_HEAP_BYTES
            )
        )
    return requested


def environment_fingerprint() -> Dict[str, Any]:
    ascend_home = (
        os.environ.get("ASCEND_HOME_PATH")
        or os.environ.get("ASCEND_TOOLKIT_HOME")
        or os.environ.get("_ASCEND_INSTALL_PATH")
    )
    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": None,
        "torch_npu": None,
        "operator_package": None,
        "communication_backend": "mte",
        "cann_install_path": ascend_home,
        "cann_version": None,
        "soc_version": os.environ.get("SOC_VERSION", "Ascend950"),
    }
    try:
        import torch

        result["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        import torch_npu

        result["torch_npu"] = getattr(torch_npu, "__version__", None)
    except ImportError:
        pass
    try:
        import ops_transformer_catccos_mte as mte

        result["operator_package"] = getattr(mte, "__version__", None)
        result["shmem_commit"] = mte.SHMEM_COMMIT
        result["catccos_commit"] = mte.CATCCOS_COMMIT
    except ImportError:
        pass
    if ascend_home:
        for relative in (
            "version.cfg",
            "compiler/version.info",
            "x86_64-linux/version.info",
            "aarch64-linux/version.info",
        ):
            version_path = Path(ascend_home) / relative
            if version_path.is_file():
                result["cann_version"] = version_path.read_text(
                    encoding="utf-8", errors="replace"
                ).strip()[:4096]
                break
    return result
