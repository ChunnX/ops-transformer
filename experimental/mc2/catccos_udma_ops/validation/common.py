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
_DEFAULT_UDMA_HEAP_BYTES = 1 << 30
_UDMA_KERNEL_WORKSPACE_BYTES = 400 * _MIB + 4096
_UDMA_HEAP_HEADROOM_BYTES = 64 * _MIB


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


def udma_init_method_from_env() -> str:
    """Build the numeric ``tcp://IPv4:port`` URI accepted by SHMEM."""
    host = os.environ.get("UDMA_MASTER_ADDR", "127.0.0.1")
    try:
        address = socket.gethostbyname(host)
    except OSError as error:
        raise RuntimeError(
            "Unable to resolve UDMA_MASTER_ADDR={!r} to IPv4".format(host)
        ) from error
    port = int(
        os.environ.get(
            "UDMA_MASTER_PORT", str(int(os.environ.get("MASTER_PORT", "29500")) + 1)
        )
    )
    if port <= 0 or port > 65535:
        raise RuntimeError("UDMA_MASTER_PORT must be in [1, 65535]")
    return "tcp://{}:{}".format(address, port)


def validation_udma_heap_size(
    cases: Sequence[Mapping[str, Any]], *, concurrent_input_buffers: int
) -> int:
    """Size the SHMEM heap for workspace plus the largest validation input(s)."""
    if not cases:
        raise ValueError("at least one validation case is required to size the UDMA heap")
    if concurrent_input_buffers <= 0:
        raise ValueError("concurrent_input_buffers must be positive")

    max_input_bytes = max(
        int(case["shape"]["m"]) * int(case["shape"]["k"]) * 2
        for case in cases
    )
    minimum = (
        _UDMA_KERNEL_WORKSPACE_BYTES
        + concurrent_input_buffers * max_input_bytes
        + _UDMA_HEAP_HEADROOM_BYTES
    )
    automatic = max(_DEFAULT_UDMA_HEAP_BYTES, 1 << (minimum - 1).bit_length())

    configured = os.environ.get("UDMA_HEAP_SIZE")
    if configured is None:
        return automatic
    try:
        requested = int(configured)
    except ValueError as error:
        raise RuntimeError("UDMA_HEAP_SIZE must be an integer number of bytes") from error
    if requested < minimum:
        raise RuntimeError(
            "UDMA_HEAP_SIZE={} is too small for this validation batch; "
            "at least {} bytes are required (automatic size: {})".format(
                requested, minimum, automatic
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
        import ops_transformer_catccos_udma as udma

        result["operator_package"] = getattr(udma, "__version__", None)
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
