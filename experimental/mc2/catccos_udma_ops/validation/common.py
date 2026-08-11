# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Shared helpers for validation launchers and workers."""

from __future__ import annotations

import csv
import datetime
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union


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
