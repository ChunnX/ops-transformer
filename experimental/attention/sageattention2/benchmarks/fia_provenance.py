# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Resolve the FIA libraries selected by the installed torch_npu API loader."""

import ctypes
import hashlib
import os
from pathlib import Path

import torch_npu  # noqa: F401 -- registers the NPU backend


class DlInfo(ctypes.Structure):
    _fields_ = [
        ("filename", ctypes.c_char_p),
        ("base", ctypes.c_void_p),
        ("symbol", ctypes.c_char_p),
        ("address", ctypes.c_void_p),
    ]


def fia_provenance(expected_library=None):
    # torch_npu 2.9 exposes its existing resolver. This only queries symbols;
    # execution and allocation still use the public PyTorch operator.
    library = Path(torch_npu.__file__).parent / "lib/libtorch_npu.so"
    loader = ctypes.CDLL(str(library))
    resolve = loader._Z16GetOpApiFuncAddrPKc
    resolve.argtypes = [ctypes.c_char_p]
    resolve.restype = ctypes.c_void_p
    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    providers = {}
    for version in range(2, 6):
        for suffix in ("", "GetWorkspaceSize"):
            name = f"aclnnFusedInferAttentionScoreV{version}{suffix}"
            address = resolve(name.encode())
            info = DlInfo()
            if (
                not address
                or not dladdr(address, ctypes.byref(info))
                or not info.filename
            ):
                raise RuntimeError(f"Cannot identify the library providing {name}")
            path = Path(info.filename.decode()).resolve()
            if expected_library is not None and path != expected_library.resolve():
                raise RuntimeError(
                    f"{name} resolves to {path}; expected {expected_library.resolve()}"
                )
            providers[name] = str(path)
    return {
        "scope": "API libraries only; device kernel paths need separate runtime-log verification",
        "api_providers": providers,
        "library_sha256": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in sorted(set(providers.values()))
        },
        "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH", ""),
    }
