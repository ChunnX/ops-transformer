# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Load one communication backend per process before resolving SHMEM symbols."""

import importlib
import sys
import threading

_lock = threading.Lock()
_process_load_lock = sys.__dict__.setdefault("_ops_transformer_catccos_load_lock", threading.RLock())
_loaded = False


def load_extension():
    global _loaded
    with _lock, _process_load_lock:
        if _loaded:
            return
        backend = getattr(sys, "_ops_transformer_catccos_backend", None)
        if backend not in (None, "mte") or "ops_transformer_catccos_udma._C" in sys.modules:
            raise RuntimeError(
                "UDMA and MTE extensions cannot be loaded in the same process: "
                "SHMEM global state and library symbols are shared. Use separate torchrun workers."
            )
        try:
            importlib.import_module("torch_npu")
            importlib.import_module("._C", __package__)
        except ImportError as error:
            raise RuntimeError(
                "The MTE NPU extension is unavailable. Build and install with "
                "scripts/build_wheel.sh --backend mte on the Ascend950 host; "
                "CPU tuning-plan commands do not need this extension."
            ) from error
        sys._ops_transformer_catccos_backend = "mte"
        _loaded = True
