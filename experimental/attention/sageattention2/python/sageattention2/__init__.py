"""
SageAttention2 Python package for Ascend NPU (Atlas 950PR).
"""
import ctypes
import os
from . import _extension, _ops
from .dispatch import (
    run_sageattn_cpu_reference,
    sageattn,
    sageattn_qk_int8_pv_fp8_asc,
)
from .monkey_patch import (
    is_sdpa_patched,
    replace_sdpa,
    restore_sdpa,
    sdpa_patch,
)


def enable_npu_op_timeout(seconds: int = 15) -> bool:
    """Configures Ascend NPU hardware kernel watchdog timeout via AscendCL runtime.

    Calls aclrtSetOpExecuteTimeOut(ctypes.c_uint32(seconds)) to ensure that any
    hardware kernel deadlocks or execution stalls raise RuntimeError instead of
    hanging the process indefinitely.

    Returns True if successfully configured, False if libascendcl is unavailable
    or the runtime call failed.
    """
    if seconds is None:
        try:
            seconds = int(os.environ.get("SAGE_OP_TIMEOUT", "15"))
        except (ValueError, TypeError):
            seconds = 15
    else:
        try:
            seconds = int(seconds)
        except (ValueError, TypeError):
            seconds = 15

    libascendcl = None
    for name in ("libascendcl.so", "libascendcl.dylib"):
        try:
            libascendcl = ctypes.CDLL(name)
            break
        except Exception:
            pass

    if libascendcl is None:
        for env_key in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH", "CANN_ROOT"):
            root = os.environ.get(env_key)
            if not root:
                continue
            for sub in ("lib64", "x86_64-linux/lib64", "aarch64-linux/lib64", "lib"):
                candidate = os.path.join(root, sub, "libascendcl.so")
                if os.path.isfile(candidate):
                    try:
                        libascendcl = ctypes.CDLL(candidate)
                        break
                    except Exception:
                        pass
            if libascendcl is not None:
                break

    if libascendcl is None:
        common_candidates = [
            "/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so",
            "/usr/local/Ascend/cann/lib64/libascendcl.so",
            "/usr/local/Ascend/latest/lib64/libascendcl.so",
        ]
        for candidate in common_candidates:
            if os.path.isfile(candidate):
                try:
                    libascendcl = ctypes.CDLL(candidate)
                    break
                except Exception:
                    pass

    if libascendcl is not None and hasattr(libascendcl, "aclrtSetOpExecuteTimeOut"):
        try:
            libascendcl.aclrtSetOpExecuteTimeOut.argtypes = [ctypes.c_uint32]
            libascendcl.aclrtSetOpExecuteTimeOut.restype = ctypes.c_int
            ret = libascendcl.aclrtSetOpExecuteTimeOut(ctypes.c_uint32(seconds))
            return ret == 0
        except Exception:
            return False

    return False


# Auto-configure op execute timeout watchdog on import
try:
    _init_timeout = int(os.environ.get("SAGE_OP_TIMEOUT", "15"))
    enable_npu_op_timeout(_init_timeout)
except Exception:
    pass

__all__ = [
    "sageattn",
    "sageattn_qk_int8_pv_fp8_asc",
    "run_sageattn_cpu_reference",
    "replace_sdpa",
    "restore_sdpa",
    "sdpa_patch",
    "is_sdpa_patched",
    "enable_npu_op_timeout",
    "_ops",
    "_extension",
]

