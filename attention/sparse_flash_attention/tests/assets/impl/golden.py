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

"""Pytest-backed CPU golden adapter for SparseFlashAttention."""

import faulthandler
import importlib.util
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import torch


PYTEST_GOLDEN_MODULE = None

_DIAGNOSTIC_ENV = "SFA_TTK_IMPORT_DEBUG"
_PREFLIGHT_ENV = "SFA_TTK_IMPORT_PREFLIGHT"
_DIAGNOSTIC_PREFIX = "[SFA-TTK-DIAG]"
_DIAGNOSTIC_FILE_NAME = "ttk-sfa-import-diagnostic.log"
_DIAGNOSTIC_ENV_KEYS = (
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_VISIBLE_DEVICES",
    "ASCEND_DEVICE_ID",
    "CONDA_PREFIX",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONFAULTHANDLER",
    "PYTHONPATH",
    "SFA_TTK_IMPORT_PREFLIGHT",
    "TORCH_DEVICE_BACKEND_AUTOLOAD",
)
_DIAGNOSTIC_LOG_FILE = None


def _diagnostic_enabled():
    value = os.environ.get(_DIAGNOSTIC_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _preflight_enabled():
    if not _diagnostic_enabled():
        return False
    value = os.environ.get(_PREFLIGHT_ENV, "1")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _diagnostic_value(value, limit=2048):
    text = repr(value)
    if len(text) > limit:
        return f"{text[:limit]}...<truncated>"
    return text


def _diagnostic_file():
    global _DIAGNOSTIC_LOG_FILE
    if not _diagnostic_enabled():
        return None
    if _DIAGNOSTIC_LOG_FILE is not None:
        return _DIAGNOSTIC_LOG_FILE
    path = Path.cwd() / _DIAGNOSTIC_FILE_NAME
    try:
        _DIAGNOSTIC_LOG_FILE = path.open("a", encoding="utf-8", buffering=1)
    except OSError as exc:
        print(
            f"{_DIAGNOSTIC_PREFIX} cannot open diagnostic file {path}: {exc}",
            file=sys.stderr,
            flush=True,
        )
    return _DIAGNOSTIC_LOG_FILE


def _diagnostic_write(message, force=False):
    if not force and not _diagnostic_enabled():
        return
    print(message, file=sys.stderr, flush=True)
    log_file = _diagnostic_file()
    if log_file is not None:
        print(message, file=log_file, flush=True)


def _diagnostic_log(event, force=False, **fields):
    """Emit flushed adapter diagnostics that survive TTK child-process failures."""
    if not force and not _diagnostic_enabled():
        return
    context = {
        "event": event,
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
    }
    context.update(fields)
    details = " ".join(f"{key}={_diagnostic_value(value)}" for key, value in context.items())
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    _diagnostic_write(f"{_DIAGNOSTIC_PREFIX} {timestamp} {details}", force=force)


def _module_file(module):
    if module is None:
        return None
    return getattr(module, "__file__", None)


def _find_spec_origin(module_name):
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:
        return f"<find_spec failed: {type(exc).__name__}: {exc}>"
    return None if spec is None else spec.origin


def _enable_fault_handler():
    try:
        output = _diagnostic_file() or sys.stderr
        faulthandler.enable(file=output, all_threads=True)
        _diagnostic_log(
            "faulthandler_ready",
            enabled=faulthandler.is_enabled(),
            output_file=getattr(output, "name", "<stderr>"),
        )
    except Exception as exc:
        _diagnostic_log(
            "faulthandler_enable_failure",
            force=True,
            exception_type=type(exc).__name__,
            exception=str(exc),
        )


def _log_runtime_snapshot(pytest_dir, module_path):
    if not _diagnostic_enabled():
        return
    environment = {key: os.environ.get(key) for key in _DIAGNOSTIC_ENV_KEYS if os.environ.get(key)}
    modules = {
        name: _module_file(sys.modules.get(name))
        for name in ("generate_tensor_data", "sparse_flash_attention_golden", "tensorflow", "torch", "torch_npu")
    }
    module_specs = {
        name: _find_spec_origin(name)
        for name in ("tensorflow", "torch", "torch_npu")
    }
    _diagnostic_log(
        "runtime_snapshot",
        interpreter=sys.executable,
        python_version=sys.version,
        platform=platform.platform(),
        cwd=os.getcwd(),
        torch_version=getattr(torch, "__version__", None),
        pytest_dir=str(pytest_dir),
        module_path=str(module_path),
        modules=modules,
        module_specs=module_specs,
        environment=environment,
        sys_path_head=sys.path[:8],
    )


def _preflight_output(value, limit=2048):
    if not value:
        return ""
    return value[-limit:]


def _log_preflight_stream(stream_name, value):
    if not value:
        return
    limit = 16384
    truncated = len(value) > limit
    excerpt = value[-limit:]
    header = (
        f"{_DIAGNOSTIC_PREFIX} generator_preflight_{stream_name} "
        f"truncated={truncated} bytes={len(value)}"
    )
    _diagnostic_write(f"{header}\n{excerpt.rstrip()}\n{header}_end", force=True)


def _run_generator_preflight(pytest_dir):
    """Probe pytest's import-time native dependencies without risking the TTK worker."""
    if not _preflight_enabled():
        return
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(pytest_dir)!r})\n"
        "import generate_tensor_data\n"
        "print(getattr(generate_tensor_data, '__file__', '<unknown>'))\n"
    )
    environment = os.environ.copy()
    environment["PYTHONFAULTHANDLER"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", code],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _diagnostic_log(
            "generator_preflight_failure",
            force=True,
            exception_type=type(exc).__name__,
            exception=str(exc),
        )
        raise RuntimeError("Cannot complete isolated pytest generator import preflight") from exc

    _diagnostic_log(
        "generator_preflight_complete",
        returncode=result.returncode,
        stdout_tail=_preflight_output(result.stdout),
        stderr_tail=_preflight_output(result.stderr),
    )
    if result.returncode != 0:
        _log_preflight_stream("stdout", result.stdout)
        _log_preflight_stream("stderr", result.stderr)
        raise RuntimeError(
            "Isolated pytest generator import preflight failed "
            f"with return code {result.returncode}; see {_DIAGNOSTIC_FILE_NAME}"
        )


def _log_exception(event, exc):
    _diagnostic_log(
        event,
        force=True,
        exception_type=type(exc).__name__,
        exception=str(exc),
    )
    _diagnostic_write(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip(),
        force=True,
    )


class CaseDataStore:
    """Keep the pytest-generated result paired with its TTK testcase."""

    def __init__(self):
        self._cases = {}

    def put(self, testcase_name, data):
        if testcase_name is None:
            raise RuntimeError("SparseFlashAttention requires testcase_name for pytest-backed golden")
        self._cases[str(testcase_name)] = data

    def get(self, testcase_name):
        data = self._cases.get(str(testcase_name)) if testcase_name is not None else None
        if data is None:
            raise RuntimeError(
                "SparseFlashAttention golden requires customize_inputs to run before golden generation"
            )
        return data


CASE_DATA = CaseDataStore()


def load_pytest_golden_module():
    """Load the pytest implementation without polluting another operator's imports."""
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        _diagnostic_log(
            "pytest_module_cache_hit",
            module_file=_module_file(PYTEST_GOLDEN_MODULE),
        )
        return PYTEST_GOLDEN_MODULE

    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "sparse_flash_attention_golden.py"
    _enable_fault_handler()
    _log_runtime_snapshot(pytest_dir, module_path)
    saved_generator = sys.modules.pop("generate_tensor_data", None)
    sys.path.insert(0, str(pytest_dir))
    _diagnostic_log(
        "pytest_load_start",
        loader="spec_from_file_location",
        saved_generator_file=_module_file(saved_generator),
        pytest_dir=str(pytest_dir),
        module_path=str(module_path),
    )
    try:
        _run_generator_preflight(pytest_dir)
        spec = importlib.util.spec_from_file_location("sfa_ttk_pytest_golden", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create a loader for {module_path}")
        _diagnostic_log(
            "pytest_load_spec_ready",
            spec_name=spec.name,
            loader_type=type(spec.loader).__name__,
            origin=spec.origin,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        _diagnostic_log(
            "pytest_exec_start",
            spec_name=spec.name,
            module_file=_module_file(module),
        )
        spec.loader.exec_module(module)
        _diagnostic_log(
            "pytest_exec_complete",
            spec_name=spec.name,
            module_file=_module_file(module),
        )
    except Exception as exc:
        _log_exception("pytest_load_failure", exc)
        sys.modules.pop("sfa_ttk_pytest_golden", None)
        raise
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            _diagnostic_log(
                "pytest_path_cleanup_missing",
                force=True,
                pytest_dir=str(pytest_dir),
            )
        sys.modules.pop("generate_tensor_data", None)
        if saved_generator is not None:
            sys.modules["generate_tensor_data"] = saved_generator
        _diagnostic_log(
            "pytest_load_cleanup_complete",
            generator_restored=saved_generator is not None,
        )
    PYTEST_GOLDEN_MODULE = module
    _diagnostic_log(
        "pytest_module_ready",
        module_file=_module_file(PYTEST_GOLDEN_MODULE),
    )
    return module


def to_int_list(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(to_int_list(item))
        return result
    return [int(value)]


def dtype_name(tensor):
    if tensor is None or not torch.is_tensor(tensor):
        return "fp16"
    mapping = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
        torch.int32: "int32",
        torch.int8: "int8",
    }
    try:
        return mapping[tensor.dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported SparseFlashAttention pytest dtype: {tensor.dtype}") from exc


def _range_at(input_ranges, index, default):
    if input_ranges and index < len(input_ranges) and input_ranges[index] is not None:
        value = input_ranges[index]
        if isinstance(value, (list, tuple)) and len(value) in (1, 2):
            return list(value)
    return list(default)


def _prefix_lengths(values, total, field):
    if not values:
        return [int(total)]
    values = [int(item) for item in values]
    if values[-1] == int(total) and all(values[idx] >= values[idx - 1] for idx in range(1, len(values))):
        return values
    if sum(values) == int(total) and all(item >= 0 for item in values):
        total_value = 0
        prefix = []
        for item in values:
            total_value += item
            prefix.append(total_value)
        return prefix
    raise ValueError(f"{field} must be cumulative or sum to T={total}, got {values}")


def _sequence_values(value, values_override, layout, total, batch, field):
    values = to_int_list(values_override) if values_override is not None else to_int_list(value)
    if layout == "TND":
        return _prefix_lengths(values, total, field)
    if not values:
        return [int(total)] * int(batch)
    if len(values) == 1 and int(batch) > 1:
        return values * int(batch)
    if len(values) != int(batch):
        raise ValueError(f"{field} length must be 1 or batch={batch}, got {values}")
    return values


def build_pytest_params(query, key, value, sparse_indices, scale_value, *, block_table=None,
                        actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
                        query_rope=None, key_rope=None, sparse_block_size=1,
                        layout_query="BSND", layout_kv="BSND", sparse_mode=3,
                        attention_mode=0, return_softmax_lse=False, input_ranges=None,
                        testcase_name=None, **kwargs):
    """Translate the TTK API tensors to the parameter dictionary used by pytest."""
    if query is None or key is None or sparse_indices is None:
        raise ValueError("SparseFlashAttention requires query, key, and sparse_indices")

    rope_dim = int(kwargs.get("rope_head_dim", query_rope.shape[-1] if query_rope is not None else 64))
    if layout_query == "TND":
        batch = len(to_int_list(kwargs.get("actual_seq_lengths_query_values")))
        batch = batch or len(to_int_list(actual_seq_lengths_query)) or 1
        q_total = int(query.shape[0])
        q_heads = int(query.shape[1])
        q_seq = q_total
    else:
        batch = int(query.shape[0])
        q_total = batch * int(query.shape[1])
        q_heads = int(query.shape[2])
        q_seq = int(query.shape[1])

    actual_q = _sequence_values(
        actual_seq_lengths_query,
        kwargs.get("actual_seq_lengths_query_values"),
        layout_query,
        int(query.shape[0]) if layout_query == "TND" else q_seq,
        batch,
        "actual_seq_lengths_query",
    )

    if layout_kv == "PA_BSND":
        block_size = int(kwargs.get("block_size", key.shape[1]))
        block_num = int(kwargs.get("block_num", key.shape[0]))
        kv_heads = int(key.shape[2])
        kv_total = max(to_int_list(kwargs.get("actual_seq_lengths_kv_values")) or
                       to_int_list(actual_seq_lengths_kv) or [block_size])
        key_shape = [batch, kv_total, kv_heads, int(key.shape[3])]
    elif layout_kv == "TND":
        block_size = int(kwargs.get("block_size", 1))
        block_num = int(kwargs.get("block_num", 0))
        kv_heads = int(key.shape[1])
        kv_total = int(key.shape[0])
        key_shape = list(key.shape)
    else:
        block_size = int(kwargs.get("block_size", 1))
        block_num = int(kwargs.get("block_num", 0))
        kv_heads = int(key.shape[2])
        kv_total = int(key.shape[1])
        key_shape = list(key.shape)

    actual_kv = _sequence_values(
        actual_seq_lengths_kv,
        kwargs.get("actual_seq_lengths_kv_values"),
        layout_kv,
        kv_total,
        batch,
        "actual_seq_lengths_kv",
    )
    if layout_kv == "PA_BSND":
        key_shape[1] = max(actual_kv)

    if query_rope is None:
        query_rope_shape = list(query.shape)
        query_rope_shape[-1] = rope_dim
    else:
        query_rope_shape = list(query_rope.shape)
    if key_rope is None or layout_kv == "PA_BSND":
        key_rope_shape = list(key_shape)
        key_rope_shape[-1] = rope_dim
    else:
        key_rope_shape = list(key_rope.shape)

    group = q_heads // kv_heads
    if layout_query == "TND":
        softmax_shape = [kv_heads, int(query.shape[0]), group]
    else:
        softmax_shape = [batch, kv_heads, q_seq, group]

    params = {
        "case_name": testcase_name or "ttk_sparse_flash_attention",
        "layout_query": layout_query,
        "layout_kv": layout_kv,
        "actualseqlengths": actual_q,
        "actualseqlengthskv": actual_kv,
        "scalevalue": float(scale_value),
        "sparsemode": int(sparse_mode),
        "sparse_blocksize": int(sparse_block_size),
        "shape_input": {
            "query": list(query.shape),
            "key": key_shape,
            "value": key_shape,
            "sparse_indices": list(sparse_indices.shape),
            "block_table": list(block_table.shape) if block_table is not None else [batch],
            "query_rope": query_rope_shape,
            "key_rope": key_rope_shape,
            "sinks": [q_heads],
        },
        "dtype_input": {
            "query": dtype_name(query),
            "key": dtype_name(key),
            "value": dtype_name(value if value is not None else key),
            "sparse_indices": "int32",
            "block_table": "int32",
            "query_rope": dtype_name(query_rope if query_rope is not None else query),
            "key_rope": dtype_name(key_rope if key_rope is not None else key),
            "sinks": "fp32",
        },
        "range_input": {
            "query": _range_at(input_ranges, 0, (-10.0, 10.0)),
            "key": _range_at(input_ranges, 1, (-10.0, 10.0)),
            "sparse_indices": _range_at(input_ranges, 3, (0, max(int(sparse_indices.shape[-1]) - 1, 0))),
            "block_table": _range_at(input_ranges, 4, (0, max(block_num - 1, 0))),
            "query_rope": _range_at(input_ranges, 7, (-10.0, 10.0)),
            "key_rope": _range_at(input_ranges, 8, (-10.0, 10.0)),
            "sinks": (-0.2, 0.2),
        },
        "dtype_output": [dtype_name(query)],
        "shape_output": {
            "attn_out": list(query.shape),
            "softmax_max": softmax_shape,
            "softmax_sum": softmax_shape,
        },
        "rope_head_dim": rope_dim,
        "attention_mode": int(attention_mode),
        "return_softmax_lse": bool(return_softmax_lse),
        "use_sinks": False,
        "block_size": block_size,
        "block_num": block_num,
    }
    return params


def normalize_pytest_outputs(outputs, query, return_softmax_lse):
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
        raise RuntimeError("SparseFlashAttention pytest compute_cpu returned an invalid output tuple")
    attn_out = outputs[0].detach().cpu().to(dtype=query.dtype)
    if not bool(return_softmax_lse):
        empty = torch.zeros(0, dtype=torch.float32)
        return attn_out, empty, empty
    return tuple(output.detach().cpu() for output in outputs)


def cpu_sparse_flash_attention(query, key, value, sparse_indices, scale_value, *,
                               testcase_name=None, **kwargs):
    """Return the result generated by pytest for this exact TTK testcase."""
    del query, key, value, sparse_indices, scale_value, kwargs
    return CASE_DATA.get(testcase_name)["golden"]


__golden__ = {
    "e2e": {
        "torch_npu.npu_sparse_flash_attention": "cpu_sparse_flash_attention",
    }
}
