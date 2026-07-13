#!/usr/bin/python
# -*- coding: utf-8 -*-
"""CPU golden adapter for SparseFlashMla TTK cases."""

import importlib.util
import sys
from pathlib import Path

import torch


_CASE_DATA = {}
_PYTEST_GOLDEN = None
_PYTEST_UTILS = None


def _pytest_dir():
    return Path(__file__).resolve().parents[2] / "pytest"


def _load_pytest_module(stem, filename):
    name = f"smla_pytest_{stem}"
    if name in sys.modules:
        return sys.modules[name]

    pytest_dir = _pytest_dir()
    module_path = pytest_dir / filename
    inserted = str(pytest_dir) not in sys.path
    if inserted:
        sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        return module
    finally:
        if inserted:
            sys.path.remove(str(pytest_dir))


def _load_pytest_golden():
    global _PYTEST_GOLDEN
    if _PYTEST_GOLDEN is None:
        _PYTEST_GOLDEN = _load_pytest_module("golden", "sparse_flash_mla_golden.py")
    return _PYTEST_GOLDEN


def _load_pytest_utils():
    global _PYTEST_UTILS
    if _PYTEST_UTILS is None:
        _PYTEST_UTILS = _load_pytest_module("utils", "utils.py")
    return _PYTEST_UTILS


def _torch_dtype_from_name(value, default=None):
    if isinstance(value, torch.dtype):
        return value
    text = str(value).lower()
    if "bfloat16" in text or "bf16" in text:
        return torch.bfloat16
    if "float16" in text or "fp16" in text:
        return torch.float16
    if "float32" in text or "fp32" in text or text == "float":
        return torch.float32
    if "float8_e4m3fn" in text and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    return default if default is not None else torch.float16


def _list_or_none(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, tuple):
        return list(value)
    return value


def _build_params(kwargs):
    tensor_dtypes = kwargs.get("tensor_dtypes") or ()
    params = dict(kwargs)
    for key in (
        "cu_seqlens_q",
        "cu_seqlens_ori_kv",
        "cu_seqlens_cmp_kv",
        "seqused_q",
        "seqused_ori_kv",
        "seqused_cmp_kv",
        "cmp_residual_kv",
    ):
        params[key] = params.get(f"{key}_values", params.get(key))
    params.setdefault("testcase_name", kwargs.get("testcase_name"))
    params.setdefault(
        "q_type",
        _torch_dtype_from_name(tensor_dtypes[0] if len(tensor_dtypes) > 0 else "bfloat16"),
    )
    params.setdefault(
        "ori_kv_type",
        _torch_dtype_from_name(tensor_dtypes[1] if len(tensor_dtypes) > 1 else params["q_type"]),
    )
    params.setdefault(
        "cmp_kv_type",
        _torch_dtype_from_name(tensor_dtypes[2] if len(tensor_dtypes) > 2 else params["ori_kv_type"]),
    )
    for key in ("q_type", "ori_kv_type", "cmp_kv_type"):
        default = params["q_type"] if key != "q_type" else torch.bfloat16
        params[key] = _torch_dtype_from_name(params.get(key), default)
    for key in (
        "cu_seqlens_q",
        "cu_seqlens_ori_kv",
        "cu_seqlens_cmp_kv",
        "seqused_q",
        "seqused_ori_kv",
        "seqused_cmp_kv",
        "cmp_residual_kv",
    ):
        params[key] = _list_or_none(params.get(key))
    for key in ("q_datarange", "ori_kv_datarange", "cmp_kv_datarange"):
        if params.get(key) is None:
            params[key] = "[-10, 10]"
        elif isinstance(params[key], (list, tuple)):
            params[key] = repr(list(params[key]))
    params.setdefault("qk_equal_len", False)
    params.setdefault("return_softmax_lse", False)
    params.setdefault("ori_kv_topk_mode", "full")
    params.setdefault("cmp_kv_topk_mode", "full")
    params.setdefault("ori_sparse_indices_mode", "full")
    params.setdefault("cmp_sparse_indices_mode", "full")
    params.setdefault("actlen_mode", "full")
    return params


def prepare_case(kwargs):
    """Generate inputs and golden through the existing pytest implementation."""
    params = _build_params(kwargs)
    pytest_utils = _load_pytest_utils()
    pytest_golden = _load_pytest_golden()
    case_params = pytest_utils.generate_case_with_default_param(params)
    data = pytest_golden.gen_data(case_params, case_params.get("template_mode"))
    name = kwargs.get("testcase_name") or case_params.get("testcase_name")
    if name is not None:
        _CASE_DATA[str(name)] = data
    return data


def cpu_sparse_flash_mla(q, *, return_softmax_lse=False, testcase_name=None, **kwargs):
    del q, kwargs
    data = _CASE_DATA.get(str(testcase_name)) if testcase_name is not None else None
    if data is None:
        raise RuntimeError(
            "SparseFlashMla TTK golden requires customize_inputs to generate pytest data first"
        )
    lse = data.get("softmax_lse")
    if lse is None or not bool(return_softmax_lse):
        lse = torch.zeros(0, dtype=torch.float32)
    return data["cpu_output"], lse
