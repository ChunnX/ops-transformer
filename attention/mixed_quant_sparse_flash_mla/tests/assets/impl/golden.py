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

"""CPU Golden adapter for MixedQuantSparseFlashMla TTK cases."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


OPERATOR = "mixed_quant_sparse_flash_mla"
SIDECAR_NAME = "metadata_inputs.pt"
SIDECAR_VERSION = 1
DATA_STATE_KEY = f"{OPERATOR}.pytest_data"
METADATA_STATE_KEY = f"{OPERATOR}.metadata_inputs"
PYTEST_MODULE_NAME = "mqsmla_pytest_golden"
PYTEST_MODULE_FILE = "mixed_quant_sparse_flash_mla_golden.py"

NON_INPUT_CSV_FIELDS = {
    "is_enabled",
    "precision_tolerances",
    "absolute_precision",
    "metadata_only",
}


def csv_contract_digest(context):
    """Bind persisted metadata inputs to the CSV contract that generated them."""
    fields = getattr(context, "csv_fields", None)
    if not fields:
        return None
    contract = {
        name: value
        for name, value in fields.items()
        if name not in NON_INPUT_CSV_FIELDS
    }
    attributes = getattr(context, "attributes", None)
    if attributes is not None:
        attributes = dict(attributes)
        attributes.pop("metadata_only", None)
        contract["attributes"] = attributes
    serialized = json.dumps(
        contract, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def clone_metadata_value(value):
    """Copy only the small pytest metadata arguments into a CPU-only payload."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {name: clone_metadata_value(item) for name, item in value.items()}
    if isinstance(value, list):
        return [clone_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_metadata_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(
        f"unsupported MixedQuantSparseFlashMla metadata input type: {type(value).__name__}"
    )


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class CaseDataStore:
    """Share pytest data in-process and persist only metadata API inputs."""

    def __init__(self):
        self.case_data = {}

    def put(self, testcase_name, data):
        if testcase_name is not None:
            self.case_data[str(testcase_name)] = data

    def persist(self, testcase_name, context):
        if context is None or testcase_name is None:
            return
        name = str(testcase_name)
        data = self.case_data.pop(name, None)
        if data is None:
            raise RuntimeError("MixedQuantSparseFlashMla pytest case data is unavailable")
        metadata_input = data.get("metadata_input")
        if not isinstance(metadata_input, dict):
            raise ValueError("MixedQuantSparseFlashMla pytest data lacks metadata_input")

        context.state[DATA_STATE_KEY] = {"testcase_name": name, "data": data}
        payload = {
            "version": SIDECAR_VERSION,
            "operator": OPERATOR,
            "testcase_name": name,
            "csv_contract": csv_contract_digest(context),
            "metadata_input": clone_metadata_value(metadata_input),
        }
        context.state[METADATA_STATE_KEY] = payload
        if context.sidecar_dir is not None:
            path = context.sidecar_file(SIDECAR_NAME)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                torch.save(payload, temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)

    def get(self, testcase_name, context=None):
        if testcase_name is None:
            return None
        name = str(testcase_name)
        if context is None:
            return self.case_data.get(name)
        entry = context.state.get(DATA_STATE_KEY)
        if entry is None:
            return None
        if entry.get("testcase_name") != name:
            raise ValueError(
                "MixedQuantSparseFlashMla context data belongs to a different testcase"
            )
        return entry.get("data")

    def load_metadata_input(self, testcase_name, context):
        if testcase_name is None or context is None:
            return None
        name = str(testcase_name)
        data = self.get(name, context)
        if data is not None:
            return data.get("metadata_input")

        payload = context.state.get(METADATA_STATE_KEY)
        if payload is None and context.sidecar_dir is not None:
            path = context.sidecar_file(SIDECAR_NAME)
            if path.is_file():
                payload = load_pt(path)
        if payload is None:
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("version") != SIDECAR_VERSION
            or payload.get("operator") != OPERATOR
            or payload.get("testcase_name") != name
        ):
            raise ValueError(
                "MixedQuantSparseFlashMla metadata sidecar has an incompatible schema"
            )
        if payload.get("csv_contract") != csv_contract_digest(context):
            raise ValueError(
                "MixedQuantSparseFlashMla metadata sidecar does not match the current CSV contract"
            )
        metadata_input = payload.get("metadata_input")
        if not isinstance(metadata_input, dict):
            raise ValueError("MixedQuantSparseFlashMla metadata sidecar lacks metadata_input")
        context.state[METADATA_STATE_KEY] = payload
        return metadata_input


CASE_DATA = CaseDataStore()


def load_pytest_golden():
    """Load the pytest CPU reference only when the Golden stage needs it."""
    if PYTEST_MODULE_NAME in sys.modules:
        return sys.modules[PYTEST_MODULE_NAME]
    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    path = pytest_dir / PYTEST_MODULE_FILE
    inserted = str(pytest_dir) not in sys.path
    if inserted:
        sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(PYTEST_MODULE_NAME, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[PYTEST_MODULE_NAME] = module
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        sys.modules.pop(PYTEST_MODULE_NAME, None)
        raise RuntimeError(
            "Failed to load MixedQuantSparseFlashMla pytest Golden module; "
            f"module={path.resolve()}; original error: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if inserted:
            sys.path.remove(str(pytest_dir))


def get_case_data(testcase_name, context=None):
    return CASE_DATA.get(testcase_name, context)


def get_metadata_input(testcase_name, context):
    metadata_input = CASE_DATA.load_metadata_input(testcase_name, context)
    if metadata_input is None:
        raise RuntimeError(
            "MixedQuantSparseFlashMla requires input-stage metadata inputs before pre-NPU"
        )
    return metadata_input


def materialize_golden(data):
    if data.get("cpu_output") is None:
        load_pytest_golden().generate_cpu_golden(data)
    return data


def activate_case_data(testcase_name, context):
    data = CASE_DATA.get(testcase_name, context)
    if data is None:
        raise RuntimeError(
            "MixedQuantSparseFlashMla Golden requires pytest data from the input-stage context"
        )
    return materialize_golden(data)


def cpu_mixed_quant_sparse_flash_mla(
        q, *, return_softmax_lse=False, testcase_name=None,
        context=None, **kwargs):
    del q, kwargs
    data = activate_case_data(testcase_name, context)
    softmax_lse = data.get("cpu_lse") if bool(return_softmax_lse) else None
    return data["cpu_output"], softmax_lse


def cpu_aclnn_mixed_quant_sparse_flash_mla(
        q, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices,
        ori_block_table, cmp_block_table, cu_seqlens_q,
        cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q,
        seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv,
        ori_topk_length, cmp_topk_length, sinks, metadata, quant_mode,
        rope_head_dim, softmax_scale, cmp_ratio, ori_mask_mode,
        cmp_mask_mode, ori_win_left, ori_win_right, layout_q, layout_kv,
        topk_value_mode, return_softmax_lse, attn_out, softmax_lse_out,
        testcase_name=None, context=None, **kwargs):
    """Return the pytest Golden for the ACLNN C API parameter order."""
    del (
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        ori_topk_length,
        cmp_topk_length,
        sinks,
        metadata,
        quant_mode,
        rope_head_dim,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_q,
        layout_kv,
        topk_value_mode,
        attn_out,
        softmax_lse_out,
    )
    return cpu_mixed_quant_sparse_flash_mla(
        q,
        return_softmax_lse=return_softmax_lse,
        testcase_name=testcase_name,
        context=context,
        **kwargs,
    )
