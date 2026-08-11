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

"""CPU Golden adapter for QuantLightningIndexer V2 TTK cases."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


OPERATOR = "quant_lightning_indexer_v2"
SIDECAR_NAME = "metadata_inputs.pt"
SIDECAR_VERSION = 1
DATA_STATE_KEY = f"{OPERATOR}.pytest_data"
METADATA_STATE_KEY = f"{OPERATOR}.metadata_inputs"
PYTEST_MODULE_NAME = "qli_v2_pytest_golden"
PYTEST_MODULE_FILE = "quant_lightning_indexer_v2_golden.py"

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
        "unsupported QuantLightningIndexer V2 metadata input type: "
        f"{type(value).__name__}"
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
        self.active_testcase_name = None

    def clear(self):
        self.case_data.clear()
        self.active_testcase_name = None

    def put(self, testcase_name, data):
        if testcase_name is not None:
            self.case_data[str(testcase_name)] = data

    def persist(self, testcase_name, context):
        if context is None or testcase_name is None:
            return
        name = str(testcase_name)
        data = self.case_data.pop(name, None)
        if data is None:
            raise RuntimeError("QuantLightningIndexer V2 pytest case data is unavailable")
        metadata_input = data.get("metadata_input")
        if not isinstance(metadata_input, dict):
            raise ValueError("QuantLightningIndexer V2 pytest data lacks metadata_input")

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
                "QuantLightningIndexer V2 context data belongs to a different testcase"
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
                "QuantLightningIndexer V2 metadata sidecar has an incompatible schema"
            )
        if payload.get("csv_contract") != csv_contract_digest(context):
            raise ValueError(
                "QuantLightningIndexer V2 metadata sidecar does not match the current CSV contract"
            )
        metadata_input = payload.get("metadata_input")
        if not isinstance(metadata_input, dict):
            raise ValueError(
                "QuantLightningIndexer V2 metadata sidecar lacks metadata_input"
            )
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
            "Failed to load QuantLightningIndexer V2 pytest Golden module; "
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
            "QuantLightningIndexer V2 requires input-stage metadata inputs before pre-NPU"
        )
    return metadata_input


def materialize_golden(data):
    if data.get("cpu_result") is None:
        load_pytest_golden().generate_cpu_golden(data)
    return data


def activate_case_data(testcase_name, context):
    data = CASE_DATA.get(testcase_name, context)
    if data is None:
        raise RuntimeError(
            "QuantLightningIndexer V2 Golden requires pytest data from the input-stage context"
        )
    CASE_DATA.active_testcase_name = str(testcase_name)
    return materialize_golden(data)


def get_compare_data(testcase_name, context=None):
    if testcase_name is None:
        testcase_name = CASE_DATA.active_testcase_name
    if testcase_name is None:
        return None
    data = CASE_DATA.get(testcase_name, context)
    return None if data is None else materialize_golden(data)


def set_compare_data(testcase_name, data, context=None):
    name = str(testcase_name)
    CASE_DATA.active_testcase_name = name
    if context is None:
        CASE_DATA.case_data[name] = data
        return
    context.state[DATA_STATE_KEY] = {"testcase_name": name, "data": data}


def cpu_quant_lightning_indexer_v2(
        query, key, weights, query_dequant_scale, key_dequant_scale,
        topk, quant_mode, *, return_value=0, testcase_name=None,
        context=None, **kwargs):
    """Materialize Golden from the exact pytest data produced by input."""
    del query, key, weights, query_dequant_scale, key_dequant_scale
    del topk, quant_mode, kwargs
    data = activate_case_data(testcase_name, context)
    if int(return_value):
        sparse_value = data["cpu_topk_value"]
    else:
        sparse_value = torch.zeros(0, dtype=data["topk_value"].dtype)
    return data["cpu_result"], sparse_value


def cpu_aclnn_qli_v2(
        query, key, weights, query_dequant_scale, key_dequant_scale,
        cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
        cmp_residual_k, block_table, output_idx_offset, metadata, topk,
        quant_mode, max_seqlen_q, layout_q, layout_k, mask_mode,
        cmp_ratio, return_value, sparse_indices_out, sparse_values_out,
        testcase_name=None, context=None, **kwargs):
    """Return the pytest Golden for the ACLNN C API parameter order."""
    del (
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        metadata,
        max_seqlen_q,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
        sparse_indices_out,
    )
    sparse_indices, sparse_values = cpu_quant_lightning_indexer_v2(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        topk,
        quant_mode,
        return_value=return_value,
        testcase_name=testcase_name,
        context=context,
        **kwargs,
    )
    if not int(return_value):
        if sparse_values_out is None:
            raise ValueError("ACLNN QLI_V2 requires the sparseValuesOut tensor slot")
        if torch.is_tensor(sparse_values_out):
            sparse_values = torch.zeros(
                tuple(sparse_values_out.shape), dtype=sparse_values_out.dtype
            )
        else:
            sparse_values = np.zeros_like(np.asarray(sparse_values_out))
    return sparse_indices, sparse_values
