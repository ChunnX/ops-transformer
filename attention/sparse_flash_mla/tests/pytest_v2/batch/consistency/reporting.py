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

"""Build bounded, reproducible input summaries for white-box reports."""

import hashlib
import json
import math
import os
from typing import Any, Dict, Iterable, Optional

import torch

from .adapter import CaseAdapter
from .schema import get_q_lengths


PROVENANCE_KEY = "_consistency_provenance"
CONTROL_TENSORS = {
    "cu_seqlens_q",
    "cu_seqlens_ori_kv",
    "cu_seqlens_cmp_kv",
    "seqused_q",
    "seqused_ori_kv",
    "seqused_cmp_kv",
    "cmp_residual_kv",
    "ori_topk_length",
    "cmp_topk_length",
}


def attach_provenance(
    input_data: Dict[str, Any],
    source_pt: Optional[str] = None,
    source_excel: Optional[str] = None,
    source_sheet: Optional[str] = None,
    stage: str = "loaded_pt",
) -> Dict[str, Any]:
    """Attach report-only source information without changing operator inputs."""
    provenance = dict(input_data.get(PROVENANCE_KEY, {}))
    if source_pt:
        provenance["source_pt"] = os.path.abspath(source_pt)
    if source_excel:
        provenance["source_excel"] = os.path.abspath(source_excel)
    if source_sheet:
        provenance["source_sheet"] = source_sheet
    provenance["stage"] = stage
    input_data[PROVENANCE_KEY] = provenance
    return input_data


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().to("cpu").contiguous().reshape(-1)
    return contiguous.view(torch.uint8).numpy().tobytes()


def _sample_values(tensor: torch.Tensor, limit: int) -> list:
    flat = tensor.detach().to("cpu").reshape(-1)
    if flat.numel() > limit:
        head = max(1, limit // 2)
        tail = max(1, limit - head)
        sampled = torch.cat((flat[:head], flat[-tail:]))
    else:
        sampled = flat
    if sampled.dtype.is_floating_point:
        sampled = sampled.to(torch.float32)
    return [
        value if not isinstance(value, float) or math.isfinite(value) else str(value)
        for value in sampled.tolist()
    ]


def summarize_tensor(name: str, tensor: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
    """Summarize a tensor without serializing large Q/KV payloads into JSON."""
    if tensor is None:
        return None
    raw = _tensor_bytes(tensor)
    element_size = int(tensor.element_size())
    result: Dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "element_size": element_size,
        "byte_size": int(tensor.numel()) * element_size,
        "md5": hashlib.md5(raw).hexdigest(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if name in CONTROL_TENSORS and tensor.numel() <= 256:
        result["values"] = _sample_values(tensor, 256)
        result["values_complete"] = True
    else:
        result["sample"] = _sample_values(tensor, 12)
        result["sample_policy"] = "first_6_and_last_6" if tensor.numel() > 12 else "all"
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() <= 256:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "values": _sample_values(value, 256),
            }
        return summarize_tensor("metadata_tensor", value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def stable_json_digest(value: Any) -> str:
    """Return a deterministic SHA256 for report-compatible nested values."""
    encoded = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _first_value(values: Iterable[Any], default: int = 0) -> int:
    for value in values:
        return int(value)
    return default


def build_input_summary(
    input_data: Dict[str, Any],
    role: str,
    mode: Optional[str] = None,
    transform: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe the exact PT/derived input consumed by metadata and the main op."""
    adapter = CaseAdapter(input_data)
    tensors = adapter.tensors
    metadata = adapter.metadata
    params = adapter.params
    n1 = int(adapter.get_metadata("N1", 0))
    n2 = int(adapter.get_metadata("N2", 0))
    tensor_summaries = {
        name: summarize_tensor(name, value)
        for name, value in sorted(tensors.items())
        if isinstance(value, torch.Tensor) or value is None
    }
    scalar_inputs = {
        name: _json_value(value)
        for name, value in sorted(tensors.items())
        if not isinstance(value, torch.Tensor) and value is not None
    }
    provenance = dict(input_data.get(PROVENANCE_KEY, {}))
    case_name = input_data.get("case_name") or input_data.get("Testcase_Name")
    if case_name is None:
        case_name = params.get("testcase_name") or params.get("Testcase_Name")
    summary = {
        "schema_version": 1,
        "operator": adapter.name,
        "role": role,
        "mode": mode,
        "case_name": case_name,
        "provenance": provenance,
        "dimensions": {
            "B": adapter.get_batch_size(),
            "S1_physical": get_q_lengths(input_data),
            "S1_effective": get_q_lengths(input_data, valid_only=True),
            "S2_max": int(adapter.get_metadata("max_seqlen_ori_kv", 0)),
            "N1": n1,
            "N2": n2,
            "G": n1 // n2 if n2 and n1 % n2 == 0 else None,
            "D": int(adapter.get_metadata("D", 0)),
            "layout_q": adapter.get_layout_q(),
            "layout_kv": adapter.get_layout_kv(),
        },
        "semantics": {
            "template": input_data.get("template_run_mode")
            or params.get("template_mode")
            or params.get("template_run_mode"),
            "ori_mask_mode": int(tensors.get("ori_mask_mode", metadata.get("ori_mask_mode", 0))),
            "cmp_mask_mode": int(tensors.get("cmp_mask_mode", metadata.get("cmp_mask_mode", 0))),
            "ori_win_left": int(tensors.get("ori_win_left", metadata.get("ori_win_left", -1))),
            "ori_win_right": int(tensors.get("ori_win_right", metadata.get("ori_win_right", 0))),
            "cmp_ratio": int(tensors.get("cmp_ratio") or metadata.get("cmp_ratio") or 1),
            "topk": int(metadata.get("K") or metadata.get("topk") or 0),
            "first_effective_q": _first_value(get_q_lengths(input_data, valid_only=True)),
        },
        "tensor_inputs": tensor_summaries,
        "scalar_inputs": scalar_inputs,
        "metadata_input": _json_value(metadata),
        "metadata_input_sha256": stable_json_digest(metadata),
        "params": _json_value(params),
        "params_sha256": stable_json_digest(params),
        "cpu_output": summarize_tensor("cpu_output", input_data.get("cpu_output")),
        "softmax_lse": summarize_tensor("softmax_lse", input_data.get("softmax_lse")),
        "transform": _json_value(transform) if transform else None,
    }
    return summary
