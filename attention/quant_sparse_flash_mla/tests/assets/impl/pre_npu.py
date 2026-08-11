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

"""QuantSparseFlashMla metadata stage between input restoration and the main API."""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

from ttk.test_spec import PreNpuResult


METADATA_INDEX = 20
GOLDEN_MODULE_NAME = "qsmla_ttk_golden"


def load_golden_module():
    """Load the operator-owned pytest state store without package imports."""
    if GOLDEN_MODULE_NAME in sys.modules:
        return sys.modules[GOLDEN_MODULE_NAME]
    path = Path(__file__).with_name("golden.py")
    try:
        spec = importlib.util.spec_from_file_location(GOLDEN_MODULE_NAME, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[GOLDEN_MODULE_NAME] = module
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        sys.modules.pop(GOLDEN_MODULE_NAME, None)
        raise RuntimeError(
            "Failed to load QuantSparseFlashMla pytest state store; "
            f"module={path.resolve()}; original error: {type(exc).__name__}: {exc}"
        ) from exc


def metadata_arguments(context):
    """Return the exact metadata arguments prepared by pytest input generation."""
    return load_golden_module().get_metadata_input(
        context.testcase_name, context
    )


def parse_metadata_only_option(kwargs):
    """Keep absent, explicit false, and explicit true as three separate states."""
    if "metadata_only" not in kwargs:
        return None
    value = kwargs["metadata_only"]
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError("metadata_only must be a boolean")


def is_materialized(metadata):
    if torch.is_tensor(metadata):
        return bool(torch.count_nonzero(metadata).item())
    return bool(np.count_nonzero(metadata))


def to_npu(value, empty_if_none=False):
    if value is None:
        return torch.empty(0, dtype=torch.int32, device="npu") if empty_if_none else None
    if torch.is_tensor(value):
        return value.npu()
    return torch.as_tensor(value).npu()


def copy_metadata(target, source):
    source = source.detach().cpu()
    if tuple(target.shape) != tuple(source.shape):
        raise ValueError(
            f"QSMLA metadata shape mismatch: placeholder={tuple(target.shape)}, "
            f"generated={tuple(source.shape)}"
        )
    if torch.is_tensor(target):
        target.copy_(source.to(dtype=target.dtype, device=target.device))
    else:
        np.copyto(target, source.numpy().astype(target.dtype, copy=False))


def persist_metadata(context, metadata):
    """Publish generated metadata into replay input storage when it exists."""
    case_dir = context.manual_case_dir
    file_format = context.manual_data_format
    if case_dir is None or file_format is None:
        return
    matches = tuple(case_dir.glob(f"input_{METADATA_INDEX}_*.{file_format}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"QSMLA expected one stored metadata input, found {len(matches)} in {case_dir}"
        )
    path = matches[0]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    array = (
        metadata.detach().cpu().contiguous().numpy()
        if torch.is_tensor(metadata) else np.asarray(metadata)
    )
    if file_format == "bin":
        array.tofile(temporary)
    elif file_format == "npy":
        with temporary.open("wb") as file:
            np.save(file, array, allow_pickle=False)
    elif file_format == "pt":
        torch.save(torch.from_numpy(np.array(array, copy=True)), temporary)
    else:
        raise ValueError(f"QSMLA cannot persist manual-data format {file_format!r}")
    temporary.replace(path)


def run_torch(target, arguments):
    generated = torch.ops.cann_ops_transformer.quant_sparse_flash_mla_metadata(
        arguments["num_heads_q"],
        arguments["num_heads_kv"],
        arguments["head_dim"],
        arguments["quant_mode"],
        cu_seqlens_q=to_npu(arguments["cu_seqlens_q"], empty_if_none=True),
        cu_seqlens_ori_kv=to_npu(
            arguments["cu_seqlens_ori_kv"], empty_if_none=True
        ),
        cu_seqlens_cmp_kv=to_npu(
            arguments["cu_seqlens_cmp_kv"], empty_if_none=True
        ),
        seqused_q=to_npu(arguments["seqused_q"], empty_if_none=True),
        seqused_ori_kv=to_npu(arguments["seqused_ori_kv"]),
        seqused_cmp_kv=to_npu(arguments["seqused_cmp_kv"]),
        cmp_residual_kv=to_npu(arguments["cmp_residual_kv"]),
        ori_topk_length=to_npu(arguments["ori_topk_length"]),
        cmp_topk_length=to_npu(arguments["cmp_topk_length"]),
        batch_size=arguments["batch_size"],
        max_seqlen_q=arguments["max_seqlen_q"],
        max_seqlen_ori_kv=arguments["max_seqlen_ori_kv"],
        max_seqlen_cmp_kv=arguments["max_seqlen_cmp_kv"],
        ori_topk=arguments["ori_topk"],
        cmp_topk=arguments["topk"],
        cmp_ratio=arguments["cmp_ratio"],
        ori_mask_mode=arguments["ori_mask_mode"],
        cmp_mask_mode=arguments["cmp_mask_mode"],
        ori_win_left=arguments["ori_win_left"],
        ori_win_right=arguments["ori_win_right"],
        layout_q=arguments["layout_q"],
        layout_kv=arguments["layout_kv"],
        has_ori_kv=arguments["has_ori_kv"],
        has_cmp_kv=arguments["has_cmp_kv"],
    )
    copy_metadata(target, generated)


def run_aclnn(context, metadata, arguments):
    context.run_aclnn(
        "aclnnQuantSparseFlashMlaMetadata",
        tensors={
            "cuSeqlensQOptional": arguments["cu_seqlens_q"],
            "cuSeqlensOriKvOptional": arguments["cu_seqlens_ori_kv"],
            "cuSeqlensCmpKvOptional": arguments["cu_seqlens_cmp_kv"],
            "sequsedQOptional": arguments["seqused_q"],
            "sequsedOriKvOptional": arguments["seqused_ori_kv"],
            "sequsedCmpKvOptional": arguments["seqused_cmp_kv"],
            "cmpResidualKvOptional": arguments["cmp_residual_kv"],
            "oriTopkLengthOptional": arguments["ori_topk_length"],
            "cmpTopkLengthOptional": arguments["cmp_topk_length"],
            "metaData": metadata,
        },
        attributes={
            "numHeadsQ": arguments["num_heads_q"],
            "numHeadsKv": arguments["num_heads_kv"],
            "headDim": arguments["head_dim"],
            "quantMode": arguments["quant_mode"],
            "batchSize": arguments["batch_size"],
            "maxSeqlenQ": arguments["max_seqlen_q"],
            "maxSeqlenOriKv": arguments["max_seqlen_ori_kv"],
            "maxSeqlenCmpKv": arguments["max_seqlen_cmp_kv"],
            "oriTopk": arguments["ori_topk"],
            "cmpTopk": arguments["topk"],
            "cmpRatio": arguments["cmp_ratio"],
            "oriMaskMode": arguments["ori_mask_mode"],
            "cmpMaskMode": arguments["cmp_mask_mode"],
            "oriWinLeft": arguments["ori_win_left"],
            "oriWinRight": arguments["ori_win_right"],
            "layoutQOptional": arguments["layout_q"],
            "layoutKvOptional": arguments["layout_kv"],
            "hasOriKv": arguments["has_ori_kv"],
            "hasCmpKv": arguments["has_cmp_kv"],
        },
        output_names=("metaData",),
    )


def run(*args, context=None, **kwargs):
    """Generate/reuse metadata according to the three-state stop attribute."""
    del args
    if context is None:
        raise RuntimeError("QSMLA metadata stage requires an explicitly injected context")
    tensors = tuple(context.input_tensors or ())
    if len(tensors) <= METADATA_INDEX or tensors[METADATA_INDEX] is None:
        raise ValueError("QSMLA CSV must reserve tensor slot 20 for metadata")

    stop_after_stage = parse_metadata_only_option(kwargs)
    metadata = tensors[METADATA_INDEX]
    if stop_after_stage is None and is_materialized(metadata):
        return None

    arguments = metadata_arguments(context)
    if context.case_type == "aclnn":
        run_aclnn(context, metadata, arguments)
    elif context.case_type == "e2e":
        run_torch(metadata, arguments)
    else:
        raise ValueError(f"QSMLA pre-NPU does not support case type {context.case_type!r}")
    # Explicit false refreshes only this invocation; prepare/metadata-only
    # flows are the paths that publish metadata back to replay storage.
    if stop_after_stage is not False:
        persist_metadata(context, metadata)
    if stop_after_stage is True:
        return PreNpuResult(stop=True, reason="QSMLA metadata generated")
    return None
