#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""In-process MoeDistributeDispatch -> Combine chain for TTK."""

import logging
import importlib.util
import os
import subprocess
import sys
import time
import tempfile
import base64
import json
from pathlib import Path

import numpy as np
import torch


def _make_dispatch_context(combine_context, plugin_path, rank_idx):
    from ttk.core_modules.aclnn.op_api_info_keeper import OpApiInfoKeeper
    from ttk.core_modules.testcase_manager import TestcaseAclnn
    from ttk.core_modules.npu.op_api.mc2_dispatcher import patch_moe_rank_attributes

    original_case_name = combine_context.testcase_name.rsplit(
        "__MoeDistributeCombine", 1
    )[0]
    original_attributes = getattr(combine_context, "original_dict", None) or {}
    configured_dispatch_api = original_attributes.get("_dispatchApiName")
    if configured_dispatch_api:
        api_name = str(configured_dispatch_api)
    else:
        api_name = "aclnnMoeDistributeDispatchV3"
    op_api_info = OpApiInfoKeeper().info_of(api_name)
    if op_api_info is None:
        raise RuntimeError(f"{api_name} is unavailable in the current CANN package")

    dispatch_context = TestcaseAclnn()
    dispatch_context.api_name = api_name
    dispatch_context.device_ids = combine_context.device_ids
    dispatch_context.my_rank = combine_context.my_rank
    dispatch_context.remark = combine_context.remark
    dispatch_context.testcase_name = (
        f"{combine_context.testcase_name.rsplit('__MoeDistributeCombine', 1)[0]}"
        f"__MoeDistributeDispatch{api_name[-2:]}"
    )
    dispatch_context.attributes = {
        key: value
        for key, value in combine_context.attributes.items()
        if key in op_api_info.params
    }
    try:
        import ast

        original_attributes = ast.literal_eval(original_attributes.get("attributes", "{}"))
    except (ValueError, SyntaxError):
        original_attributes = {}
    for attr_name in ("isDispatchScales", "_dispatchCommAlg", "isCombineX"):
        if attr_name in combine_context.attributes:
            dispatch_context.attributes[attr_name] = combine_context.attributes[
                attr_name
            ]
        elif attr_name in original_attributes:
            dispatch_context.attributes[attr_name] = original_attributes[attr_name]
    dispatch_context.attributes["isExpand"] = int(
        combine_context.attributes.get("isExpand", 0)
    )
    if "isXRandom" in combine_context.attributes:
        dispatch_context.attributes["isXRandom"] = combine_context.attributes[
            "isXRandom"
        ]
    if int(dispatch_context.attributes.get("quantMode", 0)) == 2:
        dispatch_context.attributes["isDispatchScales"] = int(
            combine_context.attributes.get("isDispatchScales", 0)
        )
    logging.debug(
        "Moe cascade dispatch attributes original=%s current=%s",
        original_attributes,
        dispatch_context.attributes,
    )
    dispatch_context.attributes["_isElastic"] = int(combine_context.attributes.get("isElastic", 0))
    if "localMoeExpertNum" in combine_context.attributes:
        dispatch_context.attributes["localMoeExpertNum"] = int(
            combine_context.attributes["localMoeExpertNum"]
        )
    if "sharedExpertNum" in dispatch_context.attributes:
        dispatch_context.attributes["sharedExpertRankNum"] = int(
            combine_context.attributes.get(
                "sharedExpertRankNum",
                dispatch_context.attributes["sharedExpertRankNum"],
            )
        )
        dispatch_context.attributes["rankNumPerSharedExpert"] = int(
            combine_context.attributes.get("rankNumPerSharedExpert", 0)
        )
    if "expertTokenNumsType" in dispatch_context.attributes:
        dispatch_context.attributes["expertTokenNumsType"] = 1
    dispatch_context.output_tensor_indexes = tuple(
        index for index, name in enumerate(op_api_info.tensors)
        if name.endswith("Out") or name == "expandXOut"
    )
    dispatch_context.output_inplace_indexes = ()
    combine_tensor_names = OpApiInfoKeeper().info_of(combine_context.api_name).tensors
    combine_mask_index = combine_tensor_names.index("xActiveMaskOptional")
    combine_mask_shape = combine_context.tensor_view_shapes[combine_mask_index]
    dispatch_context.attributes["isActivateMask"] = int(
        combine_context.attributes.get("isActivateMask", 0)
    )
    dispatch_context.attributes["xActiveMaskDim"] = int(
        combine_context.attributes.get("xActiveMaskDim", 0)
    )
    dispatch_context.attributes["_moeVariableBs"] = int(
        combine_context.attributes.get("_moeVariableBs", 0)
    )
    if "commAlg" in op_api_info.params:
        dispatch_context.attributes["commAlg"] = str(
            combine_context.attributes.get("_dispatchCommAlg") or "fullmesh_v1"
        )
    if "_moeDeepep" not in dispatch_context.attributes:
        remarks = {
            item.split("=", 1)[0].strip(): item.split("=", 1)[1].strip()
            for item in str(combine_context.remark or "").split(",")
            if "=" in item
        }
        dispatch_context.attributes["_moeDeepep"] = int(remarks.get("isDeepep", 0))
    if "epRankId" in dispatch_context.attributes:
        dispatch_context.attributes["epRankId"] = 0
    if int(combine_context.attributes.get("isElastic", 0)):
        available_ranks = _case_available_ranks(combine_context)
        combine_context.attributes["_moeElasticAvailableRanks"] = available_ranks
        dispatch_context.attributes["_moeElasticAvailableRanks"] = available_ranks
        if rank_idx not in set(available_ranks):
            dispatch_context.attributes["_moeElasticSuppressed"] = 1
    if "isElastic" in combine_context.attributes:
        dispatch_context.attributes["isElastic"] = combine_context.attributes["isElastic"]
    if "isDispatchScales" in combine_context.attributes:
        dispatch_context.attributes["isDispatchScales"] = combine_context.attributes[
            "isDispatchScales"
        ]
    for attr_name in ("shareBrokenCardNum", "moeBrokenCardNum"):
        if attr_name in combine_context.attributes:
            dispatch_context.attributes[attr_name] = combine_context.attributes[attr_name]
            generated_case = _moe_case_data_for_context(combine_context, rank_idx)
    generated_case = _moe_case_data_for_context(combine_context, rank_idx)
    combine_context.attributes["_moeCaseData"] = generated_case
    is_deepep = int(combine_context.attributes.get("_moeDeepep", 0) or 0) == 1
    global_bs = int(combine_context.attributes.get("globalBs", generated_case["x"].shape[0]))
    ep_world_size = int(combine_context.attributes.get("epWorldSize", len(combine_context.device_ids)))
    per_rank_bs = max(global_bs // ep_world_size, 1)
    source_x = generated_case["x"]
    source_expert_ids = generated_case["expert_ids"]
    golden_expand_x = generated_case["expand_x"]
    golden_expand_idx = generated_case["expand_idx"]
    golden_ep_recv_counts = generated_case["ep_recv_counts"]
    if int(combine_context.attributes.get("isExpand", 0)) == 1:
        source_expert_ids = generated_case["expert_ids_expand"]

    shapes = []
    dtypes = []
    input_ranges = []
    for name in op_api_info.tensors:
        if name == "x":
            shapes.append(tuple(source_x.shape))
            dtypes.append(str(source_x.dtype).replace("torch.", ""))
            input_ranges.append((-1, 1))
        elif name == "expertIds":
            source_expert_ids_rows = int(
                combine_context.attributes.get("_moeVariableBs", 0)
            ) and source_x.shape[0]
            if source_expert_ids_rows:
                source_expert_ids = source_expert_ids[:source_expert_ids_rows]
            shapes.append(tuple(source_expert_ids.shape))
            dtypes.append(str(source_expert_ids.dtype).replace("torch.", ""))
            input_ranges.append((0, int(dispatch_context.attributes["moeExpertNum"]) - 1))
        elif name == "scalesOptional":
            if int(dispatch_context.attributes.get("quantMode", 0)) == 2:
                h = combine_context.flatten_tensors[0].shape[1]
                if api_name == "aclnnMoeDistributeDispatchV2":
                    expert_num = int(
                        dispatch_context.attributes.get("sharedExpertRankNum", 0)
                    ) + int(dispatch_context.attributes.get("moeExpertNum", 0))
                else:
                    expert_num = int(
                        dispatch_context.attributes.get("sharedExpertNum", 0)
                    ) + int(
                        dispatch_context.attributes.get("moeExpertNum", 0)
                    )
                shapes.append((expert_num, h))
                dtypes.append("float32")
                input_ranges.append((-1, 1))
            else:
                shapes.append(None)
                dtypes.append(None)
                input_ranges.append((None, None))
        elif name == "xActiveMaskOptional":
            bs = int(source_x.shape[0])
            zero_expert_num = int(dispatch_context.attributes.get("zeroExpertNum", 0))
            copy_expert_num = int(dispatch_context.attributes.get("copyExpertNum", 0))
            const_expert_num = int(dispatch_context.attributes.get("constExpertNum", 0))
            shapes.append(
                (bs,)
                if int(dispatch_context.attributes.get("xActiveMaskDim", 1)) == 1
                else (bs, source_expert_ids.shape[1])
            )
            if (
                zero_expert_num + copy_expert_num + const_expert_num > 0
            ):
                shapes[-1] = (bs, 1)
            dtypes.append("bool")
            input_ranges.append((None, None))
            x_active_mask_dim = int(dispatch_context.attributes.get("xActiveMaskDim", 1))
            if x_active_mask_dim == 0:
                shapes[-1] = None
                dtypes[-1] = None
                input_ranges[-1] = (None, None)
        elif name == "expertScalesOptional":
            shapes.append(tuple(source_expert_ids.shape))
            dtypes.append("float32")
            input_ranges.append((0, 1))
        elif name == "performanceInfoOptional":
            shapes.append(None)
            dtypes.append("int64")
            input_ranges.append((None, None))
        elif name == "expandXOut":
            shapes.append(tuple(golden_expand_x.shape))
            if int(dispatch_context.attributes.get("quantMode", 0)) == 2:
                dtypes.append("int8")
            else:
                dtypes.append(str(golden_expand_x.dtype).replace("torch.", ""))
            input_ranges.append((None, None))
        elif name == "dynamicScalesOut":
            dynamic_scales = generated_case.get("dynamic_scales")
            if dynamic_scales is not None:
                shapes.append(tuple(dynamic_scales.shape))
                dtypes.append(str(dynamic_scales.dtype).replace("torch.", ""))
            else:
                shapes.append(tuple(golden_expand_x.shape[:1]))
                dtypes.append("float32")
            input_ranges.append((None, None))
        elif name == "assistInfoForCombineOut":
            shapes.append(tuple(golden_expand_idx.shape))
            dtypes.append("int32")
            input_ranges.append((None, None))
        elif name == "expertTokenNumsOut":
            local_expert_count = (
                1
                if rank_idx < int(dispatch_context.attributes.get("sharedExpertRankNum", 0))
                else max(int(dispatch_context.attributes.get("localMoeExpertNum", 0)), 1)
            )
            shapes.append((local_expert_count,))
            dtypes.append("int64")
            input_ranges.append((None, None))
        elif name == "epRecvCountsOut":
            shapes.append(tuple(golden_ep_recv_counts.shape))
            dtypes.append("int32")
            input_ranges.append((None, None))
        elif name == "tpRecvCountsOut":
            tp_world_size = int(dispatch_context.attributes.get("tpWorldSize", 1))
            shapes.append((max(tp_world_size, 1),))
            dtypes.append("int32")
            input_ranges.append((None, None))
        elif name == "expandScalesOut":
            shapes.append((max(source_x.shape[0], 1),))
            dtypes.append("float32")
            input_ranges.append((None, None))
        else:
            shapes.append(None)
            dtypes.append(None)
            input_ranges.append((None, None))

    dispatch_context.tensor_view_shapes = tuple(shapes)
    dispatch_context.tensor_dtypes = tuple(dtypes)
    dispatch_context.scalar_dtypes = ()
    dispatch_context.scalar_data_ranges = ()
    dispatch_context._scalar_list_dist = None
    dispatch_context.input_data_ranges = tuple(input_ranges)
    dispatch_context.output_inplace_indexes = ()
    dispatch_context.tensor_formats = tuple("ND" for _ in shapes)
    dispatch_context.tensor_storage_shapes = ()
    dispatch_context.tensor_view_strides = ()
    dispatch_context.tensor_view_offsets = ()
    dispatch_context._flat_tensors = None
    dispatch_context._flat_tensor_view_shapes = None
    dispatch_context._flat_tensor_dtypes = None
    dispatch_context.np_storages = None
    dispatch_context._tensor_list_dist = None
    dispatch_context._output_dtypes = None
    dispatch_context._pure_output_indexes = tuple(dispatch_context.output_tensor_indexes)
    dispatch_context._flat_tensor_storage_shapes = None
    dispatch_context._flat_tensor_view_strides = None
    dispatch_context._flat_tensor_view_offsets = None
    dispatch_context._flat_precision_tolerances = None
    dispatch_context._flat_absolute_precision = None
    dispatch_context._is_torch_dtype_support = None
    patch_moe_rank_attributes(
        dispatch_context,
        rank_idx,
        len(combine_context.device_ids or ()),
    )
    dispatch_context._output_view_shapes = None
    dispatch_context._pure_output_indexes = tuple(dispatch_context.output_tensor_indexes)
    dispatch_context._param_plan_cache = None
    dispatch_context._output_view_shapes = None
    dispatch_context._flat_tensor_view_shapes = None
    dispatch_context._flat_tensor_dtypes = None
    dispatch_context._flat_input_data_ranges = None
    dispatch_context._output_dtypes = None
    dispatch_context._pure_attrs = None
    dispatch_context._flat_tensors = None
    dispatch_context.np_storages = None
    dispatch_context.tensors = None

    return dispatch_context


def _moe_dispatch_inputs(*args, **kwargs):
    from ttk.core_modules.plugin_loader import get_plugin_function

    input_func = get_plugin_function(
        kwargs.get("api_name", "aclnnMoeDistributeDispatchV3"), "input", "aclnn"
    )
    if input_func is None:
        return args
    return input_func(*args, **kwargs)


def _moe_case_data_for_context(context, rank_idx):
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[4]
        / "common"
        / "tests"
        / "assets"
        / "impl"
        / "moe_case_data.py"
    )
    spec = importlib.util.spec_from_file_location("mc2_moe_case_data_cascade", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expand_x = context.flatten_tensors[0]
    expert_ids = context.flatten_tensors[1]
    remark = {
        item.split("=", 1)[0].strip(): item.split("=", 1)[1].strip()
        for item in str(context.remark or "").split(",")
        if "=" in item
    }
    config = module.MoeCaseConfig(
        global_bs=int(context.attributes.get("globalBs", expert_ids.shape[0])),
        hidden_size=int(expand_x.shape[1]),
        top_k=int(expert_ids.shape[1]),
        moe_expert_num=int(context.attributes.get("moeExpertNum", 1)),
        seed=int(remark.get("seed", 1)),
        ep_world_size=int(context.attributes.get("epWorldSize", len(context.device_ids or ()))),
        tp_world_size=int(context.attributes.get("tpWorldSize", 1)),
        local_moe_expert_num_override=int(
            context.attributes.get("localMoeExpertNum", 0)
        ),
        shared_expert_num=int(context.attributes.get("sharedExpertNum", 0)),
        shared_expert_rank_num=int(context.attributes.get("sharedExpertRankNum", 0)),
        const_expert_num=int(context.attributes.get("constExpertNum", 0)),
        zero_expert_num=int(context.attributes.get("zeroExpertNum", 0)),
        copy_expert_num=int(context.attributes.get("copyExpertNum", 0)),
        rank_num_per_shared_expert=int(
            context.attributes.get("rankNumPerSharedExpert", 0)
        ),
        input_dtype=str(expand_x.dtype).replace("torch.", ""),
        is_x_random=int(context.attributes.get("isXRandom", 0)) == 1,
        active_mask_dim=int(context.attributes.get("xActiveMaskDim", 0)),
        is_mask=int(context.attributes.get("xActiveMaskDim", 0)) > 0,
        is_variable_bs=bool(int(context.attributes.get("_moeVariableBs", 0))),
        is_quant=(
            int(context.attributes.get("quantMode", 0)) > 0
        ),
        is_dispatch_scales=bool(
            int(context.attributes.get("isDispatchScales", 0))
        ),
        is_topk_balance=(
            (
                context.attributes.get("_dispatchCommAlg")
                or context.attributes.get("commAlg")
            )
            == "fullmesh_v2"
        ),
        is_combine_x=(
            int(context.attributes.get("isCombineX", 0)) == 1
        ),
        is_shared_expert_x=bool(
            int(context.attributes.get("isSharedExpertX", 0))
        ),
        is_add_rms_norm=context.api_name
        in (
            "aclnnMoeDistributeCombineAddRmsNorm",
            "aclnnMoeDistributeCombineAddRmsNormV2",
        ),
        is_deepep=(
            remark.get("isDeepep") == "1"
            or int(context.attributes.get("_moeDeepep", 0) or 0) == 1
        ),
        is_elastic=int(context.attributes.get("isElastic", 0)) == 1,
        elastic_shared_rank_num=int(
            context.attributes.get("shareBrokenCardNum", 0)
        ),
        elastic_moe_rank_num=int(
            context.attributes.get("moeBrokenCardNum", 0)
        ),
        comm_quant_mode=int(context.attributes.get("commQuantMode", 0)),
        is_expand=int(context.attributes.get("isExpand", 0)) == 1,
    )
    return module.rank_case(
        module.load_or_generate_case(config), config, rank_idx
    )


def _moe_initial_case_data_for_context(context, rank_idx):
    module = _load_moe_case_data_module()
    config = _moe_case_config_for_context(context)
    return module.rank_case(
        module.load_or_generate_case(config), config, rank_idx
    )


def _case_available_ranks(context):
    module = _load_moe_case_data_module()
    config = _moe_case_config_for_context(context)
    return module.MoeCaseConfig.available_ranks.__get__(config)


def _load_moe_case_data_module():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[4]
        / "common"
        / "tests"
        / "assets"
        / "impl"
        / "moe_case_data.py"
    )
    spec = importlib.util.spec_from_file_location("mc2_moe_case_data_cascade", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _moe_case_config_for_context(context):
    module = _load_moe_case_data_module()
    expand_x = context.flatten_tensors[0]
    expert_ids = context.flatten_tensors[1]
    remark = {
        item.split("=", 1)[0].strip(): item.split("=", 1)[1].strip()
        for item in str(context.remark or "").split(",")
        if "=" in item
    }
    return module.MoeCaseConfig(
        global_bs=int(context.attributes.get("globalBs", expert_ids.shape[0])),
        hidden_size=int(expand_x.shape[1]),
        top_k=int(expert_ids.shape[1]),
        moe_expert_num=int(context.attributes.get("moeExpertNum", 1)),
        seed=int(remark.get("seed", 1)),
        ep_world_size=int(
            context.attributes.get("epWorldSize", len(context.device_ids or ()))
        ),
        tp_world_size=int(context.attributes.get("tpWorldSize", 1)),
        local_moe_expert_num_override=int(
            context.attributes.get("localMoeExpertNum", 0)
        ),
        shared_expert_num=int(context.attributes.get("sharedExpertNum", 0)),
        shared_expert_rank_num=int(
            context.attributes.get("sharedExpertRankNum", 0)
        ),
        const_expert_num=int(context.attributes.get("constExpertNum", 0)),
        zero_expert_num=int(context.attributes.get("zeroExpertNum", 0)),
        copy_expert_num=int(context.attributes.get("copyExpertNum", 0)),
        rank_num_per_shared_expert=int(
            context.attributes.get("rankNumPerSharedExpert", 0)
        ),
        input_dtype=str(expand_x.dtype).replace("torch.", ""),
        is_x_random=int(context.attributes.get("isXRandom", 0)) == 1,
        active_mask_dim=int(context.attributes.get("xActiveMaskDim", 0)),
        is_mask=int(context.attributes.get("xActiveMaskDim", 0)) > 0,
        is_variable_bs=bool(int(context.attributes.get("_moeVariableBs", 0))),
        is_quant=(int(context.attributes.get("quantMode", 0)) > 0),
        is_dispatch_scales=bool(
            int(context.attributes.get("isDispatchScales", 0))
        ),
        is_topk_balance=(
            (
                context.attributes.get("_dispatchCommAlg")
                or context.attributes.get("commAlg")
            )
            == "fullmesh_v2"
        ),
        is_combine_x=(int(context.attributes.get("isCombineX", 0)) == 1),
        is_shared_expert_x=bool(
            int(context.attributes.get("isSharedExpertX", 0))
        ),
        is_add_rms_norm=context.api_name
        in (
            "aclnnMoeDistributeCombineAddRmsNorm",
            "aclnnMoeDistributeCombineAddRmsNormV2",
        ),
        is_deepep=(
            remark.get("isDeepep") == "1"
            or int(context.attributes.get("_moeDeepep", 0) or 0) == 1
        ),
        is_elastic=int(context.attributes.get("isElastic", 0)) == 1,
        elastic_shared_rank_num=int(
            context.attributes.get("shareBrokenCardNum", 0)
        ),
        elastic_moe_rank_num=int(
            context.attributes.get("moeBrokenCardNum", 0)
        ),
        comm_quant_mode=int(context.attributes.get("commQuantMode", 0)),
        is_expand=int(context.attributes.get("isExpand", 0)) == 1,
    )


def _padded_moe_case_data_for_context(context, rank_idx):
    from ttk.core_modules.npu.op_api.mc2_dispatcher import _get_moe_dispatch_a

    generated = _moe_case_data_for_context(context, rank_idx)
    expand_x = generated.get("expand_x")
    if expand_x is not None:
        bs = int(generated["expert_ids"].shape[0])
        top_k = int(generated["expert_ids"].shape[1])
        a_value, _ = _get_moe_dispatch_a(
            context.attributes,
            rank_idx,
            bs,
            top_k,
        )
        if expand_x.shape[0] < a_value:
            padding = torch.zeros(
                (a_value - expand_x.shape[0], expand_x.shape[1]),
                dtype=expand_x.dtype,
            )
            generated["expand_x"] = torch.cat((expand_x, padding))
    return generated


def _execute_moe_add_rms_norm_e2e(thread_context, device, stream_value):
    if thread_context.api_name != "aclnnMoeDistributeCombineAddRmsNormV2":
        raise RuntimeError(
            f"Unexpected AddRmsNorm E2E API: {thread_context.api_name}"
        )

    from ttk.core_modules.npu.op_api.profiling_structure import ApiProfilingResult

    context = thread_context
    output_indexes = list(context.output_tensor_indexes)
    rank_idx = context.device_ids.index(context.my_rank)
    contexts = getattr(_execute_moe_add_rms_norm_e2e, "_thread_contexts", None)
    device_ids = getattr(_execute_moe_add_rms_norm_e2e, "_device_ids", None)
    rank_contexts = getattr(_execute_moe_add_rms_norm_e2e, "_rank_contexts", None)
    if contexts is not None and device_ids is not None and rank_contexts is not None:
        ready = getattr(_execute_moe_add_rms_norm_e2e, "_ready_contexts", set())
        completed = getattr(_execute_moe_add_rms_norm_e2e, "_completed_event", None)
        ready.add(id(context))
        if rank_idx == 0 and len(ready) < len(device_ids):
            deadline = time.monotonic() + 1800
            while len(ready) < len(device_ids) and time.monotonic() < deadline:
                time.sleep(0.01)
            if len(ready) < len(device_ids):
                raise TimeoutError("Moe AddRmsNorm V2 E2E ranks were not ready")
            try:
                _run_moe_add_rms_norm_e2e_process(contexts, device_ids, rank_contexts)
            finally:
                if completed is not None:
                    completed.set()
        elif completed is not None:
            completed.wait(1800)
            if not completed.is_set():
                raise TimeoutError("Moe AddRmsNorm V2 E2E rank0 did not finish")
    payload = rank_contexts.get(
        (device_ids[rank_idx] if device_ids else context.my_rank, "add_rms_norm_payload")
    )
    suppressed = bool(int(context.attributes.get("_moeElasticSuppressed", 0)))
    if suppressed:
        return ApiProfilingResult.fail("SUPPRESSED")
    if payload is not None:
        context._moe_add_rms_norm_e2e_payload = payload
    else:
        payload = getattr(context, "_moe_add_rms_norm_e2e_payload", None)
    if payload is None:
        raise RuntimeError("Moe AddRmsNorm V2 E2E payload is missing")
    output_bytes, output_shapes, payload_rank = payload
    if payload_rank != rank_idx or len(output_bytes[0]) != len(output_indexes):
        raise RuntimeError(
            f"Moe AddRmsNorm V2 E2E payload mismatch on rank {rank_idx}"
        )
    return ApiProfilingResult(
        True,
        "E2E_EXECUTED",
        "E2E_EXECUTED",
        tuple(output_bytes[0]),
        tuple(output_shapes),
    )


def _run_moe_add_rms_norm_e2e_process(thread_contexts, device_ids, rank_contexts):
    import json
    import subprocess
    import sys
    import tempfile

    from ttk.config.loader import get_cascade_config

    first_context = next(iter(thread_contexts.values()))
    if first_context.api_name != "aclnnMoeDistributeCombineAddRmsNormV2":
        return

    from ttk.core_modules.aclnn.op_api_info_keeper import OpApiInfoKeeper

    with tempfile.TemporaryDirectory(prefix="moe_add_rms_norm_e2e_") as run_dir:
        processes = []
        worker_jobs = []
        endpoint_port = _allocate_tcp_port()
        available_ranks = set(
            int(rank)
            for rank in first_context.attributes.get("_moeElasticAvailableRanks", range(len(device_ids)))
        )
        for rank_idx, device_id in enumerate(device_ids):
            input_path = str(Path(run_dir) / f"inputs_rank_{rank_idx}.npz")
            plan_path = str(Path(run_dir) / f"plan_rank_{rank_idx}.json")
            context = thread_contexts[device_id]
            tensors = list(context.flatten_tensors)
            tensor_names = OpApiInfoKeeper().info_of(context.api_name).tensors

            def tensor_by_name(name):
                index = next(
                    (
                        tensor_index
                        for tensor_index, tensor_name in enumerate(tensor_names)
                        if tensor_name == name
                    ),
                    -1,
                )
                return tensors[index] if index >= 0 else None

            output_path = str(Path(run_dir) / f"output_rank_{rank_idx}.npz")
            plan = {
                "rank": rank_idx,
                "output_path": output_path,
                "input_path": input_path,
                "seed": 42,
                "availableRanks": sorted(available_ranks),
                "attributes": {
                    key: value
                    for key, value in context.attributes.items()
                    if key in {
                        "epWorldSize", "tpWorldSize", "moeExpertNum",
                        "localMoeExpertNum",
                        "expertShardType", "sharedExpertNum", "sharedExpertRankNum",
                        "globalBs", "commQuantMode", "normEps", "zeroExpertNum",
                        "copyExpertNum", "constExpertNum",
                        "isElastic", "_moeVariableBs", "isActivateMask",
                        "xActiveMaskDim",
                    }
                },
                "elasticInfoOptional": encode(
                    getattr(golden_dispatch_context, "flatten_tensors", [None] * 5)[5]
                )
                if golden_dispatch_context is not None
                else "",
                "isExpand": int(context.attributes.get("isExpand", 0)),
            }
            with open(plan_path, "w", encoding="utf-8") as plan_file:
                json.dump(plan, plan_file)
            def tensor_to_numpy(tensor):
                if tensor is None:
                    return None
                value = tensor.detach().cpu()
                if value.dtype == torch.bfloat16:
                    value = value.view(torch.uint16)
                return value.numpy()

            save_arrays = {
                "expand_x": tensor_to_numpy(tensor_by_name("expandX")),
                "expert_ids": tensor_to_numpy(tensor_by_name("expertIds")),
                "expand_idx": tensor_to_numpy(tensor_by_name("assistInfoForCombine")),
                "ep_send_counts": tensor_to_numpy(tensor_by_name("epSendCounts")),
                "expert_scales": tensor_to_numpy(tensor_by_name("expertScales")),
                "residual_x": (
                    tensor_to_numpy(
                        tensor_by_name("residualX").reshape(
                            -1, 1, tensor_by_name("gamma").numel()
                        )
                    )
                    if tensor_by_name("residualX") is not None
                    and tensor_by_name("residualX").dim() == 2
                    else tensor_to_numpy(tensor_by_name("residualX"))
                ),
                "gamma": tensor_to_numpy(tensor_by_name("gamma")),
                "tp_send_counts": tensor_to_numpy(
                    tensor_by_name("tpSendCountsOptional")
                ),
                "x_active_mask": tensor_to_numpy(tensor_by_name("xActiveMaskOptional")),
                "elastic_info": tensor_to_numpy(
                    tensor_by_name("elasticInfoOptional")
                ),
                "shared_expert_x": tensor_to_numpy(
                    tensor_by_name("sharedExpertXOptional")
                ),
                "ori_x": tensor_to_numpy(tensor_by_name("oriXOptional")),
                "const_expert_alpha_1": tensor_to_numpy(
                    tensor_by_name("constExpertAlpha1Optional")
                ),
                "const_expert_alpha_2": tensor_to_numpy(
                    tensor_by_name("constExpertAlpha2Optional")
                ),
                "const_expert_v": tensor_to_numpy(
                    tensor_by_name("constExpertVOptional")
                ),
            }
            save_arrays = {
                key: value if value is not None else np.zeros(0)
                for key, value in save_arrays.items()
            }
            np.savez(input_path, **save_arrays)
            worker_jobs.append((rank_idx, device_id, input_path, plan_path, output_path))

        import gc

        for device_id in device_ids:
            thread_device = rank_contexts.get((device_id, "device"))
            if thread_device is None:
                continue
            for memory in tuple(thread_device._acl_device_memories):
                thread_device._free_device_memory(memory)
            thread_device._release_acl_memory()
            thread_device._acl_device_memories.clear()
            thread_device._acl_tensor_to_device_mem.clear()
        import ctypes

        handles = getattr(first_context, "_multi_device_hccl_handles", None)
        stream_handles = ()
        context_handles = ()
        if handles:
            stream_handles = tuple(handles.get("stm_vals", ()))
            context_handles = tuple(handles.get("ctx_vals", ()))
            hccl_dll = ctypes.CDLL("libhcomm.so")
            for handle in handles.get("extra_comms", ()):
                hccl_dll.HcclCommDestroy(ctypes.c_void_p(handle))
            for did in device_ids:
                hccl_dll.HcclCommDestroy(
                    ctypes.c_void_p(handles["ep_comm_handles"].get(did, 0))
                )
            for did in device_ids:
                hccl_dll.HcclCommDestroy(
                    ctypes.c_void_p(handles["comm_handles"][list(device_ids).index(did)])
                )
            handles["extra_comms"] = ()
            handles["ep_comm_handles"] = {}
            handles["comm_handles"] = {}
            handles["ctx_vals"] = ()
            handles["stm_vals"] = ()
            first_context._multi_device_hccl_handles = None
        gc.collect()
        torch.npu.empty_cache()
        if handles:
            acl_dll = ctypes.CDLL("libascendcl.so")
            for device_id in device_ids:
                if device_id >= len(stream_handles) or device_id >= len(context_handles):
                    continue
                acl_dll.aclrtSetDevice(ctypes.c_int32(device_id))
                acl_dll.aclrtSynchronizeStream(
                    ctypes.c_void_p(handles["stm_vals"][device_id])
                )
                acl_dll.aclrtDestroyStream(
                    ctypes.c_void_p(handles["stm_vals"][device_id])
                )
                acl_dll.aclrtDestroyContext(
                    ctypes.c_void_p(handles["ctx_vals"][device_id])
                )
                acl_dll.aclrtResetDevice(ctypes.c_int32(device_id))

        for rank_idx, device_id, input_path, plan_path, output_path in worker_jobs:
            env = os.environ.copy()
            hccl_port_range = get_cascade_config().hccl_port_range
            if hccl_port_range:
                env["HCCL_NPU_SOCKET_PORT_RANGE"] = hccl_port_range
            else:
                env.pop("HCCL_NPU_SOCKET_PORT_RANGE", None)
            env["HCCL_CONNECT_TIMEOUT"] = str(
                get_cascade_config().hccl_connect_timeout_s
            )
            if int(get_cascade_config().hccl_buffer_size) > 0:
                env["HCCL_BUFFSIZE"] = str(get_cascade_config().hccl_buffer_size)
            else:
                env.pop("HCCL_BUFFSIZE", None)
            env["MOE_ADD_RMS_NORM_PLAN"] = plan_path
            env["MOE_ADD_RMS_NORM_RANK"] = str(rank_idx)
            env["MOE_ADD_RMS_NORM_DEVICE"] = str(device_id)
            env["MOE_ADD_RMS_NORM_ENDPOINT"] = f"tcp://127.0.0.1:{endpoint_port}"
            worker_path = (
                Path(__file__).resolve().parent / "add_rms_norm_e2e_worker.py"
            )
            command = [sys.executable, str(worker_path)]
            logging.info("Moe AddRmsNorm V2 E2E process rank=%s cmd=%s", rank_idx, command)
            processes.append(
                (
                    rank_idx,
                    device_id,
                    subprocess.Popen(
                        command,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    ),
                    output_path,
                )
            )


        for rank_idx, device_id, process, output_path in processes:
            try:
                output, _ = process.communicate(timeout=1800)
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate()
                raise RuntimeError(
                    f"Moe AddRmsNorm V2 E2E worker timed out on rank {rank_idx}"
                ) from None
            if output:
                logging.info(
                    "Moe AddRmsNorm V2 E2E worker rank=%s output:\n%s",
                    rank_idx,
                    output,
                )
            if process.returncode != 0:
                logging.error(
                    "Moe AddRmsNorm V2 E2E worker rank=%s failed output:\n%s",
                    rank_idx,
                    output,
                )
                raise RuntimeError(
                    f"Moe AddRmsNorm V2 E2E worker failed on rank {rank_idx}: "
                    f"rc={process.returncode}"
                )
            if rank_idx not in available_ranks:
                continue
            if not Path(output_path).is_file():
                raise RuntimeError(f"Moe AddRmsNorm V2 E2E output missing on rank {rank_idx}")
            with np.load(output_path, allow_pickle=False) as payload:
                rank_contexts[(device_id, "add_rms_norm_payload")] = (
                    (
                        [payload[f"actual_{index}"].tobytes() for index in range(3)],
                        [payload[f"reference_{index}"].tobytes() for index in range(3)],
                    ),
                    [tuple(shape) for shape in payload["output_shapes"]],
                    int(payload["rank"]),
                )


def _allocate_tcp_port():
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _fill_dispatch_inputs(dispatch_context, rank_idx):
    from ttk.core_modules.npu.op_api.input_generation import InputGenerator
    from ttk.core_modules.npu.op_api.profiling_structure import ApiProfilingResult
    from ttk.core_modules.aclnn.op_api_info_keeper import OpApiInfoKeeper
    from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor

    InputGenerator(dispatch_context).gen()
    flat_tensors = list(dispatch_context.flatten_tensors)
    np_storages = list(dispatch_context.np_storages)
    api_tensor_names = tuple(OpApiInfoKeeper().info_of(dispatch_context.api_name).tensors)
    tp_output_index = api_tensor_names.index("tpRecvCountsOut")
    for index in dispatch_context.output_tensor_indexes:
        if flat_tensors[index] is not None:
            flat_tensors[index].zero_()
    if tp_output_index in dispatch_context.output_tensor_indexes and flat_tensors[tp_output_index] is not None:
        flat_tensors[tp_output_index].fill_(1)

    api_tensor_names = tuple(OpApiInfoKeeper().info_of(dispatch_context.api_name).tensors)
    elastic_index = next(
        (
            index
            for index, name in enumerate(api_tensor_names)
            if name == "elasticInfoOptional"
        ),
        None,
    )
    if elastic_index is not None:
        flat_tensors[elastic_index] = None
        np_storages[elastic_index] = None
    if dispatch_context.tensors is not None:
        kwargs = dict(dispatch_context.attributes)
        if int(dispatch_context.attributes.get("_moeDeepep", 0) or 0) == 1:
            original_expert_ids = dispatch_context.flatten_tensors[1]
            if (
                original_expert_ids is not None
                and original_expert_ids.dim() > 1
                and original_expert_ids.shape[1] > 1
            ):
                kwargs["_moeDeepepOriginalTopK"] = int(original_expert_ids.shape[1])
        kwargs.update(
            {
                "api_name": dispatch_context.api_name,
                "testcase_name": dispatch_context.testcase_name,
                "remark": dispatch_context.remark,
                "xActiveMaskOptional": int(
                    dispatch_context.attributes.get("xActiveMaskDim", 0)
                    if int(dispatch_context.attributes.get("isActivateMask", 0))
                    else 0
                ),
            }
        )
        tensor_args = list(dispatch_context.flatten_tensors)[:6]
        tensor_args.extend(None for _ in range(6 - len(tensor_args)))
        adjusted = _moe_dispatch_inputs(*tensor_args, **kwargs)
        if adjusted is not None:
            adjusted = tuple(adjusted)
            if (
                dispatch_context.api_name == "aclnnMoeDistributeDispatchV2"
                and int(dispatch_context.attributes.get("quantMode", 0)) == 2
                and len(adjusted) > 2
                and adjusted[2] is not None
            ):
                generated = _moe_case_data_for_context(dispatch_context, rank_idx)
                padded_scales = generated.get("padded_scales")
                if padded_scales is None:
                    raise RuntimeError(
                        "MoeDistributeDispatchV2 requires padded scales, but the golden data did not generate them"
                    )
                adjusted = adjusted[:2] + (padded_scales,) + tuple(adjusted[3:])
            original_tensors = list(dispatch_context.tensors)
            original_tensors[: len(adjusted)] = adjusted
            dispatch_context.tensors = tuple(original_tensors)
            dispatch_context._flat_tensors = None
            dispatch_context._flat_tensor_view_shapes = None
            flat_tensors = list(dispatch_context.flatten_tensors)
            flat_shapes = list(dispatch_context.flat_tensor_view_shapes)
            for flat_index, flat_tensor in enumerate(flat_tensors):
                if flat_tensor is None or flat_index >= len(flat_shapes):
                    continue
                flat_shapes[flat_index] = tuple(flat_tensor.shape)
            dispatch_context._flat_tensor_view_shapes = tuple(flat_shapes)
            np_storages = list(dispatch_context.np_storages)
            for index, tensor in enumerate(flat_tensors):
                if tensor is not None and index in (0, 1, 2, 3, 4):
                    from ttk.utilities.dtypes import torch_to_numpy_tensor

                    np_storages[index] = torch_to_numpy_tensor(tensor.detach().cpu()).reshape(-1)
    for index in (0, 1, 2, 3, 4):
        target = flat_tensors[index]
        if target is None:
            continue
        source = target
        if index == 2 and int(dispatch_context.attributes.get("quantMode", 0)) == 0:
            continue
        if int(dispatch_context.attributes.get("_moeVariableBs", 0)) and index in (0, 1, 4):
            continue
        logging.debug(
            "Moe dispatch input source dev=%s index=%s shape=%s first=%s mask=%s",
            rank_idx,
            index,
            tuple(target.shape),
            source.reshape(-1)[:1].tolist(),
            None,
        )
        if isinstance(source, torch.Tensor):
            from ttk.utilities.dtypes import torch_to_numpy_tensor

            source = torch_to_numpy_tensor(source.detach().cpu())
        else:
            source = np.asarray(source)
        target_dtype = target.dtype if isinstance(target, np.ndarray) else np.dtype(str(target.dtype).replace("torch.", ""))
        target_size = int(target.size) if isinstance(target, np.ndarray) else int(target.numel())
        if tuple(source.shape) != tuple(target.shape):
            source = source.reshape(-1)[:target_size]
            if source.size < target_size:
                source = np.pad(source, (0, target_size - source.size))
        if "float8_e8m0" in str(target_dtype):
            source = source.astype(np.uint8, copy=False)
        else:
            source = source.astype(target_dtype, copy=False)
        if isinstance(target, np.ndarray):
            target.reshape(-1)[:] = source
        else:
            target.copy_(numpy_to_torch_tensor(source.reshape(tuple(target.shape))))
        np_storages[index] = source.reshape(-1)

    dispatch_context._flat_tensors = flat_tensors
    dispatch_context.np_storages = np_storages


def _make_dispatch_warmup_context(dispatch_context, source_x, source_expert_ids):
    from ttk.core_modules.aclnn.op_api_info_keeper import OpApiInfoKeeper
    from ttk.core_modules.npu.op_api.mc2_dispatcher import patch_moe_rank_attributes
    from ttk.core_modules.testcase_manager import TestcaseAclnn

    op_api_info = OpApiInfoKeeper().info_of(dispatch_context.api_name)
    warmup_context = TestcaseAclnn()
    warmup_context.api_name = dispatch_context.api_name
    warmup_context.device_ids = dispatch_context.device_ids
    warmup_context.my_rank = dispatch_context.my_rank
    warmup_context.remark = dispatch_context.remark
    warmup_context.attributes = dict(dispatch_context.attributes)
    warmup_context.attributes.update(
        {
            "globalBs": len(dispatch_context.device_ids),
            "quantMode": 2,
            "expertTokenNumsType": 1,
        }
    )

    hidden_size = source_x.shape[1]
    topk = source_expert_ids.shape[1]
    ep_world_size = int(warmup_context.attributes.get("epWorldSize", len(dispatch_context.device_ids)))
    shared_rank_num = int(warmup_context.attributes.get("sharedExpertRankNum", 0))
    shared_expert_num = int(warmup_context.attributes.get("sharedExpertNum", 0))
    local_expert_num = max(
        ep_world_size - shared_rank_num,
        1,
    )
    rank_num_per_shared_expert = max(shared_rank_num // max(shared_expert_num, 1), 1)
    max_shared_group_num = -(-ep_world_size // rank_num_per_shared_expert)
    a_value = max(max_shared_group_num, topk)

    shapes = []
    dtypes = []
    input_ranges = []
    for name in op_api_info.tensors:
        if name == "x":
            shapes.append((1, hidden_size))
            dtypes.append(str(source_x.dtype).replace("torch.", ""))
            input_ranges.append((-1024, 1024))
        elif name == "expertIds":
            shapes.append((1, topk))
            dtypes.append("int32")
            input_ranges.append((0, max(topk - 1, 0)))
        elif name == "dynamicScalesOut":
            shapes.append((a_value,))
            dtypes.append("float32")
            input_ranges.append((None, None))
        elif name == "assistInfoForCombineOut":
            shapes.append((max(topk, a_value * 128),))
            dtypes.append("int32")
            input_ranges.append((None, None))
        elif name == "expertTokenNumsOut":
            shapes.append((local_expert_num,))
            dtypes.append("int64")
            input_ranges.append((None, None))
        elif name == "epRecvCountsOut":
            shapes.append((ep_world_size * local_expert_num,))
            dtypes.append("int32")
            input_ranges.append((None, None))
        elif name == "tpRecvCountsOut":
            tp_world_size = int(warmup_context.attributes.get("tpWorldSize", 1))
            if tp_world_size > 1:
                shapes.append((tp_world_size,))
                dtypes.append("int32")
            else:
                shapes.append((0,))
                dtypes.append("int32")
            input_ranges.append((None, None))
        elif name == "expandXOut":
            shapes.append((a_value, hidden_size))
            dtypes.append("int8")
            input_ranges.append((None, None))
        elif name == "expandScalesOut":
            shapes.append((a_value,))
            dtypes.append("float32")
            input_ranges.append((None, None))
        elif name == "scalesOptional":
            shapes.append(None)
            dtypes.append("float32")
            input_ranges.append((None, None))
        elif name == "xActiveMaskOptional":
            shapes.append(None)
            dtypes.append("bool")
            input_ranges.append((None, None))
        elif name == "expertScalesOptional":
            shapes.append(None)
            dtypes.append("float32")
            input_ranges.append((None, None))
        elif name == "elasticInfoOptional":
            shapes.append(None)
            dtypes.append("int32")
            input_ranges.append((None, None))
        elif name == "performanceInfoOptional":
            shapes.append(None)
            dtypes.append("int64")
            input_ranges.append((None, None))
        else:
            shapes.append(None)
            dtypes.append(None)
            input_ranges.append((None, None))

    warmup_context.tensor_view_shapes = tuple(shapes)
    warmup_context.tensor_dtypes = tuple(dtypes)
    warmup_context.scalar_dtypes = ()
    warmup_context.scalar_data_ranges = ()
    warmup_context.input_data_ranges = tuple(input_ranges)
    warmup_context.output_tensor_indexes = dispatch_context.output_tensor_indexes
    warmup_context.output_inplace_indexes = ()
    warmup_context.tensor_formats = tuple("ND" for _ in shapes)
    warmup_context.tensor_storage_shapes = ()
    warmup_context.tensor_view_strides = ()
    warmup_context.tensor_view_offsets = ()
    patch_moe_rank_attributes(
        warmup_context,
        int(dispatch_context.attributes.get("epRankId", dispatch_context.my_rank or 0)),
        len(dispatch_context.device_ids or ()),
    )
    warmup_context._param_plan_cache = None
    return warmup_context


def _load_deepep_dispatch_payload(
    thread_contexts, device_ids, dispatch_context, device_id, rank_idx
):
    import importlib.util
    import json
    import subprocess
    import sys
    import tempfile
    from ttk.core_modules.aclnn.op_api_info_keeper import OpApiInfoKeeper
    from ttk.config.loader import get_cascade_config, get_moe_config
    from ttk.core_modules.npu.op_api.profiling_structure import ApiProfilingResult
    from ttk.utilities import get_global_storage
    from ttk.utilities import set_global_storage
    from ttk.utilities.classes import SWITCHES

    state = getattr(_load_deepep_dispatch_payload, "_state", None)
    state_key = (tuple(device_ids), dispatch_context.testcase_name)
    if state is not None and state["key"] == state_key:
        payload = state["payloads"].get(device_id)
        if payload is None:
            return ApiProfilingResult.fail("SUPPRESSED")
        output_bytes, output_shapes, payload_rank = payload
        if payload_rank != rank_idx:
            raise RuntimeError(
                f"Moe DeepEP dispatch payload rank mismatch: {payload_rank} != {rank_idx}"
            )
        output_indexes = dispatch_context.output_tensor_indexes
        return ApiProfilingResult(
            True,
            "E2E_EXECUTED",
            "E2E_EXECUTED",
            tuple(output_bytes),
            tuple(output_shapes),
        )

    lock = getattr(_load_deepep_dispatch_payload, "_lock", None)
    if lock is None:
        import threading

        lock = threading.Lock()
        _load_deepep_dispatch_payload._lock = lock
    with lock:
        state = getattr(_load_deepep_dispatch_payload, "_state", None)
        if state is not None and state["key"] == state_key:
            payload = state["payloads"].get(device_id)
            if payload is None:
                return ApiProfilingResult.fail("SUPPRESSED")
            output_bytes, output_shapes, payload_rank = payload
            if payload_rank != rank_idx:
                raise RuntimeError(
                    f"Moe DeepEP dispatch payload rank mismatch: {payload_rank} != {rank_idx}"
                )
            output_indexes = dispatch_context.output_tensor_indexes
            return ApiProfilingResult(
                True,
                "E2E_EXECUTED",
                "E2E_EXECUTED",
                tuple(output_bytes),
                tuple(output_shapes),
            )

        return _generate_deepep_dispatch_payload(
            thread_contexts,
            device_ids,
            dispatch_context,
            device_id,
            rank_idx,
            state_key,
        )


def _generate_deepep_dispatch_payload(
    thread_contexts,
    device_ids,
    dispatch_context,
    device_id,
    rank_idx,
    state_key,
):
    import importlib.util
    import shutil
    import tempfile
    from ttk.config.loader import get_cascade_config, get_moe_config
    from ttk.core_modules.npu.op_api.profiling_structure import ApiProfilingResult
    from ttk.utilities import get_global_storage, set_global_storage
    payload_contexts = getattr(_run_moe_dispatch_cascade, "_rank_contexts", {})
    for payload_device in device_ids:
        payload_context = payload_contexts.get(payload_device)
        if payload_context is None:
            continue
        if "_moeDeepepExpertIdsExpand" not in payload_context.attributes:
            expert_ids = payload_context.flatten_tensors[1]
            payload_context.attributes["_moeDeepepExpertIdsExpand"] = (
                expert_ids[:, :1].contiguous()
                if expert_ids is not None and expert_ids.dim() > 1
                else expert_ids
            )

    module_path = (
        Path(__file__).resolve().parents[4]
        / "moe_distribute_dispatch"
        / "tests"
        / "assets"
        / "impl"
        / "elastic_dispatch_executor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mc2_moe_distribute_dispatch_executor", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    current_storage = get_global_storage()
    original_plugin_path = getattr(current_storage, "plugin_path", None)
    cache_root = Path(get_moe_config().cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        if not getattr(current_storage, "plugin_path", None):
            set_global_storage(SWITCHES())
            current_storage = get_global_storage()
        plugin_path = original_plugin_path or getattr(
            get_global_storage(), "plugin_path", None
        )
        payloads = None
        last_error = None
        for attempt in range(3):
            run_dir = tempfile.mkdtemp(
                prefix=f"ttk_moe_elastic_attempt{attempt}_", dir=str(cache_root)
            )
            try:
                module_payloads = module._run_external_worker(
                    payload_contexts, tuple(device_ids), run_dir
                )
                payloads = {
                    device: payload
                    for device, payload in module_payloads.items()
                    if payload is not None
                }
                shutil.rmtree(run_dir, ignore_errors=True)
                break
            except BaseException as error:
                last_error = error
                logging.exception(
                    "Moe DeepEP dispatch external worker failed attempt=%s run_dir=%s",
                    attempt,
                    run_dir,
                )
                if attempt == 2:
                    raise
                time.sleep(1)
        if payloads is None:
            raise last_error
    finally:
        if original_plugin_path is not None:
            current_storage.plugin_path = original_plugin_path

    _load_deepep_dispatch_payload._state = {
        "key": state_key,
        "payloads": payloads,
    }
    _run_moe_dispatch_cascade._deepep_cascade_payloads = payloads

    output_indexes = dispatch_context.output_tensor_indexes
    output_bytes, output_shapes, payload_rank = payloads[device_id]
    if payload_rank != rank_idx:
        raise RuntimeError(
            f"Moe DeepEP dispatch payload rank mismatch: {payload_rank} != {rank_idx}"
        )
    return ApiProfilingResult(
        True,
        "E2E_EXECUTED",
        "E2E_EXECUTED",
        tuple(output_bytes[index] for index in range(len(output_bytes))),
        tuple(output_shapes[index] for index in range(len(output_shapes))),
    )


def _replace_combine_inputs(
    combine_context, dispatch_context, device_id, rank_idx
):
    from ttk.core_modules.aclnn.op_api_info_keeper import OpApiInfoKeeper
    from ttk.utilities.dtypes import torch_to_numpy_tensor

    output_indexes = dispatch_context.output_tensor_indexes
    dispatch_outputs = list(dispatch_context.prof_result.output_bytes)
    combine_tensors = list(combine_context.flatten_tensors)
    dispatch_tensor_names = OpApiInfoKeeper().info_of(dispatch_context.api_name).tensors

    def decode_output(output_index):
        output_position = output_indexes.index(output_index)
        dtype_name = dispatch_context.flat_tensor_dtypes[output_index]
        from ttk.utilities.dtypes import resolve_custom_numpy_dtypes

        dtype = resolve_custom_numpy_dtypes([dtype_name])[0]
        raw = dispatch_outputs[output_position]
        if raw is None or isinstance(raw, str):
            raise RuntimeError(f"Missing dispatch output at tensor index {output_index}")
        decoded = np.frombuffer(raw, dtype=dtype).copy()
        torch_dtype = getattr(torch, dtype_name) if hasattr(torch, dtype_name) else None
        if torch_dtype is None:
            raise TypeError(f"Unsupported dispatch output dtype: {dtype_name}")
        if decoded.size == 0:
            return torch.empty((0,), dtype=torch_dtype)
        if output_position < len(output_shapes):
            output_shape = tuple(output_shapes[output_position])
        else:
            output_shape = dispatch_context.flatten_tensors[output_index].shape
        return torch.frombuffer(decoded, dtype=torch_dtype).reshape(output_shape)

    is_combine_v4 = combine_context.api_name == "aclnnMoeDistributeCombineV4"
    is_combine_v3 = combine_context.api_name == "aclnnMoeDistributeCombineV3"
    is_dispatch_v4 = dispatch_context.api_name == "aclnnMoeDistributeDispatchV4"
    is_deep_ep = int(dispatch_context.attributes.get("_moeDeepep", 0) or 0) == 1
    is_variable_bs = bool(
        int(combine_context.attributes.get("_moeVariableBs", 0) or 0)
    )
    dispatch_tp_output_index = (
        dispatch_tensor_names.index("tpRecvCountsOut")
        if "tpRecvCountsOut" in dispatch_tensor_names
        else -1
    )
    dispatch_expand_x_index = dispatch_tensor_names.index("expandXOut")
    dispatch_assist_index = dispatch_tensor_names.index("assistInfoForCombineOut")
    dispatch_ep_counts_index = dispatch_tensor_names.index("epRecvCountsOut")
    dispatch_dynamic_scales_index = dispatch_tensor_names.index("dynamicScalesOut")
    tp_ws = int(combine_context.attributes.get("tpWorldSize", 1))
    expand_x_index = 0
    assist_info_index = 2
    ep_send_counts_index = 3
    is_add_rms_norm = combine_context.api_name in (
        "aclnnMoeDistributeCombineAddRmsNorm",
        "aclnnMoeDistributeCombineAddRmsNormV2",
    )
    tp_send_counts_index = 7 if is_add_rms_norm else 5
    x_active_mask_index = 8 if is_add_rms_norm else 6
    shared_expert_x_index = 13 if is_add_rms_norm else 11
    expand_scales_index = 12 if is_add_rms_norm else 10
    ori_x_index = -1
    combine_tensor_names = OpApiInfoKeeper().info_of(combine_context.api_name).tensors
    ori_x_index = next(
        (index for index, name in enumerate(combine_tensor_names) if name == "oriXOptional"),
        -1,
    )
    const_expert_alpha_1_index = next(
        (index for index, name in enumerate(combine_tensor_names) if name == "constExpertAlpha1Optional"),
        -1,
    )
    const_expert_alpha_2_index = next(
        (index for index, name in enumerate(combine_tensor_names) if name == "constExpertAlpha2Optional"),
        -1,
    )
    const_expert_v_index = next(
        (index for index, name in enumerate(combine_tensor_names) if name == "constExpertVOptional"),
        -1,
    )
    elastic_info_index = next(
        (index for index, name in enumerate(combine_tensor_names) if name == "elasticInfoOptional"),
        -1,
    )
    generated = _padded_moe_case_data_for_context(combine_context, rank_idx)
    if combine_tensors[expand_x_index] is not None:
        target_shape = combine_tensors[expand_x_index].shape
        use_generated_expand_x = (
            is_deep_ep
            or bool(
                combine_context.attributes.get("_moeCascadeGoldenExpandX", 0)
            )
        )
        if use_generated_expand_x:
            dispatch_expand_x = generated["expand_x"]
            target_dtype = dispatch_expand_x.dtype
            if target_dtype == torch.int8:
                tensor_dtypes = list(combine_context.tensor_dtypes)
                target_dtype = getattr(
                    torch,
                    tensor_dtypes[expand_x_index],
                    dispatch_expand_x.dtype,
                )
            padded_dispatch_expand_x = torch.zeros(
                target_shape,
                dtype=target_dtype,
                device=dispatch_expand_x.device,
            )
            valid_rows = min(dispatch_expand_x.shape[0], target_shape[0])
            padded_dispatch_expand_x[:valid_rows].copy_(
                dispatch_expand_x[:valid_rows].to(target_dtype)
            )
            combine_tensors[expand_x_index] = padded_dispatch_expand_x
        else:
            dispatch_expand_x = decode_output(dispatch_expand_x_index)
            if dispatch_expand_x.dtype == torch.int8:
                target_dtype = getattr(
                    combine_context,
                    "_moeCombineInputDtype",
                    None,
                )
                if target_dtype is None:
                    tensor_dtypes = list(combine_context.tensor_dtypes)
                    target_dtype = getattr(
                        torch,
                        tensor_dtypes[expand_x_index],
                        dispatch_expand_x.dtype,
                    )
                dispatch_expand_x = dispatch_expand_x.to(target_dtype)
        padded_dispatch_expand_x = torch.zeros(
            target_shape,
            dtype=dispatch_expand_x.dtype,
            device=dispatch_expand_x.device,
        )
        valid_rows = min(dispatch_expand_x.shape[0], target_shape[0])
        padded_dispatch_expand_x[:valid_rows].copy_(
            dispatch_expand_x[:valid_rows]
        )
        combine_tensors[expand_x_index] = padded_dispatch_expand_x
    else:
        dispatch_expand_x = None
    if combine_tensors[1] is not None:
        combine_tensors[1] = generated["expert_ids"].to(combine_tensors[1].dtype).clone()
    if combine_tensors[4] is not None:
        combine_tensors[4] = generated["expert_scales"].to(combine_tensors[4].dtype).clone()
    logging.info(
        "Moe cascade generated combine inputs rank=%s expand_x=%s expert_ids=%s",
        rank_idx,
        tuple(combine_tensors[expand_x_index].shape) if combine_tensors[expand_x_index] is not None else None,
        tuple(combine_tensors[1].shape) if combine_tensors[1] is not None else None,
    )
    if is_deep_ep and int(combine_context.attributes.get("isExpand", 0)) == 1:
        combine_tensors[1] = generated["expert_ids_expand"].clone()
    if dispatch_context is None:
        dispatch_assist_info = generated["expand_idx"].reshape(-1)
    elif is_deep_ep:
        dispatch_assist_info = generated["expand_idx"].reshape(-1)
    else:
        dispatch_assist_info = decode_output(dispatch_assist_index).reshape(-1)
    target_assist_length = combine_tensors[assist_info_index].numel()
    if dispatch_assist_info.numel() != target_assist_length:
        if dispatch_assist_info.numel() > target_assist_length:
            dispatch_assist_info = dispatch_assist_info[:target_assist_length]
    if dispatch_assist_info.numel() < target_assist_length:
        padded_assist_info = torch.zeros(
            target_assist_length,
            dtype=dispatch_assist_info.dtype,
            device=dispatch_assist_info.device,
        )
        padded_assist_info[: dispatch_assist_info.numel()].copy_(dispatch_assist_info)
        dispatch_assist_info = padded_assist_info
    if dispatch_context is None:
        dispatch_ep_send_counts = generated["ep_recv_counts"]
    elif is_deep_ep:
        dispatch_ep_send_counts = generated["ep_recv_counts"]
    else:
        dispatch_ep_send_counts = decode_output(dispatch_ep_counts_index)
    target_ep_count_size = combine_tensors[ep_send_counts_index].numel()
    if dispatch_ep_send_counts.numel() < target_ep_count_size:
        dispatch_ep_send_counts = torch.cat(
            [
                dispatch_ep_send_counts,
                torch.full(
                    (target_ep_count_size - dispatch_ep_send_counts.numel(),),
                    int(dispatch_ep_send_counts[-1]) if dispatch_ep_send_counts.numel() else 0,
                    dtype=dispatch_ep_send_counts.dtype,
                ),
            ]
        )
    dispatch_tp_send_counts = None
    if is_deep_ep:
        dispatch_tp_send_counts = torch.zeros((max(tp_ws, 1),), dtype=torch.int32)
    elif (
        dispatch_context is not None
        and dispatch_tp_output_index in dispatch_context.output_tensor_indexes
    ):
        dispatch_tp_send_counts = decode_output(dispatch_tp_output_index)
    if dispatch_tp_send_counts is not None:
        if combine_tensors[tp_send_counts_index] is None:
            combine_tensors[tp_send_counts_index] = dispatch_tp_send_counts.to(
                torch.int32
            ).clone()
            tensor_shapes = list(combine_context.tensor_view_shapes)
            tensor_dtypes = list(combine_context.tensor_dtypes)
            tensor_shapes[tp_send_counts_index] = tuple(dispatch_tp_send_counts.shape)
            tensor_dtypes[tp_send_counts_index] = "int32"
            combine_context.tensor_view_shapes = tuple(tensor_shapes)
            combine_context.tensor_dtypes = tuple(tensor_dtypes)
            combine_context._tensor_list_dist = None
            combine_context._flat_tensor_view_shapes = None
            combine_context._flat_tensor_dtypes = None
    combine_tensors[assist_info_index] = dispatch_assist_info.reshape(-1).to(
        combine_tensors[assist_info_index].dtype
    )
    if dispatch_ep_send_counts.numel() == combine_tensors[ep_send_counts_index].numel():
        combine_tensors[ep_send_counts_index].copy_(dispatch_ep_send_counts.reshape(-1))
    else:
        combine_tensors[ep_send_counts_index] = dispatch_ep_send_counts.to(
            combine_tensors[ep_send_counts_index].dtype
        ).clone()
        tensor_shapes = list(combine_context.tensor_view_shapes)
        tensor_shapes[ep_send_counts_index] = tuple(dispatch_ep_send_counts.shape)
        combine_context.tensor_view_shapes = tuple(tensor_shapes)
        combine_context._tensor_list_dist = None
        combine_context._flat_tensor_view_shapes = None
    if dispatch_tp_send_counts is not None and combine_tensors[tp_send_counts_index] is not None:
        combine_tensors[tp_send_counts_index] = dispatch_tp_send_counts.to(
            combine_tensors[tp_send_counts_index].dtype
        ).clone()
    def copy_generated(target, source):
        if target is None or source is None:
            return
        flattened = source.reshape(-1)[: target.numel()].to(target.dtype)
        if flattened.numel() < target.numel():
            flattened = torch.cat(
                [flattened, torch.zeros(target.numel() - flattened.numel(), dtype=flattened.dtype)]
            )
        target.copy_(flattened.reshape(target.shape))

    if ori_x_index >= 0 and combine_tensors[ori_x_index] is not None:
        copy_generated(combine_tensors[ori_x_index], generated["ori_x"])
    if combine_tensors[x_active_mask_index] is not None:
        copy_generated(
            combine_tensors[x_active_mask_index],
            generated.get("x_active_mask_input", generated["x_active_mask"]),
        )
    if combine_tensors[shared_expert_x_index] is not None:
        copy_generated(
            combine_tensors[shared_expert_x_index],
            generated["shared_expert_x"],
        )
    if elastic_info_index >= 0 and combine_tensors[elastic_info_index] is not None:
        combine_tensors[elastic_info_index] = generated["elastic_info"].reshape(
            -1
        ).clone()
        tensor_shapes = list(combine_context.tensor_view_shapes)
        tensor_shapes[elastic_info_index] = (
            combine_tensors[elastic_info_index].shape
        )
        combine_context.tensor_view_shapes = tuple(tensor_shapes)
    if elastic_info_index >= 0 and combine_tensors[elastic_info_index] is not None:
        expand_scales_tensor = combine_tensors[expand_scales_index]
        mask_prefix = (
            combine_tensors[x_active_mask_index].reshape(-1)[:8].tolist()
            if combine_tensors[x_active_mask_index] is not None
            else None
        )
        ori_x_prefix = (
            combine_tensors[ori_x_index].reshape(-1)[:6].tolist()
            if ori_x_index >= 0 and combine_tensors[ori_x_index] is not None
            else None
        )
        logging.info(
            "Moe cascade combine inputs rank=%s expand_x=%s..%s mask=%s ori_x=%s expand_scales=%s "
            "ep_counts=%s..%s elastic_info=%s",
            rank_idx,
            combine_tensors[expand_x_index].reshape(-1)[:6].tolist()
            if combine_tensors[expand_x_index] is not None
            else None,
            combine_tensors[expand_x_index].reshape(-1)[-6:].tolist()
            if combine_tensors[expand_x_index] is not None
            else None,
            mask_prefix,
            ori_x_prefix,
            expand_scales_tensor.reshape(-1).tolist()
            if expand_scales_tensor is not None
            else None,
            combine_tensors[ep_send_counts_index].reshape(-1)[:6].tolist(),
            combine_tensors[ep_send_counts_index].reshape(-1)[-6:].tolist(),
            combine_tensors[elastic_info_index].reshape(-1).tolist(),
        )
    for tensor_index in (
        const_expert_alpha_1_index,
        const_expert_alpha_2_index,
        const_expert_v_index,
    ):
        if tensor_index >= 0 and combine_tensors[tensor_index] is not None:
            copy_generated(
                combine_tensors[tensor_index],
                generated[
                    {
                        const_expert_alpha_1_index: "const_expert_alpha_1",
                        const_expert_alpha_2_index: "const_expert_alpha_2",
                        const_expert_v_index: "const_expert_v",
                    }[tensor_index]
                ],
            )

    if is_add_rms_norm:
        residual_x_index = 5
        expected_shape = (
            int(combine_tensors[1].shape[0]),
            1,
            int(combine_tensors[6].numel()),
        )
        residual_x = generated["residual_x"]
        if residual_x.numel() != int(np.prod(expected_shape)):
            raise ValueError(
                f"Moe AddRmsNorm residual input size mismatch on rank {rank_idx}: "
                f"got {tuple(residual_x.shape)}, expected {expected_shape}"
            )
        if is_variable_bs:
            row_mask = combine_tensors[x_active_mask_index].reshape(-1).to(torch.bool)
            residual_x = residual_x[row_mask]
        combine_tensors[residual_x_index] = residual_x.reshape(expected_shape)
        tensor_shapes = list(combine_context.tensor_view_shapes)
        tensor_shapes[residual_x_index] = tuple(combine_tensors[residual_x_index].shape)
        combine_context.tensor_view_shapes = tuple(tensor_shapes)
        combine_context._tensor_list_dist = None
        combine_context._flat_tensor_view_shapes = None
        x_active_mask = combine_tensors[x_active_mask_index]
        if x_active_mask is not None and x_active_mask.dim() == 3:
            combine_tensors[x_active_mask_index] = x_active_mask.squeeze(1)
            tensor_shapes = list(combine_context.tensor_view_shapes)
            tensor_shapes[x_active_mask_index] = tuple(combine_tensors[x_active_mask_index].shape)
            combine_context.tensor_view_shapes = tuple(tensor_shapes)
            combine_context._tensor_list_dist = None
            combine_context._flat_tensor_view_shapes = None


    combine_context.tensors = tuple(combine_tensors)
    combine_context._flat_tensors = None
    combine_context._param_plan_cache = None
    combine_context._tensor_list_dist = None
    combine_context._flat_tensor_view_shapes = None
    combine_context._flat_tensor_storage_shapes = None
    combine_context._flat_tensor_view_strides = None
    if is_add_rms_norm:
        residual_x_index = 5
        expected_residual_x_shape = (
            int(combine_tensors[1].shape[0]),
            1,
            int(combine_tensors[6].numel()),
        )
        expected_residual_x = combine_tensors[residual_x_index]
        if (
            expected_residual_x is None
            or tuple(expected_residual_x.shape) != expected_residual_x_shape
        ):
            residual_dtype = (
                expected_residual_x.dtype
                if expected_residual_x is not None
                else combine_tensors[0].dtype
            )
            torch.manual_seed(42)
            expected_residual_x = torch.rand(
                expected_residual_x_shape, dtype=residual_dtype
            )
            combine_tensors[residual_x_index] = expected_residual_x
            tensor_shapes = list(combine_context.tensor_view_shapes)
            tensor_shapes[residual_x_index] = expected_residual_x_shape
            combine_context.tensor_view_shapes = tuple(tensor_shapes)
            combine_context._flat_tensors = None
            combine_context._param_plan_cache = None
            combine_context._tensor_list_dist = None
            combine_context._flat_tensor_view_shapes = None
            combine_context._flat_tensor_storage_shapes = None
            combine_context._flat_tensor_view_strides = None
    combine_context._flat_tensor_view_offsets = None
    combine_context._flat_input_data_ranges = None
    combine_context._flat_precision_tolerances = None
    combine_context._flat_absolute_precision = None
    combine_context._flat_tensor_dtypes = None
    if elastic_info_index >= 0 and combine_tensors[elastic_info_index] is not None:
        combine_context.attributes["isElastic"] = 1
    diagnostic_fields = {
        "expand_x": (combine_tensors[expand_x_index], generated["expand_x"]),
        "expert_ids": (combine_tensors[1], generated["expert_ids"]),
        "assist_info": (combine_tensors[assist_info_index], generated["expand_idx"]),
        "expert_scales": (combine_tensors[4], generated["expert_scales"]),
    }
    if not is_deep_ep and dispatch_expand_x is not None:
        golden_expand_x = generated["expand_x"].reshape(-1)
        actual_expand_x = combine_tensors[expand_x_index].reshape(-1)
        dispatch_output_expand_x = decode_output(dispatch_expand_x_index).reshape(-1)
        valid_count = min(golden_expand_x.numel(), actual_expand_x.numel())
        logging.info(
            "Moe cascade expand_x source diagnostics rank=%s use_generated=%s "
            "golden=%s actual=%s dispatch_output=%s",
            rank_idx,
            use_generated_expand_x,
            golden_expand_x[: min(6, valid_count)].tolist(),
            actual_expand_x[: min(6, valid_count)].tolist(),
            dispatch_output_expand_x[: min(6, valid_count)].tolist(),
        )
    diagnostic_messages = []
    for field_name, (actual, expected) in diagnostic_fields.items():
        if actual is None or expected is None:
            diagnostic_messages.append(f"{field_name}=missing(actual={actual is not None},expected={expected is not None})")
            continue
        valid_count = min(actual.reshape(-1).numel(), expected.reshape(-1).numel())
        prefix_equal = torch.equal(
            actual.reshape(-1)[:valid_count].to(expected.dtype), expected.reshape(-1)[:valid_count]
        )
        diagnostic_messages.append(
            f"{field_name}(shape={tuple(actual.shape)},expected={tuple(expected.shape)},prefix_equal={prefix_equal})"
        )
    def copy_generated(target, source):
        if target is None or source is None:
            return
        flattened = source.reshape(-1)[: target.numel()].to(target.dtype)
        if flattened.numel() < target.numel():
            flattened = torch.cat(
                [flattened, torch.zeros(target.numel() - flattened.numel(), dtype=flattened.dtype)]
            )
        target.copy_(flattened.reshape(target.shape))

    if ori_x_index >= 0 and combine_tensors[ori_x_index] is not None:
        copy_generated(combine_tensors[ori_x_index], generated["ori_x"])
    if combine_tensors[x_active_mask_index] is not None:
        copy_generated(
            combine_tensors[x_active_mask_index],
            generated.get("x_active_mask_input", generated["x_active_mask"]),
        )
    if elastic_info_index >= 0 and combine_tensors[elastic_info_index] is not None:
        combine_tensors[elastic_info_index] = generated["elastic_info"].reshape(
            -1
        ).clone()
        tensor_shapes = list(combine_context.tensor_view_shapes)
        tensor_shapes[elastic_info_index] = (
            combine_tensors[elastic_info_index].shape
        )
        combine_context.tensor_view_shapes = tuple(tensor_shapes)
    if elastic_info_index >= 0 and combine_tensors[elastic_info_index] is not None:
        expand_scales_tensor = combine_tensors[expand_scales_index]
        mask_prefix = (
            combine_tensors[x_active_mask_index].reshape(-1)[:8].tolist()
            if combine_tensors[x_active_mask_index] is not None
            else None
        )
        ori_x_prefix = (
            combine_tensors[ori_x_index].reshape(-1)[:6].tolist()
            if ori_x_index >= 0 and combine_tensors[ori_x_index] is not None
            else None
        )
        logging.info(
            "Moe cascade combine inputs rank=%s expand_x=%s..%s mask=%s ori_x=%s expand_scales=%s "
            "ep_counts=%s..%s elastic_info=%s",
            rank_idx,
            combine_tensors[expand_x_index].reshape(-1)[:6].tolist()
            if combine_tensors[expand_x_index] is not None
            else None,
            combine_tensors[expand_x_index].reshape(-1)[-6:].tolist()
            if combine_tensors[expand_x_index] is not None
            else None,
            mask_prefix,
            ori_x_prefix,
            expand_scales_tensor.reshape(-1).tolist()
            if expand_scales_tensor is not None
            else None,
            combine_tensors[ep_send_counts_index].reshape(-1)[:6].tolist(),
            combine_tensors[ep_send_counts_index].reshape(-1)[-6:].tolist(),
            combine_tensors[elastic_info_index].reshape(-1).tolist(),
        )
    for tensor_index in (
        const_expert_alpha_1_index,
        const_expert_alpha_2_index,
        const_expert_v_index,
    ):
        if tensor_index >= 0 and combine_tensors[tensor_index] is not None:
            copy_generated(
                combine_tensors[tensor_index],
                generated[
                    {
                        const_expert_alpha_1_index: "const_expert_alpha_1",
                        const_expert_alpha_2_index: "const_expert_alpha_2",
                        const_expert_v_index: "const_expert_v",
                    }[tensor_index]
                ],
            )

    if is_add_rms_norm:
        residual_x_index = 5
        expected_shape = (
            int(combine_tensors[1].shape[0]),
            1,
            int(combine_tensors[6].numel()),
        )
        residual_x = generated["residual_x"]
        if bool(int(combine_context.attributes.get("_moeVariableBs", 0) or 0)):
            row_mask = combine_tensors[x_active_mask_index].reshape(-1).to(torch.bool)
            residual_x = residual_x[row_mask]
        if residual_x.numel() != int(np.prod(expected_shape)):
            raise ValueError(
                f"Moe AddRmsNorm residual input size mismatch on rank {rank_idx}: "
                f"got {tuple(residual_x.shape)}, expected {expected_shape}"
            )
        combine_tensors[residual_x_index] = residual_x.reshape(expected_shape)
        tensor_shapes = list(combine_context.tensor_view_shapes)
        tensor_shapes[residual_x_index] = tuple(combine_tensors[residual_x_index].shape)
        combine_context.tensor_view_shapes = tuple(tensor_shapes)
        combine_context._tensor_list_dist = None
        combine_context._flat_tensor_view_shapes = None
        x_active_mask = combine_tensors[x_active_mask_index]
        if x_active_mask is not None and x_active_mask.dim() == 3:
            combine_tensors[x_active_mask_index] = x_active_mask.squeeze(1)
            tensor_shapes = list(combine_context.tensor_view_shapes)
            tensor_shapes[x_active_mask_index] = tuple(combine_tensors[x_active_mask_index].shape)
            combine_context.tensor_view_shapes = tuple(tensor_shapes)
            combine_context._tensor_list_dist = None
            combine_context._flat_tensor_view_shapes = None


    logging.info(
        "Moe cascade combine input diagnostics rank=%s %s ep_counts=%s tp_counts=%s",
        rank_idx,
        " ".join(diagnostic_messages),
        combine_tensors[ep_send_counts_index].reshape(-1).tolist(),
        combine_tensors[tp_send_counts_index].reshape(-1).tolist()
        if combine_tensors[tp_send_counts_index] is not None
        else None,
    )
    logging.info(
        "Moe cascade dispatch inputs rank=%s expand_x=%s assist_info=%s ep_counts=%s tp_counts=%s",
        rank_idx,
        tuple(dispatch_expand_x.shape) if dispatch_expand_x is not None else None,
        dispatch_assist_info[: min(18, dispatch_assist_info.numel())].tolist(),
        dispatch_ep_send_counts.tolist(),
        dispatch_tp_send_counts.tolist() if dispatch_tp_send_counts is not None else None,
    )
    return combine_tensors, generated, dispatch_expand_x, use_generated_expand_x, dispatch_assist_info, dispatch_ep_send_counts, dispatch_tp_send_counts


def _replace_generated_combine_inputs(
    combine_context,
    combine_tensors,
    rank_idx,
    dispatch_context,
    dispatch_tensor_names,
    dispatch_expand_x_index,
    dispatch_assist_index,
    dispatch_ep_counts_index,
    expand_x_index,
    assist_info_index,
    ep_send_counts_index,
    generated,
    is_deep_ep,
    is_add_rms_norm,
    x_active_mask_index,
    ori_x_index,
    elastic_info_index,
    const_expert_alpha_1_index,
    const_expert_alpha_2_index,
    const_expert_v_index,
    decode_output,
):
    from ttk.utilities.dtypes import torch_to_numpy_tensor
    import logging

    dispatch_tp_output_index = (
        dispatch_tensor_names.index("tpRecvCountsOut")
        if "tpRecvCountsOut" in dispatch_tensor_names
        else -1
    )
    tp_send_counts_index = 7 if is_add_rms_norm else 5
    expand_scales_index = 12 if is_add_rms_norm else 10
    if combine_tensors[1] is not None:
        combine_tensors[1] = generated["expert_ids"].to(combine_tensors[1].dtype).clone()
    if combine_tensors[4] is not None:
        combine_tensors[4] = generated["expert_scales"].to(combine_tensors[4].dtype).clone()
    if is_deep_ep and int(combine_context.attributes.get("isExpand", 0)) == 1:
        combine_tensors[1] = generated["expert_ids_expand"].clone()
    if dispatch_context is None:
        dispatch_assist_info = generated["expand_idx"].reshape(-1)
    elif is_deep_ep:
        dispatch_assist_info = generated["expand_idx"].reshape(-1)
    else:
        dispatch_assist_info = decode_output(dispatch_assist_index).reshape(-1)
    target_assist_length = combine_tensors[assist_info_index].numel()
    if dispatch_assist_info.numel() != target_assist_length:
        if dispatch_assist_info.numel() > target_assist_length:
            dispatch_assist_info = dispatch_assist_info[:target_assist_length]
    if dispatch_assist_info.numel() < target_assist_length:
        padded_assist_info = torch.zeros(
            target_assist_length,
            dtype=dispatch_assist_info.dtype,
            device=dispatch_assist_info.device,
        )
        padded_assist_info[: dispatch_assist_info.numel()].copy_(dispatch_assist_info)
        dispatch_assist_info = padded_assist_info
    if dispatch_context is None:
        dispatch_ep_send_counts = generated["ep_recv_counts"]
    elif is_deep_ep:
        dispatch_ep_send_counts = generated["ep_recv_counts"]
    else:
        dispatch_ep_send_counts = decode_output(dispatch_ep_counts_index)
    target_ep_count_size = combine_tensors[ep_send_counts_index].numel()
    if dispatch_ep_send_counts.numel() < target_ep_count_size:
        dispatch_ep_send_counts = torch.cat(
            [
                dispatch_ep_send_counts,
                torch.full(
                    (target_ep_count_size - dispatch_ep_send_counts.numel(),),
                    int(dispatch_ep_send_counts[-1]) if dispatch_ep_send_counts.numel() else 0,
                    dtype=dispatch_ep_send_counts.dtype,
                ),
            ]
        )
    dispatch_tp_send_counts = None
    if (
        dispatch_context is not None
        and dispatch_tp_output_index in dispatch_context.output_tensor_indexes
    ):
        dispatch_tp_send_counts = decode_output(dispatch_tp_output_index)
    if dispatch_tp_send_counts is not None and combine_tensors[tp_send_counts_index] is not None:
        combine_tensors[tp_send_counts_index] = dispatch_tp_send_counts.to(
            combine_tensors[tp_send_counts_index].dtype
        ).clone()
    combine_tensors[assist_info_index] = dispatch_assist_info.reshape(-1).to(
        combine_tensors[assist_info_index].dtype
    )
    if dispatch_ep_send_counts.numel() == combine_tensors[ep_send_counts_index].numel():
        combine_tensors[ep_send_counts_index].copy_(dispatch_ep_send_counts.reshape(-1))
    else:
        combine_tensors[ep_send_counts_index] = dispatch_ep_send_counts.to(
            combine_tensors[ep_send_counts_index].dtype
        ).clone()
        tensor_shapes = list(combine_context.tensor_view_shapes)
        tensor_shapes[ep_send_counts_index] = tuple(dispatch_ep_send_counts.shape)
        combine_context.tensor_view_shapes = tuple(tensor_shapes)
        combine_context._tensor_list_dist = None
        combine_context._flat_tensor_view_shapes = None

    def copy_generated(target, source):
        if target is None or source is None:
            return
        flattened = source.reshape(-1)[: target.numel()].to(target.dtype)
        if flattened.numel() < target.numel():
            flattened = torch.cat(
                [flattened, torch.zeros(target.numel() - flattened.numel(), dtype=flattened.dtype)]
            )
        target.copy_(flattened.reshape(target.shape))

    if ori_x_index >= 0 and combine_tensors[ori_x_index] is not None:
        copy_generated(combine_tensors[ori_x_index], generated["ori_x"])
    if combine_tensors[x_active_mask_index] is not None:
        copy_generated(
            combine_tensors[x_active_mask_index],
            generated.get("x_active_mask_input", generated["x_active_mask"]),
        )
    if elastic_info_index >= 0:
        generated_elastic_info = generated["elastic_info"].reshape(-1)
        if combine_tensors[elastic_info_index] is None:
            combine_tensors[elastic_info_index] = generated_elastic_info.clone()
            tensor_shapes = list(combine_context.tensor_view_shapes)
            tensor_shapes[elastic_info_index] = tuple(generated_elastic_info.shape)
            combine_context.tensor_view_shapes = tuple(tensor_shapes)
            combine_context._flat_tensors = None
            combine_context._param_plan_cache = None
            combine_context._tensor_list_dist = None
            combine_context._flat_tensor_view_shapes = None
            combine_context._flat_tensor_storage_shapes = None
        else:
            copy_generated(
                combine_tensors[elastic_info_index], generated_elastic_info
            )
    if is_add_rms_norm:
        residual_x_index = 5
        expected_shape = (
            int(combine_tensors[1].shape[0]),
            1,
            int(combine_tensors[6].numel()),
        )
        residual_x = generated["residual_x"]
        if residual_x.numel() != int(np.prod(expected_shape)):
            raise ValueError(
                f"Moe AddRmsNorm residual input size mismatch on rank {rank_idx}: "
                f"got {tuple(residual_x.shape)}, expected {expected_shape}"
            )
        combine_tensors[residual_x_index] = residual_x.reshape(expected_shape)
        tensor_shapes = list(combine_context.tensor_view_shapes)
        tensor_shapes[residual_x_index] = tuple(combine_tensors[residual_x_index].shape)
        combine_context.tensor_view_shapes = tuple(tensor_shapes)
        combine_context._tensor_list_dist = None
        combine_context._flat_tensor_view_shapes = None
    for tensor_index in (
        const_expert_alpha_1_index,
        const_expert_alpha_2_index,
        const_expert_v_index,
    ):
        if tensor_index >= 0 and combine_tensors[tensor_index] is not None:
            copy_generated(
                combine_tensors[tensor_index],
                generated[
                    {
                        const_expert_alpha_1_index: "const_expert_alpha_1",
                        const_expert_alpha_2_index: "const_expert_alpha_2",
                        const_expert_v_index: "const_expert_v",
                    }[tensor_index]
                ],
            )

    combine_context.tensors = tuple(combine_tensors)
    combine_context._flat_tensors = None
    combine_context._param_plan_cache = None
    combine_context._tensor_list_dist = None
    combine_context._flat_tensor_view_shapes = None
    combine_context._flat_tensor_storage_shapes = None
    combine_context._flat_tensor_view_strides = None
    combine_context._flat_tensor_view_offsets = None
    combine_context._flat_input_data_ranges = None
    combine_context._flat_tensor_dtypes = None
    combine_context.np_storages = [
        torch_to_numpy_tensor(tensor.reshape(-1).detach().cpu())
        if tensor is not None and hasattr(tensor, "detach")
        else None
        for tensor in combine_tensors
    ]


def _run_moe_dispatch_cascade(thread_contexts, device_ids, main_device, stream_value, plugin_path):
    from ttk.core_modules.npu.op_api.profiling import AclOpExecutor
    from ttk.core_modules.npu.op_api.profiling import AclInterface
    from ttk.core_modules.npu.op_api.input_generation import InputGenerator
    from ttk.core_modules.runtime.rts_interface import RTSInterface
    from ttk.utilities import get_global_storage
    from ttk.utilities import set_global_storage
    from ttk.utilities.classes import SWITCHES

    rank_contexts = {}
    _run_moe_dispatch_cascade._rank_contexts = rank_contexts
    _run_moe_dispatch_cascade._thread_contexts = thread_contexts
    _run_moe_dispatch_cascade._dispatch_contexts = {}
    _run_moe_dispatch_cascade._initial_cases = {}
    is_add_rms_norm_e2e = next(iter(thread_contexts.values())).api_name == (
        "aclnnMoeDistributeCombineAddRmsNormV2"
    )
    context_by_device = getattr(_run_moe_dispatch_cascade, "_contexts", None) or {}
    if not context_by_device:
        import ctypes

        for device_id in device_ids:
            context_by_device[device_id] = getattr(thread_contexts[device_id], "_cascade_context", None)
        if all(context_by_device.values()):
            _run_moe_dispatch_cascade._contexts = context_by_device
    context_by_device = getattr(_run_moe_dispatch_cascade, "_contexts", None) or {}
    if not context_by_device:
        import ctypes

        for device_id in device_ids:
            context_by_device[device_id] = getattr(thread_contexts[device_id], "_cascade_context", None)
        if all(context_by_device.values()):
            _run_moe_dispatch_cascade._contexts = context_by_device
    is_deepep = int(next(iter(thread_contexts.values())).attributes.get("_moeDeepep", 0) or 0)
    use_external_process = False
    if not is_deepep:
        from ttk.config.loader import get_cascade_config

        use_external_process = bool(get_cascade_config().external_process)
    if use_external_process and not is_add_rms_norm_e2e:
        _execute_external_cascade(thread_contexts, device_ids, fill_results=False)
        return
    if is_add_rms_norm_e2e:
        for rank, device_id in enumerate(device_ids):
            combine_context = thread_contexts[device_id]
            combine_context.attributes["_moeCascadeGoldenExpandX"] = int(
                bool(get_cascade_config().use_golden_expand_x)
            )
            _replace_generated_combine_inputs(
                combine_context,
                list(combine_context.flatten_tensors),
                rank,
                None,
                [],
                -1,
                -1,
                -1,
                0,
                2,
                3,
                _padded_moe_case_data_for_context(combine_context, rank),
                False,
                True,
                8,
                15,
                14,
                16,
                17,
                18,
                None,
            )
    else:
        for rank, device_id in enumerate(device_ids):
            combine_context = thread_contexts[device_id]
            current_storage = get_global_storage()
            original_plugin_path = getattr(current_storage, "plugin_path", None)
            try:
                if not original_plugin_path:
                    set_global_storage(SWITCHES())
                if not getattr(get_global_storage(), "plugin_path", None):
                    get_global_storage().plugin_path = (plugin_path,)
                dispatch_context = _make_dispatch_context(
                    combine_context, plugin_path, rank
                )
                _run_moe_dispatch_cascade._dispatch_contexts[device_id] = dispatch_context
                if is_deepep:
                    dispatch_inputs_path = (
                        Path(__file__).resolve().parents[4]
                        / "moe_distribute_dispatch"
                        / "tests"
                        / "assets"
                        / "impl"
                        / "inputs.py"
                    )
                    dispatch_inputs_spec = importlib.util.spec_from_file_location(
                        "moe_distribute_dispatch_inputs_for_cascade", dispatch_inputs_path
                    )
                    dispatch_inputs_module = importlib.util.module_from_spec(
                        dispatch_inputs_spec
                    )
                    dispatch_inputs_spec.loader.exec_module(dispatch_inputs_module)
                    input_tensor_args = (
                        torch.empty(
                            (
                                int(dispatch_context.attributes["globalBs"])
                                // int(dispatch_context.attributes["epWorldSize"]),
                                int(combine_context.tensor_view_shapes[0][1]),
                            ),
                            dtype=torch.bfloat16,
                        ),
                        torch.zeros((int(dispatch_context.attributes["globalBs"]) // int(dispatch_context.attributes["epWorldSize"]), 1), dtype=torch.int32),
                    )
                    original_top_k = int(combine_context.tensor_view_shapes[1][1])
                    input_tensor_args = (
                        input_tensor_args[0],
                        torch.zeros(
                            (
                                int(dispatch_context.attributes["globalBs"])
                                // int(dispatch_context.attributes["epWorldSize"]),
                                original_top_k,
                            ),
                            dtype=torch.int32,
                        ),
                    )
                    input_tensors = dispatch_inputs_module.moe_distribute_dispatch_inputs(
                        *input_tensor_args,
                        **dispatch_context.attributes,
                        remark=dispatch_context.remark,
                        api_name=dispatch_context.api_name,
                        _ttkPatchedEpRankId=rank,
                        _moeDeepepOriginalTopK=original_top_k,
                        _ttk_context=dispatch_context,
                    )
                    generated_initial_case = _moe_initial_case_data_for_context(
                        combine_context, rank
                    )
                    _run_moe_dispatch_cascade._initial_cases[device_id] = generated_initial_case
                    logging.info(
                        "Moe DeepEP initial case rank=%s available=%s shape=%s",
                        rank,
                        generated_initial_case is not None,
                        tuple(generated_initial_case["initial_expand_x"].shape)
                        if generated_initial_case is not None
                        and generated_initial_case.get("initial_expand_x") is not None
                        else None,
                    )
                    dispatch_context._flat_tensors = None
                    dispatch_context.np_storages = None
                    dispatch_context.tensors = input_tensors
                else:
                    InputGenerator(dispatch_context).gen()
            finally:
                if not original_plugin_path:
                    set_global_storage(current_storage)
                else:
                    current_storage.plugin_path = original_plugin_path
            rank_contexts[device_id] = dispatch_context
            if is_deepep:
                logging.info(
                    "Moe cascade dispatch prepared dev=%s shapes=%s",
                    device_id,
                    tuple(tuple(tensor.shape) if tensor is not None else None for tensor in dispatch_context.flatten_tensors),
                )
                continue
            thread_device = AclInterface.__new__(AclInterface)
            thread_device.__dict__ = main_device.__dict__.copy()
            thread_device._acl_inited = True
            thread_device._device_id = device_id
            thread_device._acl_tensor_lists = {}
            thread_device._acl_tensor_to_device_mem = {}
            thread_device._acl_tensors = set()
            thread_device._acl_scalars = set()
            thread_device._acl_int_arrays = set()
            thread_device._acl_float_arrays = set()
            thread_device._acl_bool_arrays = set()
            thread_device._acl_device_memories = set()
            thread_device._suppress_finalize = True
            thread_device._suppress_reset = True
            thread_rts = main_device._rts_interface.__dict__.copy()
            thread_rts["memory_manager"] = {}
            thread_rts["_device_id"] = device_id
            thread_rts["_acl_inited"] = True
            context_value = context_by_device.get(device_id)
            if context_value is None:
                raise RuntimeError(f"Moe cascade context is missing for device {device_id}")
            context = ctypes.c_void_p(context_value)
            thread_rts["context"] = context
            thread_rts["context_storage"] = [context]
            thread_device._rts_interface = RTSInterface.__new__(RTSInterface)
            thread_device._rts_interface.__dict__ = thread_rts
            rank_contexts[(device_id, "device")] = thread_device
            logging.info(
                "Moe cascade dispatch prepared dev=%s shapes=%s",
                device_id,
                tuple(tuple(tensor.shape) if tensor is not None else None for tensor in dispatch_context.flatten_tensors),
            )
            if dispatch_context.api_name == "aclnnMoeDistributeDispatchV2" and dispatch_context.flatten_tensors[2] is not None:
                logging.info(
                    "Moe cascade dispatch scales dev=%s shape=%s prefix=%s",
                    device_id,
                    tuple(dispatch_context.flatten_tensors[2].shape),
                    dispatch_context.flatten_tensors[2].reshape(-1)[:8].tolist(),
                )

    import threading

    if is_add_rms_norm_e2e:
        _execute_moe_add_rms_norm_e2e._thread_contexts = thread_contexts
        _execute_moe_add_rms_norm_e2e._device_ids = device_ids
        _execute_moe_add_rms_norm_e2e._rank_contexts = rank_contexts
        _execute_moe_add_rms_norm_e2e._ready_contexts = set()
        _execute_moe_add_rms_norm_e2e._completed_event = threading.Event()
        return

    if is_deepep:
        _run_moe_dispatch_cascade._deepep_cascade_payloads = (
            _run_deepep_cascade_external_worker(thread_contexts, device_ids)
        )
        _fill_external_results(
            thread_contexts,
            device_ids,
            _run_moe_dispatch_cascade._deepep_cascade_payloads,
        )
        return

    barrier_size = len(device_ids)
    barrier = threading.Barrier(barrier_size, timeout=1800)
    errors = {}
    is_deepep = int(next(iter(thread_contexts.values())).attributes.get("_moeDeepep", 0) or 0)

    def execute_dispatch(device_id, rank_idx):
        try:
            barrier.wait()
            from ttk.utilities import set_global_storage
            from ttk.utilities.classes import SWITCHES

            if not getattr(get_global_storage(), "root_path", None):
                set_global_storage(SWITCHES())
            if not getattr(get_global_storage(), "root_path", None):
                get_global_storage().root_path = os.getcwd()
            dispatch_context = rank_contexts[device_id]
            if not is_deepep:
                thread_device = rank_contexts[(device_id, "device")]
                thread_device._rts_interface.set_context(
                    thread_device._rts_interface.context_storage[0]
                )
            if int(dispatch_context.attributes.get("_moeElasticSuppressed", 0)):
                dispatch_context.prof_result = ApiProfilingResult.fail("SUPPRESSED")
                _replace_combine_inputs(
                    thread_contexts[device_id],
                    dispatch_context,
                    device_id,
                    rank_idx,
                )
                return
            else:
                if dispatch_context.prof_result is None:
                    if int(dispatch_context.attributes.get("_moeDeepep", 0) or 0):
                        dispatch_context.prof_result = _load_deepep_dispatch_payload(
                            thread_contexts, device_ids, dispatch_context, device_id, rank_idx
                        )
                if dispatch_context.prof_result is None:
                    dispatch_context.prof_result = AclOpExecutor(
                        dispatch_context, rank_contexts[(device_id, "device")]
                    ).do(stream=stream_value[device_id], skip_context_creation=True)
            if dispatch_context.prof_result is None or dispatch_context.prof_result.failed():
                raise RuntimeError(f"Moe cascade dispatch failed on device {device_id}")
            _replace_combine_inputs(
                thread_contexts[device_id],
                dispatch_context,
                device_id,
                rank_idx,
            )
        except BaseException:
            logging.exception("Moe cascade dispatch failed on device %s", device_id)
            errors[device_id] = True

    threads = [
        threading.Thread(
            target=execute_dispatch,
            args=(device_id, rank_idx),
            name=f"moe-dispatch-{device_id}",
        )
        for rank_idx, device_id in enumerate(device_ids)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError(f"Moe cascade dispatch failed on devices {sorted(errors)}")

    from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor

    for device_id in device_ids:
        combine_context = thread_contexts[device_id]
        combine_context.np_storages = [
            torch_to_numpy_tensor(tensor.reshape(-1).detach().cpu())
            if tensor is not None and hasattr(tensor, "detach")
            else None
            for tensor in combine_context.flatten_tensors
        ]


def is_deepep_dispatch_prepared(thread_contexts, device_ids):
    first_context = next(iter(thread_contexts.values()))
    if int(first_context.attributes.get("_moeDeepep", 0) or 0) != 1:
        return False
    rank_contexts = getattr(_run_moe_dispatch_cascade, "_rank_contexts", None) or {}
    return all(device_id in rank_contexts for device_id in device_ids)


def prepare_moe_dispatch_cascade(thread_contexts, device_ids):
    first_context = next(iter(thread_contexts.values()))
    if int(first_context.attributes.get("_moeDeepep", 0) or 0) != 1:
        return
    rank_contexts = getattr(_run_moe_dispatch_cascade, "_rank_contexts", None) or {}


    for rank_idx, device_id in enumerate(device_ids):
        if device_id not in rank_contexts:
            raise RuntimeError(
                f"Moe DeepEP cascade dispatch context is missing on device {device_id}"
            )
        dispatch_context = rank_contexts[device_id]
        if int(dispatch_context.attributes.get("_moeElasticSuppressed", 0)):
            dispatch_context.prof_result = ApiProfilingResult.fail("SUPPRESSED")
        else:
            dispatch_context.prof_result = _load_deepep_dispatch_payload(
                thread_contexts,
                device_ids,
                dispatch_context,
                device_id,
                rank_idx,
            )
        if dispatch_context.prof_result is None or dispatch_context.prof_result.failed():
            raise RuntimeError(f"Moe DeepEP cascade dispatch failed on device {device_id}")
        _replace_combine_inputs(
            thread_contexts[device_id], dispatch_context, device_id, rank_idx
        )

    for device_id in device_ids:
        combine_context = thread_contexts[device_id]
        combine_context.np_storages = [
            torch_to_numpy_tensor(tensor.reshape(-1).detach().cpu())
            if tensor is not None and hasattr(tensor, "detach")
            else None
            for tensor in combine_context.flatten_tensors
        ]


def _run_deepep_cascade_external_worker(thread_contexts, device_ids):
    import base64
    import json
    import socket
    import subprocess

    from cann_ops_transformer.ops import MoeDistributeBuffer
    from ttk.config.loader import get_cascade_config, get_moe_config

    _run_moe_dispatch_cascade._initial_cases = getattr(
        _run_moe_dispatch_cascade, "_initial_cases", {}
    )
    _run_moe_dispatch_cascade._dispatch_contexts = getattr(
        _run_moe_dispatch_cascade, "_dispatch_contexts", {}
    )

    cache_root = Path(get_moe_config().cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    _keep_debug_run_dir = False
    _release_main_resources(thread_contexts, device_ids)
    run_dir_obj = tempfile.TemporaryDirectory(
        prefix="ttk_moe_deepep_cascade_", dir=str(cache_root)
    )
    try:
        run_dir = run_dir_obj.name
        with socket.socket(socket.AF_INET6) as sock:
            sock.bind(("::1", 0))
            endpoint = f"tcp://[::1]:{sock.getsockname()[1]}"

        for rank_idx, device_id in enumerate(device_ids):
            context = thread_contexts[device_id]
            available_ranks = _case_available_ranks(context)
            context.attributes["_moeElasticAvailableRanks"] = available_ranks
            context.attributes["_moeElasticSuppressed"] = int(
                rank_idx not in set(available_ranks)
            )
            generated = _run_moe_dispatch_cascade._initial_cases.get(device_id)
            if generated is not None and generated.get("initial_expand_x") is None:
                logging.warning(
                    "Moe DeepEP initial case rank=%s available=True but initial_expand_x is None",
                    rank_idx,
                )
            if generated is None:
                generated = context.attributes.get("_moeCaseData")
            if (
                generated is None
                and int(context.attributes.get("isExpand", 0)) == 1
            ):
                generated = _moe_initial_case_data_for_context(context, rank_idx)
            if generated is None:
                generated = _moe_case_data_for_context(context, rank_idx)

            def plan_array(tensor):
                if tensor is None:
                    return None
                value = tensor.detach().cpu().contiguous()
                if value.dtype == torch.bfloat16:
                    value = value.view(torch.uint16)
                return value.numpy()

            global_bs = int(context.attributes.get("globalBs", 1))
            ep_world_size = int(
                context.attributes.get("epWorldSize", len(device_ids))
            )
            per_rank_bs = max(global_bs // ep_world_size, 1)
            x = generated["x"]
            expert_ids = generated["expert_ids"]
            expert_scales = generated["expert_scales"]
            if int(context.attributes.get("xActiveMaskDim", 0)) <= 0:
                x_active_mask = None
            elif bool(context.attributes.get("xActiveMaskOptional", 0)):
                x_active_mask = generated.get(
                    "x_active_mask_input", generated["x_active_mask"]
                )
            elif int(context.attributes.get("_moeVariableBs", 0)):
                x_active_mask = generated["x_active_mask_input"]
            else:
                x_active_mask = generated["x_active_mask"]
            if x_active_mask is not None and int(
                context.attributes.get("_moeVariableBs", 0)
            ) == 0:
                x_active_mask = x_active_mask.clone()
            if int(context.attributes.get("xActiveMaskDim", 0)) == 0:
                x_active_mask_for_plan = None
            else:
                x_active_mask_for_plan = x_active_mask
            x_active_mask_for_input = x_active_mask_for_plan
            if int(context.attributes.get("_moeVariableBs", 0)):
                row_mask = generated["x_active_mask"][:per_rank_bs]
            else:
                row_mask = torch.ones(per_rank_bs, dtype=torch.bool)
            x = x[:per_rank_bs]
            expert_ids = expert_ids[:per_rank_bs]
            expert_scales = expert_scales[:per_rank_bs]
            expert_ids_expand = None
            if generated.get("expert_ids_expand") is not None:
                expert_ids_expand = generated["expert_ids_expand"][:per_rank_bs]
            ori_x = generated["ori_x"][:per_rank_bs]
            if int(context.attributes.get("_moeVariableBs", 0)):
                x = x[row_mask]
                expert_ids = expert_ids[row_mask]
                expert_scales = expert_scales[row_mask]
                if expert_ids_expand is not None:
                    expert_ids_expand = expert_ids_expand[row_mask]
                ori_x = ori_x[row_mask]
            shared_expert_num = int(context.attributes.get("sharedExpertNum", 0))
            shared_expert_rank_num = int(
                context.attributes.get("sharedExpertRankNum", 0)
            )
            combine_shared_expert_input = (
                int(context.attributes.get("isSharedExpertX", 0)) == 1
            )
            shared_expert_x = (
                generated["shared_expert_x"]
                if combine_shared_expert_input
                and shared_expert_num > 0
                and shared_expert_rank_num > 0
                else None
            )
            ori_x = generated["ori_x"][:per_rank_bs]
            if int(context.attributes.get("_moeVariableBs", 0)):
                ori_x = ori_x[row_mask]
            const_expert_alpha_1 = generated["const_expert_alpha_1"]
            const_expert_alpha_2 = generated["const_expert_alpha_2"]
            const_expert_v = generated["const_expert_v"]
            elastic_info = generated["elastic_info"]
            use_golden_expand_x = bool(get_cascade_config().use_golden_expand_x)
            golden_expand_x = generated["expand_x"].to(x.dtype)
            initial_golden_expand_x = generated.get("initial_expand_x")
            if initial_golden_expand_x is not None:
                initial_golden_expand_x = initial_golden_expand_x.to(x.dtype)
            else:
                initial_golden_expand_x = golden_expand_x
            initial_golden_expand_idx = generated.get("initial_expand_idx")
            initial_golden_ep_recv_counts = generated.get("initial_ep_recv_counts")
            golden_expand_idx = generated["expand_idx"]
            golden_ep_recv_counts = generated["ep_recv_counts"]
            plan = {
                "outputPath": str(Path(run_dir) / f"output_rank_{rank_idx}.npz"),
                "x": "",
                "inputDtype": str(x.dtype).replace("torch.", ""),
                "xShape": list(x.shape),
                "expertIds": "",
                "expertIdsShape": list(expert_ids.shape),
                "expertScales": "",
                "expertScalesShape": list(expert_scales.shape),
                "xActiveMask": "",
                "xActiveMaskShape": (
                    list(x_active_mask_for_input.shape)
                    if x_active_mask_for_input is not None
                    else []
                ),
                "rowMask": "",
                "rowMaskShape": (
                    list(row_mask.shape) if row_mask is not None else []
                ),
                "sharedExpertX": "",
                "sharedExpertXShape": (
                    list(shared_expert_x.shape) if shared_expert_x is not None else []
                ),
                "oriX": "",
                "oriXShape": list(ori_x.shape),
                "constExpertAlpha1": "",
                "constExpertAlpha1Shape": list(const_expert_alpha_1.shape),
                "constExpertAlpha2": "",
                "constExpertAlpha2Shape": list(const_expert_alpha_2.shape),
                "constExpertV": "",
                "constExpertVShape": list(const_expert_v.shape),
                "goldenExpandX": "",
                "goldenExpandXShape": (
                    list(golden_expand_x.shape) if use_golden_expand_x else []
                ),
                "goldenExpandXDtype": str(golden_expand_x.dtype).replace("torch.", ""),
                "initialGoldenExpandX": "",
                "initialGoldenExpandXShape": (
                    list(initial_golden_expand_x.shape)
                    if use_golden_expand_x and int(context.attributes.get("isExpand", 0))
                    else []
                ),
                "initialGoldenExpandXDtype": str(
                    initial_golden_expand_x.dtype
                ).replace("torch.", ""),
                "initialGoldenExpandIdxShape": (
                    list(initial_golden_expand_idx.shape)
                    if initial_golden_expand_idx is not None
                    else []
                ),
                "initialGoldenEpRecvCountsShape": (
                    list(initial_golden_ep_recv_counts.shape)
                    if initial_golden_ep_recv_counts is not None
                    else []
                ),
                "goldenExpandIdxShape": (
                    list(golden_expand_idx.shape)
                    if golden_expand_idx is not None
                    else []
                ),
                "goldenEpRecvCountsShape": (
                    list(golden_ep_recv_counts.shape)
                    if golden_ep_recv_counts is not None
                    else []
                ),
                "isExpand": int(context.attributes.get("isExpand", 0)),
                "availableRanks": list(generated["available_ranks"]),
                "elasticInfo": elastic_info.detach().cpu().numpy().astype(np.int32).tolist(),
                "elasticInfoShape": list(elastic_info.shape),
                "expertIdsExpand": "",
                "expertIdsExpandShape": (
                    list(expert_ids_expand.shape)
                    if expert_ids_expand is not None
                    else []
                ),
                "useGoldenExpandX": use_golden_expand_x,
                "topK": int(expert_ids.shape[1]),
                "expandXOutputRows": int(context.flatten_tensors[0].shape[0]),
                "attributes": {
                    key: value
                    for key, value in context.attributes.items()
                    if key in {
                        "epWorldSize", "tpWorldSize", "moeExpertNum",
                        "expertShardType", "sharedExpertNum",
                        "sharedExpertRankNum", "globalBs", "commQuantMode",
                        "zeroExpertNum", "copyExpertNum", "constExpertNum",
                        "commAlg", "_dispatchCommAlg", "expertTokenNumsType",
                        "isElastic", "quantMode",
                        "shareBrokenCardNum", "moeBrokenCardNum",
                        "isCombineX",
                        "rankNumPerSharedExpert",
                        "localMoeExpertNum",
                        "_moeDeepep",
                        "aicNum",
                    }
                },
            }
            plan["attributes"]["isExpand"] = int(
                context.attributes.get("isExpand", 0)
            )
            plan["attributes"]["_moeElasticSuppressed"] = int(
                bool(context.attributes.get("_moeElasticSuppressed", 0))
            )
            plan["attributes"].update({
                "topK": int(context.flatten_tensors[1].shape[1]),
                "localMoeExpertNum": int(
                    context.attributes.get(
                        "localMoeExpertNum",
                        int(context.attributes["moeExpertNum"])
                        // max(
                            int(
                                context.attributes.get(
                                    "epWorldSize", len(device_ids)
                                )
                            )
                            - int(
                                context.attributes.get("sharedExpertRankNum", 0)
                            ),
                            1,
                        ),
                    )
                ),
                "rankNumPerSharedExpert": int(
                    context.attributes.get(
                        "rankNumPerSharedExpert",
                        int(context.attributes.get("sharedExpertRankNum", 0))
                        // int(context.attributes.get("sharedExpertNum", 0) or 1),
                    )
                ),
            })
            tensor_specs = {
                "x": x,
                "expertIds": expert_ids,
                "expertScales": expert_scales,
                "xActiveMask": x_active_mask_for_input,
                "rowMask": row_mask,
                "sharedExpertX": shared_expert_x,
                "oriX": ori_x,
                "constExpertAlpha1": const_expert_alpha_1,
                "constExpertAlpha2": const_expert_alpha_2,
                "constExpertV": const_expert_v,
                "expertIdsExpand": expert_ids_expand,
            }
            if use_golden_expand_x:
                tensor_specs.update({
                    "goldenExpandX": golden_expand_x,
                })
                if int(context.attributes.get("isExpand", 0)):
                    tensor_specs.update({
                        "initialGoldenExpandX": initial_golden_expand_x,
                        "initialGoldenExpandIdx": initial_golden_expand_idx,
                        "initialGoldenEpRecvCounts": initial_golden_ep_recv_counts,
                        "goldenExpandIdx": golden_expand_idx,
                        "goldenEpRecvCounts": golden_ep_recv_counts,
                    })
            metadata = dict(plan)
            for name in tensor_specs:
                metadata[name] = ""
            plan_path = Path(run_dir) / f"plan_rank_{rank_idx}.json"
            with plan_path.open("w", encoding="utf-8") as plan_file:
                json.dump(metadata, plan_file)
            np.savez(
                plan_path.with_suffix(".npz"),
                **{
                    name: array
                    for name, array in (
                        (name, plan_array(tensor))
                        for name, tensor in tensor_specs.items()
                    )
                    if array is not None
                },
            )

        env = os.environ.copy()
        hccl_port_range = get_cascade_config().hccl_port_range
        if hccl_port_range:
            env["HCCL_NPU_SOCKET_PORT_RANGE"] = hccl_port_range
        else:
            env.pop("HCCL_NPU_SOCKET_PORT_RANGE", None)
        env["HCCL_CONNECT_TIMEOUT"] = str(
            get_cascade_config().hccl_connect_timeout_s
        )
        env["MOE_DEEPEP_CASCADE_LAUNCH_TIMEOUT_S"] = str(
            get_cascade_config().hccl_timeout_s
        )
        env["HCCL_BUFFSIZE"] = str(
            MoeDistributeBuffer.get_low_latency_ccl_buffer_size(
                len(device_ids),
                max(int(context.attributes["globalBs"]) // len(device_ids), 1),
                int(context.flatten_tensors[0].shape[-1]),
                int(context.attributes["moeExpertNum"]),
                int(context.flatten_tensors[1].shape[1]),
                int(context.attributes.get("sharedExpertNum", 0)),
                int(context.attributes.get("sharedExpertRankNum", 0)),
                context.attributes.get("_dispatchCommAlg", ""),
            )
        )
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["MAX_JOBS"] = "1"
        worker_dir = Path(__file__).resolve().parent
        python_executable = (
            get_cascade_config().external_python
            if get_cascade_config().external_python
            else sys.executable
        )
        extension_root_name = "torch_extensions"
        if Path(python_executable).resolve() != Path(sys.executable).resolve():
            extension_root_name = f"torch_extensions_{Path(python_executable).stem}"
        env["TORCH_EXTENSIONS_DIR"] = str(cache_root / extension_root_name)
        env["PYTHONPATH"] = os.pathsep.join(
            path
            for path in (
                str(
                    Path(__file__).resolve().parents[5]
                    / "torch_extension"
                    / "build"
                    / "lib"
                ),
                env.get("PYTHONPATH", ""),
            )
            if path
        )
        launcher = subprocess.Popen(
            [
                python_executable,
                str(worker_dir / "deepep_cascade_launcher.py"),
                run_dir,
                endpoint,
                str(worker_dir / "deepep_cascade_worker.py"),
                str(len(device_ids)),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        timeout = int(get_cascade_config().hccl_timeout_s)
        try:
            output, _ = launcher.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _keep_debug_run_dir = True
            launcher.kill()
            output, _ = launcher.communicate()
            for rank_idx in range(len(device_ids)):
                output_path = Path(run_dir) / f"output_rank_{rank_idx}.npz"
                if output_path.exists():
                    output_path.unlink()
            details = []
            for rank_idx in range(len(device_ids)):
                log_path = Path(run_dir) / f"worker_rank_{rank_idx}.log"
                if log_path.is_file():
                    content = log_path.read_text(encoding="utf-8", errors="replace")
                    if content.strip():
                        details.append(f"rank={rank_idx}\n{content}")
            raise RuntimeError(
                f"Moe DeepEP cascade worker timed out after {timeout}s; "
                f"worker output:\n{output}\n" + "\n".join(details)
            ) from None
        if output:
            print(output, flush=True)
        if launcher.returncode != 0:
            _keep_debug_run_dir = True
            details = []
            for rank_idx in range(len(device_ids)):
                log_path = Path(run_dir) / f"worker_rank_{rank_idx}.log"
                if log_path.is_file():
                    content = log_path.read_text(encoding="utf-8", errors="replace")
                    if content.strip():
                        details.append(f"rank={rank_idx}\n{content}")
            raise RuntimeError(
                "Moe DeepEP cascade worker failed; "
                + "\n".join(details)
            )

        payloads = {}
        for rank_idx, device_id in enumerate(device_ids):
            output_path = Path(run_dir) / f"output_rank_{rank_idx}.npz"
            if not output_path.is_file() or output_path.stat().st_size == 0:
                payloads[device_id] = None
                continue
            with np.load(output_path, allow_pickle=False) as payload:
                payloads[device_id] = (
                    payload["output_0"].tobytes(),
                    tuple(payload["output_shapes"].tolist()),
                    int(payload["rank"]),
                )
        return payloads
    finally:
        if _keep_debug_run_dir:
            run_dir_obj._finalizer.detach()
        else:
            run_dir_obj.cleanup()


def _run_external_cascade(thread_contexts, device_ids):
    import base64
    import json
    import socket

    from ttk.config.loader import get_cascade_config, get_moe_config

    cache_root = Path(get_moe_config().cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    _keep_debug_run_dir = False
    run_dir_obj = tempfile.TemporaryDirectory(
        prefix="ttk_moe_external_cascade_", dir=str(cache_root)
    )
    try:
        run_dir = run_dir_obj.name
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            endpoint = f"tcp://127.0.0.1:{sock.getsockname()[1]}"

        for rank_idx, device_id in enumerate(device_ids):
            context = thread_contexts[device_id]
            generated = _moe_case_data_for_context(context, rank_idx)

            def encode(tensor):
                if tensor is None:
                    return ""
                value = tensor.detach().cpu().contiguous()
                if value.dtype == torch.bfloat16 or value.dtype == torch.float16:
                    value = value.view(torch.uint16)
                return base64.b64encode(value.numpy()).decode("ascii")

            attributes = {
                key: value
                for key, value in context.attributes.items()
                if key in {
                    "epWorldSize", "tpWorldSize", "moeExpertNum",
                    "expertShardType", "sharedExpertNum",
                    "sharedExpertRankNum", "globalBs", "commQuantMode",
                    "zeroExpertNum", "copyExpertNum", "constExpertNum",
                    "commAlg", "expertTokenNumsType", "isElastic",
                    "quantMode", "rankNumPerSharedExpert", "isExpand",
                }
            }
            attributes["hcclBufferSize"] = int(
                os.environ.get("HCCL_BUFFSIZE", "0") or 0
            )
            if "localMoeExpertNum" in context.attributes:
                attributes["localMoeExpertNum"] = int(
                    context.attributes["localMoeExpertNum"]
                )
            input_dtype = str(generated["x"].dtype).replace("torch.", "")
            golden_expand_x = generated["expand_x"]
            x_active_mask = generated.get(
                "x_active_mask_input", generated["x_active_mask"]
            )
            golden_expand_x_rows = x_active_mask.numel()
            if golden_expand_x.shape[0] < golden_expand_x_rows:
                golden_expand_x = torch.nn.functional.pad(
                    golden_expand_x,
                    (0, 0, 0, golden_expand_x_rows - golden_expand_x.shape[0]),
                )
            plan = {
                "deviceId": device_id,
                "outputPath": str(Path(run_dir) / f"output_rank_{rank_idx}.npz"),
                "availableRanks": list(generated["available_ranks"]),
                "inputDtype": input_dtype,
                "x": encode(generated["x"]),
                "xShape": list(generated["x"].shape),
                "expertIds": encode(generated["expert_ids"]),
                "expertIdsShape": list(generated["expert_ids"].shape),
                "expertScales": encode(generated["expert_scales"]),
                "expertScalesShape": list(generated["expert_scales"].shape),
                "useGoldenExpandX": bool(get_cascade_config().use_golden_expand_x),
                "goldenExpandX": (
                    encode(golden_expand_x)
                    if bool(get_cascade_config().use_golden_expand_x)
                    else ""
                ),
                "goldenExpandXShape": (
                    list(golden_expand_x.shape)
                    if bool(get_cascade_config().use_golden_expand_x)
                    else []
                ),
                "xActiveMask": encode(
                    generated.get("x_active_mask_input", generated["x_active_mask"])
                ),
                "xActiveMaskShape": list(
                    generated.get(
                        "x_active_mask_input", generated["x_active_mask"]
                    ).shape
                ),
                "scales": encode(
                    generated["scales"]
                ),
                "scalesShape": (
                    list(generated["scales"].shape)
                    if generated.get("scales") is not None
                    else []
                ),
                "elasticInfo": encode(generated.get("elastic_info")),
                "elasticInfoShape": (
                    list(generated["elastic_info"].shape)
                    if generated.get("elastic_info") is not None
                    else []
                ),
                "sharedExpertX": encode(generated.get("shared_expert_x")),
                "sharedExpertXShape": (
                    list(generated["shared_expert_x"].shape)
                    if generated.get("shared_expert_x") is not None
                    else []
                ),
                "oriX": encode(generated.get("ori_x")),
                "oriXShape": (
                    list(generated["ori_x"].shape)
                    if generated.get("ori_x") is not None
                    else []
                ),
                "constExpertAlpha1": encode(generated.get("const_expert_alpha_1")),
                "constExpertAlpha1Shape": (
                    list(generated["const_expert_alpha_1"].shape)
                    if generated.get("const_expert_alpha_1") is not None
                    else []
                ),
                "constExpertAlpha2": encode(generated.get("const_expert_alpha_2")),
                "constExpertAlpha2Shape": (
                    list(generated["const_expert_alpha_2"].shape)
                    if generated.get("const_expert_alpha_2") is not None
                    else []
                ),
                "constExpertV": encode(generated.get("const_expert_v")),
                "constExpertVShape": (
                    list(generated["const_expert_v"].shape)
                    if generated.get("const_expert_v") is not None
                    else []
                ),
                "attributes": attributes,
            }
            plan_path = Path(run_dir) / f"plan_rank_{rank_idx}.json"
            with plan_path.open("w", encoding="utf-8") as plan_file:
                json.dump(plan, plan_file)

        env = os.environ.copy()
        hccl_port_range = get_cascade_config().hccl_port_range
        if hccl_port_range:
            env["HCCL_NPU_SOCKET_PORT_RANGE"] = hccl_port_range
        else:
            env.pop("HCCL_NPU_SOCKET_PORT_RANGE", None)
        env["HCCL_CONNECT_TIMEOUT"] = str(
            get_cascade_config().hccl_connect_timeout_s
        )
        env["PATH"] = (
            str(Path(sys.executable).resolve().parent) + os.pathsep + env.get("PATH", "")
        )
        env["TORCH_EXTENSIONS_DIR"] = str(cache_root / "torch_extensions")
        env["TORCH_COMPILE_DISABLE"] = "1"
        timeout = int(get_cascade_config().hccl_timeout_s)
        worker_dir = Path(__file__).resolve().parent
        launcher = subprocess.Popen(
            [
                sys.executable,
                str(worker_dir / "external_cascade_launcher.py"),
                run_dir,
                endpoint,
                str(worker_dir / "external_cascade_worker.py"),
                str(len(device_ids)),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            output, _ = launcher.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            launcher.kill()
            output, _ = launcher.communicate()
            raise RuntimeError(
                f"Moe external cascade timed out after {timeout}s:\n{output}"
            ) from None
        if launcher.returncode != 0:
            details = []
            for rank_idx in range(len(device_ids)):
                log_path = Path(run_dir) / f"worker_rank_{rank_idx}.log"
                if log_path.is_file():
                    details.append(
                        f"rank={rank_idx}\n{log_path.read_text(encoding='utf-8', errors='replace')}"
                    )
            raise RuntimeError(
                f"Moe external cascade failed rc={launcher.returncode}\n"
                + "\n".join(details)
            )

        payloads = {}
        for rank_idx, device_id in enumerate(device_ids):
            output_path = Path(run_dir) / f"output_rank_{rank_idx}.npz"
            if not output_path.is_file() or output_path.stat().st_size == 0:
                payloads[device_id] = None
                continue
            with np.load(output_path, allow_pickle=False) as payload:
                payloads[device_id] = (
                    payload["output_0"].tobytes(),
                    tuple(payload["output_shapes"].tolist()),
                    int(payload["rank"]),
                )
        return payloads
    finally:
        run_dir_obj.cleanup()


def _release_main_resources(thread_contexts, device_ids):
    from ttk.core_modules.npu.op_api.profiling import (
        release_retained_multi_device_resources,
    )

    first_context = next(iter(thread_contexts.values()))
    release_retained_multi_device_resources(first_context, list(device_ids))


def _execute_external_cascade(thread_contexts, device_ids, fill_results=True):
    state = getattr(_execute_external_cascade, "_state", None)
    state_key = (tuple(device_ids), next(iter(thread_contexts.values())).testcase_name)
    if state is not None and state["key"] == state_key:
        if fill_results:
            _fill_external_results(thread_contexts, device_ids, state["payloads"])
        return

    _release_main_resources(thread_contexts, device_ids)
    state = {
        "key": state_key,
        "payloads": _run_external_cascade(thread_contexts, device_ids),
    }
    _execute_external_cascade._state = state
    if fill_results:
        _fill_external_results(thread_contexts, device_ids, state["payloads"])
    return


def _fill_external_results(thread_contexts, device_ids, payloads):
    from ttk.core_modules.npu.op_api.profiling_structure import ApiProfilingResult

    for device_id in device_ids:
        context = thread_contexts[device_id]
        rank_idx = device_ids.index(device_id)
        payload = payloads.get(device_id)
        if payload is None:
            context.prof_result = ApiProfilingResult.fail("SUPPRESSED")
            continue
        output_bytes, output_shapes, payload_rank = payload
        if not output_bytes:
            context.prof_result = ApiProfilingResult.fail("SUPPRESSED")
            continue
        if payload_rank != rank_idx:
            raise RuntimeError(
                f"Moe external cascade rank mismatch: {payload_rank} != {rank_idx}"
            )
        context.prof_result = ApiProfilingResult(
            True,
            "E2E_EXECUTED",
            "E2E_EXECUTED",
            (output_bytes,),
            (output_shapes,),
        )

def execute_deepep_cascade(thread_context, device, stream_value):
    del device, stream_value
    from ttk.core_modules.npu.op_api.profiling_structure import ApiProfilingResult

    if int(thread_context.attributes.get("_moeDeepep", 0) or 0) != 1:
        return None
    payloads = getattr(_run_moe_dispatch_cascade, "_deepep_cascade_payloads", None)
    if payloads is None:
        raise RuntimeError("Moe DeepEP cascade payload is missing")
    payload = payloads.get(thread_context.my_rank)
    if payload is None:
        return ApiProfilingResult.fail("SUPPRESSED")
    output_bytes, output_shape, payload_rank = payload
    if not output_bytes:
        return ApiProfilingResult.fail("SUPPRESSED")
    rank_idx = thread_context.device_ids.index(thread_context.my_rank)
    if payload_rank != rank_idx:
        raise RuntimeError("Moe DeepEP cascade payload rank mismatch")
    return ApiProfilingResult(
        True,
        "E2E_EXECUTED",
        "E2E_EXECUTED",
        (output_bytes,),
        (output_shape,),
    )


__all__ = ["_run_moe_dispatch_cascade"]
