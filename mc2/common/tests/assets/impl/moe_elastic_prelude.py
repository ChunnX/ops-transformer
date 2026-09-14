#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""MoeDistribute elastic warmup prelude for TTK ACLNN execution."""

import ctypes
import logging
import os
import threading
from pathlib import Path

import numpy as np
import torch


def _load_case_data_module():
    import importlib.util

    path = Path(__file__).resolve().with_name("moe_case_data.py")
    spec = importlib.util.spec_from_file_location("mc2_moe_case_data_elastic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_elastic_context(context):
    configured_ranks = context.attributes.get("_moeElasticAvailableRanks")
    is_elastic = int(context.attributes.get("isElastic", 0) or 0) == 1
    is_expand = int(context.attributes.get("isExpand", 0) or 0) == 1
    if configured_ranks is None and not is_elastic and not is_expand:
        return False
    return int(context.attributes.get("_moeDeepep", 0) or 0) == 1


def _make_warmup_context(context):
    from ttk.core_modules.aclnn.op_api_info_keeper import OpApiInfoKeeper
    from ttk.core_modules.npu.op_api.mc2_dispatcher import patch_moe_rank_attributes
    from ttk.core_modules.testcase_manager import TestcaseAclnn

    op_api_info = OpApiInfoKeeper().info_of("aclnnMoeDistributeDispatchV3")
    warmup_context = TestcaseAclnn()
    warmup_context.api_name = "aclnnMoeDistributeDispatchV3"
    warmup_context.device_ids = context.device_ids
    warmup_context.my_rank = context.my_rank
    warmup_context.remark = context.remark
    warmup_context.attributes = {
        key: value for key, value in context.attributes.items() if key in op_api_info.params
    }
    for warmup_only_excluded in ("zeroExpertNum", "copyExpertNum", "constExpertNum"):
        warmup_context.attributes.pop(warmup_only_excluded, None)
    warmup_context.attributes.update(
        {name: 0 for name in ("zeroExpertNum", "copyExpertNum", "constExpertNum")}
    )
    warmup_context.attributes.update(
        {
            "globalBs": len(context.device_ids),
            "quantMode": 2,
            "expertTokenNumsType": 1,
            "commAlg": context.attributes.get("commAlg", ""),
        }
    )

    rank = list(context.device_ids).index(context.my_rank)
    warmup_context.attributes["epRankId"] = rank
    hidden_size = int(context.tensor_view_shapes[0][1])
    topk = int(context.tensor_view_shapes[1][1])
    hidden_size = int(context.flatten_tensors[0].shape[1])
    topk = int(context.flatten_tensors[1].shape[1])
    ep_world_size = int(warmup_context.attributes.get("epWorldSize", len(context.device_ids)))
    shared_rank_num = int(warmup_context.attributes.get("sharedExpertRankNum", 0))
    moe_expert_num = int(warmup_context.attributes.get("moeExpertNum", 0))
    local_expert_num = max(moe_expert_num // max(ep_world_size - shared_rank_num, 1), 1)
    warmup_topk = min(topk, local_expert_num)
    a_value = max(ep_world_size, warmup_topk)

    shapes = []
    dtypes = []
    input_ranges = []
    for name in op_api_info.tensors:
        if name == "x":
            shape, dtype, value_range = (
                (1, hidden_size),
                str(context.flatten_tensors[0].dtype).replace("torch.", ""),
                (-1024, 1024),
            )
        elif name == "expertIds":
            shape, dtype, value_range = (1, topk), "int32", (0, max(topk - 1, 0))
        elif name == "dynamicScalesOut":
            shape, dtype, value_range = (a_value,), "float32", (None, None)
        elif name == "assistInfoForCombineOut":
            shape, dtype, value_range = (max(topk, a_value * 128),), "int32", (None, None)
        elif name == "expertTokenNumsOut":
            shape, dtype, value_range = (local_expert_num,), "int64", (None, None)
        elif name == "epRecvCountsOut":
            shape, dtype, value_range = (ep_world_size * local_expert_num,), "int32", (None, None)
        elif name == "tpRecvCountsOut":
            shape, dtype, value_range = (int(warmup_context.attributes.get("tpWorldSize", 1)),), "int32", (None, None)
        elif name == "expandXOut":
            shape, dtype, value_range = (a_value, hidden_size), "int8", (None, None)
        elif name == "expandScalesOut":
            shape, dtype, value_range = (a_value,), "float32", (None, None)
        else:
            shape, dtype, value_range = None, None, (None, None)
            if name == "scalesOptional":
                dtype = "float32"
            elif name == "xActiveMaskOptional":
                dtype = "bool"
            elif name in ("expertScalesOptional", "expandScalesOptional"):
                dtype = "float32"
            elif name == "elasticInfoOptional":
                dtype = "int32"
            elif name == "performanceInfoOptional":
                dtype = "int64"
        shapes.append(shape)
        dtypes.append(dtype)
        input_ranges.append(value_range)

    warmup_context.tensor_view_shapes = tuple(shapes)
    warmup_context.tensor_dtypes = tuple(dtypes)
    warmup_context.scalar_dtypes = ()
    warmup_context.scalar_data_ranges = ()
    warmup_context.input_data_ranges = tuple(input_ranges)
    warmup_context.output_tensor_indexes = tuple(
        index for index, name in enumerate(op_api_info.tensors) if name.endswith("Out") or name == "expandXOut"
    )
    warmup_context.output_inplace_indexes = ()
    warmup_context.tensor_formats = tuple("ND" for _ in shapes)
    warmup_context.tensor_storage_shapes = ()
    warmup_context.tensor_view_strides = ()
    warmup_context.tensor_view_offsets = ()
    patch_moe_rank_attributes(warmup_context, rank, len(context.device_ids or ()))
    warmup_context._param_plan_cache = None
    return warmup_context


def _make_thread_device(main_device, device_id, context_value):
    from ttk.core_modules.npu.op_api.profiling import AclInterface
    from ttk.core_modules.runtime.rts_interface import RTSInterface

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
    context = ctypes.c_void_p(context_value)
    thread_rts["context"] = context
    thread_rts["context_storage"] = [context]
    thread_device._rts_interface = RTSInterface.__new__(RTSInterface)
    thread_device._rts_interface.__dict__ = thread_rts
    return thread_device


def run_moe_elastic_warmup(thread_contexts, device_ids, main_device, stream_value, plugin_path):
    from ttk.core_modules.npu.op_api.input_generation import InputGenerator
    from ttk.core_modules.npu.op_api.profiling import AclOpExecutor
    from ttk.utilities import get_global_storage, set_global_storage
    from ttk.utilities.classes import SWITCHES

    first_context = thread_contexts[device_ids[0]]
    if not _is_elastic_context(first_context):
        return
    configured_ranks = first_context.attributes.get("_moeElasticAvailableRanks")
    if configured_ranks is None:
        return
    logging.info(
        "Moe elastic warmup attrs available=%s tp=%s isElastic=%s",
        configured_ranks,
        first_context.attributes.get("tpWorldSize"),
        first_context.attributes.get("isElastic"),
    )
    available_ranks = set(int(rank) for rank in configured_ranks)
    is_expand = int(first_context.attributes.get("isExpand", 0) or 0) == 1
    if is_expand:
        available_ranks = set(device_ids)
    logging.info("Moe elastic warmup start devices=%s available=%s", device_ids, sorted(available_ranks))

    rank_contexts = {}
    for device_id in device_ids:
        rank_idx = list(device_ids).index(device_id)
        if rank_idx not in available_ranks:
            thread_contexts[device_id].attributes["_moeElasticSuppressed"] = True
        warmup_context = _make_warmup_context(thread_contexts[device_id])
        InputGenerator(warmup_context).gen()
        logging.info(
            "Moe elastic warmup prepared dev=%s shapes=%s",
            device_id,
            tuple(tuple(tensor.shape) if tensor is not None else None for tensor in warmup_context.flatten_tensors),
        )
        rank_contexts[device_id] = warmup_context
        rank_contexts[(device_id, "device")] = _make_thread_device(
            main_device,
            device_id,
            getattr(thread_contexts[device_id], "_cascade_context", None),
        )

    warmup_devices = list(device_ids)
    barrier = threading.Barrier(len(warmup_devices), timeout=1800)
    errors = {}

    def execute(device_id):
        try:
            barrier.wait()
            thread_device = rank_contexts[(device_id, "device")]
            thread_device._rts_interface.set_context(thread_device._rts_interface.context_storage[0])
            if not getattr(get_global_storage(), "root_path", None):
                set_global_storage(SWITCHES())
            if not getattr(get_global_storage(), "root_path", None):
                get_global_storage().root_path = os.getcwd()
            warmup_context = rank_contexts[device_id]
            warmup_context.prof_result = AclOpExecutor(
                warmup_context, thread_device
            ).do(stream=stream_value[device_id], skip_context_creation=True)
            if warmup_context.prof_result.failed():
                raise RuntimeError(f"Moe elastic warmup failed on device {device_id}")
        except BaseException:
            logging.exception("Moe elastic warmup failed on device %s", device_id)
            errors[device_id] = True

    threads = [
        threading.Thread(target=execute, args=(device_id,), name=f"moe-elastic-warmup-{device_id}")
        for device_id in warmup_devices
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError(f"Moe elastic warmup failed on devices {sorted(errors)}")
    logging.info("Moe elastic warmup complete available=%s", sorted(available_ranks))

    logging.info(
        "Moe elastic formal call count: TTK=1 original=2; deferred to worker for %s",
        sorted(available_ranks),
    )


__all__ = ["run_moe_elastic_warmup"]
