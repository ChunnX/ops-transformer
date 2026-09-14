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

"""TestSpec adapter for MoeDistributeDispatch TTK assets.

All impl modules are loaded lazily to avoid import-time failures.
"""

import importlib.util
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}


def _load_impl_module(stem):
    if stem not in _impl_cache:
        path = ASSET_IMPL_DIR / f"{stem}.py"
        spec = importlib.util.spec_from_file_location(
            f"moe_distribute_dispatch_assets_impl_{stem}_{abs(hash(path))}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _impl_cache[stem] = module
    return _impl_cache[stem]


class AclnnMoeDistributeDispatchSpec:
    """TestSpec for aclnnMoeDistributeDispatch (ACLNN multi-device path)."""

    @staticmethod
    def golden(thread_contexts, device_ids, all_precision):
        return _load_impl_module("golden").moe_distribute_dispatch_multi_device_golden(
            thread_contexts, device_ids, all_precision
        )

    @staticmethod
    def compare(*outputs, **kwargs):
        return _load_impl_module("compare").compare(*outputs, **kwargs)

    @staticmethod
    def customize_inputs(*args, **kwargs):
        from ttk.core_modules.aclnn.op_api_info_keeper import OpApiInfoKeeper

        api_name = kwargs.get("api_name", "aclnnMoeDistributeDispatchV3")
        op_api_info = OpApiInfoKeeper().info_of(api_name)
        if op_api_info is None:
            raise RuntimeError(f"{api_name} is unavailable in the current CANN package")
        tensor_count = op_api_info.tensors.index("expandXOut")
        param_names = tuple(op_api_info.params)
        args = list(args)
        scalar_args = args[tensor_count:]
        if len(args) > tensor_count:
            args = args[:tensor_count]
        deepep_enabled = False
        remarks = {
            item.split("=", 1)[0].strip(): item.split("=", 1)[1].strip()
            for item in (kwargs.get("remark") or "").split(",")
            if "=" in item
        }
        if kwargs.get("isDeepep") or remarks.get("isDeepep") == "1":
            deepep_enabled = True
        if deepep_enabled and args and args[1] is not None and args[1].dim() > 1 and args[1].shape[1] > 1:
            kwargs["_moeDeepepOriginalTopK"] = int(args[1].shape[1])
            args[1] = args[1][:, :1]
        args.extend(None for _ in range(tensor_count - len(args)))
        extras = {key: value for key, value in kwargs.items() if key != "commAlg"}
        attrs = {
            name: extras.pop(name, None)
            for name in param_names
            if name not in op_api_info.tensors and extras.get(name) is not None
        }
        scalar_param_names = tuple(
            name for name in param_names if name not in op_api_info.tensors
        )
        for name, value in zip(scalar_param_names, scalar_args):
            if value is not None and name not in attrs:
                attrs[name] = value
        if "isElastic" in extras:
            attrs["isElastic"] = int(extras.pop("isElastic") or 0)
        if "elasticMoeRankNum" in extras:
            attrs["elasticMoeRankNum"] = int(extras.pop("elasticMoeRankNum") or 0)
        if "isDeepep" in extras:
            attrs["_moeDeepep"] = int(extras.pop("isDeepep") or 0)
        elif remarks.get("isDeepep") == "1":
            attrs["_moeDeepep"] = 1
        if "isExpand" in extras:
            attrs["isExpand"] = int(extras.pop("isExpand") or 0)
        elif remarks.get("isExpand") == "1":
            attrs["isExpand"] = 1
        if "_moeDeepep" in attrs and "_moeDeepep" in extras:
            extras.pop("_moeDeepep")
        if "_ttk_context" in extras:
            attrs["_ttk_context"] = extras.pop("_ttk_context")
        return _load_impl_module("inputs").moe_distribute_dispatch_inputs(
            *args,
            **attrs,
            **({"api_name": api_name} if "api_name" not in kwargs else {}),
            **extras,
        )

    @staticmethod
    def before_execute(thread_contexts, device_ids, main_device, stream_value, plugin_path):
        import logging

        logging.info("MoeDistributeDispatch before_execute entered")
        return _load_impl_module("elastic_dispatch_executor").run_elastic_dispatch_warmup(
            thread_contexts, device_ids, main_device, stream_value, plugin_path
        )

    @staticmethod
    def prepare_execute(thread_contexts, device_ids):
        return _load_impl_module("elastic_dispatch_executor").prepare_elastic_dispatch(
            thread_contexts, device_ids
        )

    @staticmethod
    def execute(thread_context, device, stream_value):
        if thread_context.api_name != "aclnnMoeDistributeDispatchV3":
            return None
        return _load_impl_module("elastic_dispatch_executor").execute_elastic_dispatch(
            thread_context, device, stream_value
        )


class MoeDistributeDispatchE2ESpec:
    """TestSpec for torch_npu.npu_moe_distribute_dispatch (E2E multi-device path)."""

    @staticmethod
    def golden(cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True):
        return _load_impl_module("golden").moe_distribute_dispatch_e2e_golden(
            cpu_inputs_per_rank, attrs, rank, ws, dist_avail
        )


__spec__ = {
    "aclnnMoeDistributeDispatch": "AclnnMoeDistributeDispatchSpec",
    "aclnnMoeDistributeDispatchV2": "AclnnMoeDistributeDispatchSpec",
    "aclnnMoeDistributeDispatchV3": "AclnnMoeDistributeDispatchSpec",
    "aclnnMoeDistributeDispatchV4": "AclnnMoeDistributeDispatchSpec",
    "torch_npu.npu_moe_distribute_dispatch": "MoeDistributeDispatchE2ESpec",
}
