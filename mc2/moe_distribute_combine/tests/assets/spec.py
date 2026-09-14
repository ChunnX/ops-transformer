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

"""TestSpec adapter for MoeDistributeCombine TTK assets.

All impl modules are loaded lazily to avoid import-time failures.
"""

import importlib.util
import logging
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}


def _load_case_data_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "common"
        / "tests"
        / "assets"
        / "impl"
        / "moe_case_data.py"
    )
    spec = importlib.util.spec_from_file_location("mc2_moe_case_data_combine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generated_case(kwargs, expand_x):
    module = _load_case_data_module()
    context = kwargs.get("_ttk_context")
    attributes = getattr(context, "attributes", None) or {}

    def attribute(name, default=None):
        if name in attributes:
            return attributes[name]
        return kwargs.get(name, default)

    remark = {
        item.split("=", 1)[0].strip(): item.split("=", 1)[1].strip()
        for item in str(kwargs.get("remark", "")).split(",")
        if "=" in item
    }
    config = module.MoeCaseConfig(
        global_bs=int(attribute("globalBs", remark.get("global_bs", expand_x.shape[0]))),
        hidden_size=int(expand_x.shape[1]),
        top_k=int(kwargs.get("_moeCombineTopK", 1)),
        seed=int(remark.get("seed", 1)),
        ep_world_size=int(attribute("epWorldSize", 1)),
        tp_world_size=int(attribute("tpWorldSize", 1)),
        moe_expert_num=int(attribute("moeExpertNum", 1)),
        local_moe_expert_num_override=int(attribute("localMoeExpertNum", 0)),
        shared_expert_num=int(attribute("sharedExpertNum", 0)),
        shared_expert_rank_num=int(attribute("sharedExpertRankNum", 0)),
        const_expert_num=int(attribute("constExpertNum", 0)),
        zero_expert_num=int(attribute("zeroExpertNum", 0)),
        copy_expert_num=int(attribute("copyExpertNum", 0)),
        rank_num_per_shared_expert=(
            int(attribute("sharedExpertRankNum", 0))
            // int(attribute("sharedExpertNum", 0) or 1)
        ),
        input_dtype=str(expand_x.dtype).replace("torch.", ""),
        is_mask=bool(int(attribute("xActiveMaskDim", 0))),
        active_mask_dim=int(attribute("xActiveMaskDim", 0)),
        is_variable_bs=bool(int(attribute("_moeVariableBs", 0))),
        is_quant=int(attribute("quantMode", 0)) == 2,
        is_dispatch_scales=bool(int(attribute("isDispatchScales", 0))),
        is_topk_balance=(
            (kwargs.get("_dispatchCommAlg") or kwargs.get("commAlg"))
            == "fullmesh_v2"
        ),
        is_combine_x=(
            int(attribute("isCombineX", 0)) == 1
        ),
        is_shared_expert_x=bool(int(attribute("isSharedExpertX", 0))),
        is_add_rms_norm=kwargs.get("api_name")
        in (
            "aclnnMoeDistributeCombineAddRmsNorm",
            "aclnnMoeDistributeCombineAddRmsNormV2",
        ),
        is_deepep=(
            remark.get("isDeepep") == "1"
            or int(attribute("_moeDeepep", 0) or 0) == 1
        ),
        is_x_random=int(attribute("isXRandom", 0)) == 1,
        is_elastic=(
            int(attribute("isElastic", 0)) == 1
        ),
        elastic_shared_rank_num=int(attribute("shareBrokenCardNum", 0)),
        elastic_moe_rank_num=int(attribute("moeBrokenCardNum", 0)),
        comm_quant_mode=int(attribute("commQuantMode", 0)),
        is_expand=int(attribute("isExpand", 0)) == 1,
    )
    return module, module.load_or_generate_case(config), config


def _load_impl_module(stem):
    if stem not in _impl_cache:
        path = ASSET_IMPL_DIR / f"{stem}.py"
        spec = importlib.util.spec_from_file_location(
            f"moe_distribute_combine_assets_impl_{stem}_{abs(hash(path))}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _impl_cache[stem] = module
    return _impl_cache[stem]


class AclnnMoeDistributeCombineSpec:
    """TestSpec for aclnnMoeDistributeCombine (ACLNN multi-device path)."""

    @staticmethod
    def golden(thread_contexts, device_ids, all_precision):
        return _load_impl_module("golden").moe_distribute_combine_multi_device_golden(
            thread_contexts, device_ids, all_precision
        )

    @staticmethod
    def compare(*outputs, **kwargs):
        return _load_impl_module("compare").compare(*outputs, **kwargs)

    @staticmethod
    def prepare_execute(thread_contexts, device_ids):
        return None

    @staticmethod
    def early_execute(thread_contexts, device_ids):
        module = _load_impl_module("cascade_worker")
        payloads = module._run_deepep_cascade_external_worker(
            thread_contexts, device_ids
        )
        module._fill_external_results(thread_contexts, device_ids, payloads)
        return {
            device_id: thread_contexts[device_id].prof_result
            for device_id in device_ids
        }

    @staticmethod
    def before_execute(thread_contexts, device_ids, main_device, stream_value, plugin_path):
        first_context = thread_contexts[device_ids[0]]
        use_deepep_worker = (
            first_context.api_name == "aclnnMoeDistributeCombineV3"
            and int(first_context.attributes.get("_moeDeepep", 0) or 0) == 1
        )
        if not use_deepep_worker:
            module_path = Path(__file__).parents[3] / "common" / "tests" / "assets" / "impl" / "moe_elastic_prelude.py"
            module_spec = importlib.util.spec_from_file_location("mc2_moe_elastic_prelude_combine", module_path)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            module.run_moe_elastic_warmup(thread_contexts, device_ids, main_device, stream_value, plugin_path)
        return _load_impl_module("cascade_worker")._run_moe_dispatch_cascade(
            thread_contexts, device_ids, main_device, stream_value, plugin_path
        )

    @staticmethod
    def execute(thread_context, device, stream_value):
        if (
            thread_context.api_name in (
                "aclnnMoeDistributeCombineV2",
                "aclnnMoeDistributeCombineV3",
            )
            and int(thread_context.attributes.get("_moeDeepep", 0) or 0) != 1
        ):
            from ttk.config.loader import get_cascade_config

            if get_cascade_config().external_process:
                _load_impl_module("cascade_worker")._execute_external_cascade(
                    getattr(
                        _load_impl_module("cascade_worker")._run_moe_dispatch_cascade,
                        "_thread_contexts",
                    )
                    or {thread_context.my_rank: thread_context},
                    tuple(thread_context.device_ids or (thread_context.my_rank,)),
                )
                return thread_context.prof_result
        if (
            thread_context.api_name == "aclnnMoeDistributeCombineV3"
            and int(thread_context.attributes.get("_moeDeepep", 0) or 0) == 1
        ):
            return _load_impl_module("cascade_worker").execute_deepep_cascade(
                thread_context, device, stream_value
            )
        return _load_impl_module("cascade_worker")._execute_moe_add_rms_norm_e2e(
            thread_context, device, stream_value
        )

    @staticmethod
    def customize_inputs(
        *args,
        **kwargs,
    ):
        is_add_rms_norm = kwargs.get("api_name") in (
            "aclnnMoeDistributeCombineAddRmsNorm",
            "aclnnMoeDistributeCombineAddRmsNormV2",
        )
        context = kwargs.get("_ttk_context")
        if context is not None and len(args) > 6 and args[6] is not None:
            context.attributes["xActiveMaskDim"] = int(args[6].dim())
        if is_add_rms_norm:
            if len(args) not in (33, 41):
                raise ValueError(
                    f"MoeDistributeCombineAddRmsNorm requires 33 or 41 configured parameters, got {len(args)}"
                )
            expand_x = args[0]
            expert_ids = args[1]
            assist_info_for_combine = args[2]
            ep_send_counts = args[3]
            expert_scales = args[4]
            residual_x = args[5]
            gamma = args[6]
            tp_send_counts = args[7]
            x_active_mask = args[8]
            activation_scale = args[9]
            weight_scale = args[10]
            group_list = args[11]
            expand_scales = args[12]
            shared_expert_x = args[13]
            ep_rank_id = int(args[21] if len(args) == 41 else args[16])
            norm_eps = float(args[34] if len(args) == 41 else args[29])
            kwargs["_moeCombineTopK"] = int(expert_ids.shape[1]) if expert_ids is not None and expert_ids.dim() > 1 else 1
            if context is not None:
                context.attributes.setdefault("isActivateMask", int(x_active_mask is not None))
                context.attributes.setdefault(
                    "xActiveMaskDim",
                    int(x_active_mask.dim() if x_active_mask is not None else 0),
                )

            import torch
            module, generated_case, config = _generated_case(kwargs, expand_x)
            generated = module.rank_case(generated_case, config, ep_rank_id)
            context = kwargs.get("_ttk_context")
            if context is not None:
                context.attributes["_moeCaseData"] = generated

            def copy_tensor(target, source):
                if target is None:
                    return
                if source.numel() > target.numel():
                    raise ValueError(
                        f"AddRmsNorm source is larger than target {tuple(target.shape)}: {tuple(source.shape)}"
                    )
                if target.numel() == 0:
                    return
                flattened = source.reshape(-1)
                if flattened.numel() < target.numel():
                    flattened = torch.cat(
                        [
                            flattened,
                            torch.zeros(
                                target.numel() - flattened.numel(), dtype=flattened.dtype
                            ),
                        ]
                    )
                target.copy_(flattened[: target.numel()].reshape(target.shape).to(target.dtype))

            copy_tensor(
                expand_x,
                generated.get(
                    "initial_expand_x",
                    generated["expand_x"],
                ),
            )
            copy_tensor(expert_ids, generated["expert_ids"])
            copy_tensor(
                assist_info_for_combine,
                generated.get(
                    "initial_expand_idx",
                    generated["expand_idx"],
                ),
            )
            ep_send_counts_source = generated.get(
                "initial_ep_recv_counts",
                generated["ep_recv_counts"],
            )
            ep_send_counts.resize_(ep_send_counts_source.shape)
            ep_send_counts.copy_(ep_send_counts_source.to(ep_send_counts.dtype))
            copy_tensor(expert_scales, generated["expert_scales"])
            generator = torch.Generator(device="cpu").manual_seed(42)
            random_residual = torch.rand(
                residual_x.shape,
                dtype=residual_x.dtype,
                generator=generator,
            )
            copy_tensor(residual_x, random_residual)
            gamma.fill_(1)
            if tp_send_counts is not None:
                tp_send_counts.fill_(int(generated["tokens"].sum()))
            if x_active_mask is not None:
                copy_tensor(
                    x_active_mask,
                    generated.get("x_active_mask_input", generated["x_active_mask"]),
                )
            if shared_expert_x is not None:
                copy_tensor(shared_expert_x, generated["shared_expert_x"])
            if activation_scale is not None:
                activation_scale.zero_()
            if weight_scale is not None:
                weight_scale.zero_()
            if group_list is not None:
                group_list.zero_()
            if expand_scales is not None:
                expand_scales.zero_()
            if norm_eps <= 0:
                raise ValueError(f"AddRmsNorm normEps must be positive, got {norm_eps}")
            return args[:19]
        if len(args) == 36:
            context = kwargs.get("_ttk_context")
            if context is not None and context.device_ids:
                context.attributes["epRankId"] = list(context.device_ids).index(
                    context.my_rank
                )
            return args[:17]
        if len(args) == 28:
            context = kwargs.get("_ttk_context")
            if context is not None and args[3] is not None:
                context.attributes["xActiveMaskDim"] = int(args[3].dim())
            return (*args[:11], None, args[27])
        if len(args) > 36 and isinstance(args[18], str):
            ori_x = args[13]
            ep_world_size = int(args[19])
            ep_rank_id = int(args[20])
            shared_expert_rank_num = int(args[27])
            return (
                *args[:18],
                args[36],
            )
        if len(args) not in (13, 18):
            raise ValueError(
                f"MoeDistributeCombine requires 13 or 18 configured tensor parameters, got {len(args)}"
            )
        is_v3 = len(args) == 18
        expand_x = args[0]
        expert_ids = args[1]
        assist_info_for_combine = args[2]
        ep_send_counts = args[3]
        expert_scales = args[4]
        tp_send_counts = args[5] if len(args) > 5 else kwargs.get("tpSendCountsOptional")
        x_active_mask = args[6] if len(args) > 6 else kwargs.get("xActiveMaskOptional")
        activation_scale = args[7] if len(args) > 7 else kwargs.get("activationScaleOptional")
        weight_scale = args[8] if len(args) > 8 else kwargs.get("weightScaleOptional")
        group_list = args[9] if len(args) > 9 else kwargs.get("groupListOptional")
        expand_scales = args[10] if len(args) > 10 else kwargs.get("expandScalesOptional")
        shared_expert_x = args[11] if len(args) > 11 else kwargs.get("sharedExpertXOptional")
        row_mask = None
        if args[-1] is not None:
            args[-1].zero_()
        ori_x = None
        if len(args) > 36:
            ep_world_size = int(args[19])
            ep_rank_id = int(args[20])
            shared_expert_rank_num = int(args[27])
        else:
            ep_world_size = int(args[13])
            context = kwargs.get("_ttk_context")
            if context is not None and context.device_ids:
                ep_rank_id = int(context.attributes.get("epRankId", 0))
            else:
                ep_rank_id = int(kwargs.get("epRankId", 0))
            shared_expert_rank_num = int(args[22])
        kwargs["_moeCombineTopK"] = int(expert_ids.shape[1]) if expert_ids is not None and expert_ids.dim() > 1 else 1
        import torch
        context = kwargs.get("_ttk_context")
        if context is not None and x_active_mask is not None:
            context.attributes["xActiveMaskDim"] = int(x_active_mask.dim())
        module, generated_case, config = _generated_case(kwargs, expand_x)
        generated = module.rank_case(generated_case, config, ep_rank_id)
        if context is not None:
            context.attributes["_moeCaseData"] = generated
        if x_active_mask is not None and x_active_mask.dim() == 1:
            row_mask = x_active_mask.to(torch.bool)

        def copy_tensor(target, source):
            if target is None:
                return
            source = source.reshape(-1)[: target.numel()].to(target.dtype)
            if source.numel() < target.numel():
                source = torch.cat(
                    [source, torch.zeros(target.numel() - source.numel(), dtype=source.dtype)]
                )
            target.copy_(source.reshape(target.shape))

        source_expand_x = generated["expand_x"]
        if source_expand_x.shape != tuple(expand_x.shape):
            padded_expand_x = torch.zeros(
                tuple(expand_x.shape), dtype=expand_x.dtype
            )
            valid_rows = min(source_expand_x.shape[0], expand_x.shape[0])
            padded_expand_x[:valid_rows] = source_expand_x[:valid_rows].to(
                expand_x.dtype
            )
            expand_x.copy_(padded_expand_x)
        copy_tensor(expert_ids, generated["expert_ids"])
        copy_tensor(assist_info_for_combine, generated["expand_idx"])
        source_ep_send_counts = generated["ep_recv_counts"]
        if source_ep_send_counts.numel() < ep_send_counts.numel():
            source_ep_send_counts = torch.cat(
                [
                    source_ep_send_counts,
                    torch.full(
                        (ep_send_counts.numel() - source_ep_send_counts.numel(),),
                        int(source_ep_send_counts[-1]),
                        dtype=source_ep_send_counts.dtype,
                    ),
                ]
            )
        copy_tensor(ep_send_counts, source_ep_send_counts)
        copy_tensor(expert_scales, generated["expert_scales"])
        if tp_send_counts is not None:
            if tp_send_counts.numel() == 1:
                tp_send_counts.fill_(int(generated["expand_x"].shape[0]))
            else:
                tp_send_counts.fill_(int(generated["tokens"].sum()))
        if x_active_mask is not None:
            mask = generated["x_active_mask"]
            if mask.numel() < x_active_mask.numel():
                mask = torch.cat(
                    [mask, torch.zeros(x_active_mask.numel() - mask.numel(), dtype=mask.dtype)]
                )
            x_active_mask.copy_(mask.reshape(x_active_mask.shape).to(x_active_mask.dtype))
            if mask.numel() == x_active_mask.numel() and x_active_mask.dim() == 1:
                row_mask = x_active_mask.to(torch.bool)
        if shared_expert_x is not None:
            copy_tensor(shared_expert_x, generated["shared_expert_x"])
        if is_v3:
            ep_rank_id = int(kwargs.get("epRankId", 0))
            if args[12] is not None:
                copy_tensor(args[12], torch.zeros(1, dtype=torch.int32))
            if args[13] is not None:
                copy_tensor(args[13], generated["x"][row_mask])
            if args[14] is not None:
                copy_tensor(args[14], generated["const_expert_alpha_1"])
            if args[15] is not None:
                copy_tensor(args[15], generated["const_expert_alpha_2"])
            if args[16] is not None:
                copy_tensor(args[16], generated["const_expert_v"])
            return args
        if ori_x is not None:
            copy_tensor(ori_x, generated["x"])
        golden_tokens = generated["tokens"]
        if tp_send_counts is not None:
            if tp_send_counts.numel() == 1:
                tp_send_counts.fill_(int(generated["expand_x"].shape[0]))
            else:
                tp_send_counts.fill_(int(golden_tokens.sum()))
        if x_active_mask is not None:
            x_active_mask.copy_(generated["x_active_mask"])
        return args


class MoeDistributeCombineE2ESpec:
    """TestSpec for torch_npu.npu_moe_distribute_combine (E2E multi-device path)."""

    @staticmethod
    def golden(cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True):
        return _load_impl_module("golden").moe_distribute_combine_e2e_golden(
            cpu_inputs_per_rank, attrs, rank, ws, dist_avail
        )

    @staticmethod
    def input(
        expand_x,
        expert_ids,
        expand_idx,
        ep_send_counts,
        expert_scales,
        group_ep,
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        *,
        tp_send_counts=None,
        x_active_mask=None,
        expand_scales=None,
        shared_expert_x=None,
        **kwargs,
    ):
        del expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales
        del group_ep, ep_world_size, ep_rank_id, moe_expert_num
        del tp_send_counts, x_active_mask, expand_scales, shared_expert_x, kwargs


__spec__ = {
    "aclnnMoeDistributeCombine": "AclnnMoeDistributeCombineSpec",
    "aclnnMoeDistributeCombineV2": "AclnnMoeDistributeCombineSpec",
    "aclnnMoeDistributeCombineV3": "AclnnMoeDistributeCombineSpec",
    "aclnnMoeDistributeCombineV4": "AclnnMoeDistributeCombineSpec",
    "aclnnMoeDistributeCombineAddRmsNorm": "AclnnMoeDistributeCombineSpec",
    "aclnnMoeDistributeCombineAddRmsNormV2": "AclnnMoeDistributeCombineSpec",
    "torch_npu.npu_moe_distribute_combine": "MoeDistributeCombineE2ESpec",
    "torch_npu.npu_moe_distribute_combine_v2": "MoeDistributeCombineE2ESpec",
    "cann_ops_transformer.npu_moe_distribute_combine": "MoeDistributeCombineE2ESpec",
}
