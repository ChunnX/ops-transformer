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

"""Golden for aclnnMoeDistributeCombine / torch_npu.npu_moe_distribute_combine.

MoeDistributeCombine: all_to_allv(expand_x) -> scatter back to original positions -> sum.
Pure CPU golden, no HCCL cascade (communication simulated on CPU).
Single output: x_out (summed and un-permuted back to original token order).

E2E golden returns a list of x_out tensors (one per rank).
"""

import logging

import numpy as np


def _load_framework():
    """Load golden utils from mc2/common/tests/assets."""
    import importlib.util
    from pathlib import Path

    utils_path = (
        Path(__file__).resolve().parents[4]
        / "common"
        / "tests"
        / "assets"
        / "golden_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"mc2_common_golden_utils_{abs(hash(utils_path))}", utils_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from ttk.core_modules.npu.op_api.comparison import Comparator

    return mod.to_torch_f32, mod.fmt_compare_result, Comparator


def _load_case_data_module():
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
    spec = importlib.util.spec_from_file_location(
        f"mc2_moe_case_data_{abs(hash(path))}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decode_output(raw, dtype_name, shape):
    from ttk.utilities.dtypes import numpy_bfloat16, numpy_to_torch_tensor

    if raw is None or isinstance(raw, str):
        raise RuntimeError(f"Missing Combine output bytes: {raw!r}")
    if isinstance(raw, (list, tuple)):
        raw = b"".join(raw)
    if dtype_name == "bfloat16":
        np_dtype = numpy_bfloat16()
    else:
        np_dtype = np.dtype(dtype_name)
    decoded = np.frombuffer(raw, dtype=np_dtype).copy()
    tensor = numpy_to_torch_tensor(decoded)
    shape = tuple(shape)
    element_count = int(np.prod(shape))
    if tensor.numel() == element_count:
        return tensor.reshape(shape)
    if element_count == 0:
        return tensor[:0].reshape(shape)
    if tensor.numel() < element_count:
        tail_count = int(np.prod(shape[1:])) if len(shape) > 1 else 0
        actual_rows = tensor.numel() // tail_count if tail_count else 0
        if tail_count and tensor.numel() % tail_count == 0 and actual_rows <= shape[0]:
            return tensor.reshape((actual_rows, *shape[1:]))
        raise ValueError(
            "Combine output byte count mismatch: "
            f"expected shape={shape}, got {tensor.numel()} elements"
        )
    return tensor[:element_count].reshape(shape)


def _simulate_moe_alltoallv(all_rank_inputs, world_size, send_counts_per_rank):
    """Simulate all_to_allv on CPU: each rank sends chunks to all other ranks."""
    import torch

    rank_outputs = {}
    for target_rank in range(world_size):
        received_chunks = []
        for src_rank in range(world_size):
            src_data = all_rank_inputs[src_rank]
            src_counts = send_counts_per_rank[src_rank]
            offset = 0
            for dst_idx in range(world_size):
                count = int(src_counts[dst_idx])
                if dst_idx == target_rank and count > 0:
                    received_chunks.append(src_data[offset : offset + count])
                offset += count
        if received_chunks:
            rank_outputs[target_rank] = torch.cat(received_chunks, dim=0)
        else:
            h = all_rank_inputs[0].shape[-1] if all_rank_inputs[0].dim() > 1 else 1
            rank_outputs[target_rank] = torch.zeros(
                0, h, dtype=all_rank_inputs[0].dtype
            )
    return rank_outputs


# ---------------------------------------------------------------------------
# E2E golden: returns list of x_out tensors (one per rank)
# ---------------------------------------------------------------------------


def moe_distribute_combine_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    import torch

    world_size = len(cpu_inputs_per_rank)
    ep_ws = int(attrs.get("ep_world_size", world_size))
    moe_expert_num = int(attrs.get("moe_expert_num", 1))
    local_expert_num = moe_expert_num // ep_ws if ep_ws > 0 else moe_expert_num

    # Combine inputs per rank:
    #  [0] expand_x: dispatched+received tokens (padded to tp_ws * A_moe)
    #  [1] expert_ids: original expert_ids (bs, k)
    #  [2] expand_idx: indices mapping expand_x rows back to (token, k) positions
    #  [3] ep_send_counts: per-rank send counts (cumsum format, ep_ws values)
    #  [4] expert_scales: per-token scaling factors (bs, k) or (bs*k,)
    all_rank_expand_x = {}
    all_rank_send_counts = {}
    all_rank_expert_ids = {}
    all_rank_expand_idx = {}
    all_rank_expert_scales = {}

    for r in range(world_size):
        inputs = cpu_inputs_per_rank[r]
        expand_x = (
            inputs[0].float() if (len(inputs) > 0 and inputs[0] is not None) else None
        )
        expert_ids = inputs[1] if (len(inputs) > 1 and inputs[1] is not None) else None
        expand_idx = inputs[2] if (len(inputs) > 2 and inputs[2] is not None) else None
        ep_send_counts = (
            inputs[3] if (len(inputs) > 3 and inputs[3] is not None) else None
        )
        expert_scales = (
            inputs[4] if (len(inputs) > 4 and inputs[4] is not None) else None
        )

        if expand_x is None:
            all_rank_expand_x[r] = torch.zeros(0, 1)
            all_rank_send_counts[r] = [0] * ep_ws
            all_rank_expert_ids[r] = torch.zeros(0, 1, dtype=torch.int32)
            all_rank_expand_idx[r] = torch.zeros(0, dtype=torch.int32)
            all_rank_expert_scales[r] = torch.zeros(0)
            continue

        h = expand_x.shape[1]

        # Derive send_counts from ep_send_counts (cumsum format)
        # ep_send_counts is cumsum, so send_counts[i] = ep_send_counts[i] - ep_send_counts[i-1]
        if ep_send_counts is not None and ep_send_counts.numel() > 0:
            esc = ep_send_counts.float()
            send_counts = []
            prev = 0.0
            for i in range(min(ep_ws, esc.numel())):
                cur = float(esc[i])
                send_counts.append(int(cur - prev))
                prev = cur
            while len(send_counts) < ep_ws:
                send_counts.append(0)
        else:
            # Fallback: evenly distribute
            num_tokens = expand_x.shape[0]
            per_rank = num_tokens // ep_ws
            send_counts = [per_rank] * ep_ws
            remainder = num_tokens % ep_ws
            for i in range(remainder):
                send_counts[i] += 1

        all_rank_expand_x[r] = expand_x
        all_rank_send_counts[r] = send_counts
        all_rank_expert_ids[r] = (
            expert_ids
            if expert_ids is not None
            else torch.zeros(0, 1, dtype=torch.int32)
        )
        all_rank_expand_idx[r] = (
            expand_idx if expand_idx is not None else torch.zeros(0, dtype=torch.int32)
        )
        all_rank_expert_scales[r] = (
            expert_scales if expert_scales is not None else torch.zeros(0)
        )

    # Simulate alltoallv: each rank sends its expand_x to destination ranks
    alltoallv_out = _simulate_moe_alltoallv(
        all_rank_expand_x, world_size, all_rank_send_counts
    )

    # For each rank, scatter the received tokens back to original positions and sum
    goldens = []
    for r in range(world_size):
        a2a_result = alltoallv_out.get(r, torch.zeros(0, 1))
        expert_ids = all_rank_expert_ids[r]
        expand_idx = all_rank_expand_idx[r]
        expert_scales = all_rank_expert_scales[r]
        expand_x = all_rank_expand_x[r]

        h = a2a_result.shape[1] if a2a_result.dim() > 1 else 1
        bs = expert_ids.shape[0] if expert_ids.dim() > 0 else 0
        k = expert_ids.shape[1] if expert_ids.dim() > 1 else 1

        x_out = torch.zeros(bs, h)

        if a2a_result.numel() > 0 and bs > 0 and expand_idx.numel() > 0:
            num_valid = min(expand_idx.shape[0], a2a_result.shape[0])
            for idx_pos in range(num_valid):
                orig_flat_idx = int(expand_idx[idx_pos])
                orig_token = orig_flat_idx // k if k > 0 else orig_flat_idx
                k_idx = orig_flat_idx % k if k > 0 else 0
                if orig_token < bs:
                    scale = 1.0
                    if expert_scales.numel() > 0:
                        if (
                            expert_scales.dim() > 1
                            and orig_token < expert_scales.shape[0]
                            and k_idx < expert_scales.shape[1]
                        ):
                            scale = float(expert_scales[orig_token][k_idx])
                        elif (
                            expert_scales.dim() == 1
                            and orig_flat_idx < expert_scales.shape[0]
                        ):
                            scale = float(expert_scales[orig_flat_idx])
                    x_out[orig_token] = (
                        x_out[orig_token] + a2a_result[idx_pos].float() * scale
                    )

        goldens.append(x_out.contiguous())
    return goldens


# ---------------------------------------------------------------------------
# ACLNN multi-device golden
# ---------------------------------------------------------------------------


def moe_distribute_combine_multi_device_golden(
    thread_contexts, device_ids, all_precision
):
    import torch
    import importlib.util
    from pathlib import Path

    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    case_module = _load_case_data_module()
    is_deep_ep = bool(attrs.get("_moeDeepep", 0))
    case_config = case_module.MoeCaseConfig(
        global_bs=int(attrs.get("globalBs", first_ctx.flatten_tensors[0].shape[0])),
        hidden_size=int(first_ctx.flatten_tensors[0].shape[1]),
        top_k=int(first_ctx.flatten_tensors[1].shape[1]),
        seed=int(next(
            (
                int(part.split("=", 1)[1])
                for part in str(first_ctx.remark or "").split(",")
                if part.startswith("seed=")
            ),
            1,
        )),
        ep_world_size=int(attrs.get("epWorldSize", len(device_ids))),
        tp_world_size=int(attrs.get("tpWorldSize", 1)),
        moe_expert_num=int(attrs.get("moeExpertNum", 1)),
        local_moe_expert_num_override=int(
            attrs.get("localMoeExpertNum", 0)
        ),
        shared_expert_num=int(attrs.get("sharedExpertNum", 0)),
        shared_expert_rank_num=(
            int(attrs.get("sharedExpertRankNum", 0))
            or int(attrs.get("sharedExpertNum", 0))
            * int(attrs.get("rankNumPerSharedExpert", 0))
        ),
        const_expert_num=int(attrs.get("constExpertNum", 0)),
        rank_num_per_shared_expert=int(
            attrs.get("rankNumPerSharedExpert", 0)
            or attrs.get("sharedExpertRankNum", 0)
            // max(attrs.get("sharedExpertNum", 0), 1)
        ),
        input_dtype=str(first_ctx.tensor_dtypes[0]).replace("torch.", ""),
        is_mask=int(attrs.get("xActiveMaskDim", 0)) > 0,
        is_variable_bs=bool(int(attrs.get("_moeVariableBs", 0))),
        is_quant=int(attrs.get("quantMode", 0)) == 2,
        is_dispatch_scales=bool(attrs.get("isDispatchScales", 0)),
        is_topk_balance=attrs.get("_dispatchCommAlg", "fullmesh_v1") == "fullmesh_v2",
        is_x_random=int(attrs.get("isXRandom", 0)) == 1,
        zero_expert_num=int(attrs.get("zeroExpertNum", 0)),
        copy_expert_num=int(attrs.get("copyExpertNum", 0)),
        is_combine_x=(
            int(attrs.get("isCombineX", 0)) == 1
        ),
        is_shared_expert_x=bool(attrs.get("isSharedExpertX", 0)),
        is_add_rms_norm=first_ctx.api_name
        in (
            "aclnnMoeDistributeCombineAddRmsNorm",
            "aclnnMoeDistributeCombineAddRmsNormV2",
        ),
        is_deepep=is_deep_ep,
        is_elastic=int(attrs.get("isElastic", 0)) == 1,
        elastic_shared_rank_num=int(attrs.get("shareBrokenCardNum", 0)),
        elastic_moe_rank_num=int(attrs.get("moeBrokenCardNum", 0)),
        comm_quant_mode=int(attrs.get("commQuantMode", 0)),
        is_expand=int(attrs.get("isExpand", 0)) == 1,
        active_mask_dim=int(attrs.get("xActiveMaskDim", 0)),
    )
    case = case_module.load_or_generate_case(case_config)
    golden_x = case["golden_x"]
    is_add_rms_norm = first_ctx.api_name in (
        "aclnnMoeDistributeCombineAddRmsNorm",
        "aclnnMoeDistributeCombineAddRmsNormV2",
    )
    golden_y = case.get("golden_y") if is_add_rms_norm else None
    golden_rstd_out = case.get("golden_rstd_out") if is_add_rms_norm else None
    if is_add_rms_norm:
        golden_x = case.get("add_rms_norm_x", golden_x)
    if is_add_rms_norm and (golden_y is None or golden_rstd_out is None):
        raise ValueError("AddRmsNorm golden outputs are missing")
    x_active_mask_input_list = case.get("x_active_mask_input_list")
    compare_path = (
        Path(__file__).resolve().parents[4]
        / "common"
        / "tests"
        / "assets"
        / "mc2_compare.py"
    )
    compare_spec = importlib.util.spec_from_file_location(
        f"mc2_common_compare_{abs(hash(compare_path))}", compare_path
    )
    compare_module = importlib.util.module_from_spec(compare_spec)
    compare_spec.loader.exec_module(compare_module)

    for rank, did in enumerate(device_ids):
        tc = thread_contexts[did]
        ep_rank_id = int(tc.attributes.get("epRankId", rank))
        golden_rank = ep_rank_id
        active_mask_dim = int(first_ctx.attributes.get("xActiveMaskDim", 0))
        if (
            golden_rank < 0
            or (
                not case_config.is_expand
                and ep_rank_id not in case["available_ranks"]
            )
            or getattr(tc.prof_result, "api_prof", None) == "SUPPRESSED"
        ):
            all_precision.append(f"rank{did}:PASS(SUPPRESSED)")
            continue
        expected_outputs = (
            (
                golden_y[golden_rank].float(),
                golden_rstd_out[golden_rank].float(),
                golden_x[golden_rank].float(),
            )
            if is_add_rms_norm
            else (golden_x[golden_rank],)
        )
        if active_mask_dim > 0:
            row_mask = case.get("row_mask_list")
            if row_mask is None:
                raise ValueError("Active-mask Combine golden row mask is missing")
            expected_outputs = tuple(
                output[row_mask[golden_rank]]
                for output in expected_outputs
            )
        output_indexes = tc.output_tensor_indexes
        if len(output_indexes) != len(expected_outputs):
            raise ValueError(
                f"MoeDistributeCombine output count mismatch: got {len(output_indexes)}, "
                f"expected {len(expected_outputs)}"
            )
        output_results = []
        for output_position, expected in enumerate(expected_outputs):
            expected = expected.contiguous()
            reference = None
            output_index = output_indexes[output_position]
            actual = _decode_output(
                tc.prof_result.output_bytes[output_position],
                tc.flat_output_dtypes[output_position],
                expected.shape,
            )
            actual = actual.to(expected.dtype)
            if is_add_rms_norm:
                payload = getattr(tc, "_moe_add_rms_norm_e2e_payload", None)
                if payload is None:
                    raise RuntimeError(
                        f"MoeDistributeCombineAddRmsNorm reference missing on rank {did}"
                    )
                reference_bytes = payload[0][1][output_position]
                reference = _decode_output(
                    reference_bytes,
                    tc.flat_output_dtypes[output_position],
                    expected.shape,
                ).to(expected.dtype)
            if bool(int(first_ctx.attributes.get("_moeVariableBs", 0))):
                per_rank_bs = max(
                    int(first_ctx.attributes["globalBs"])
                    // int(first_ctx.attributes["epWorldSize"]),
                    1,
                )
                ep_rank = rank % int(first_ctx.attributes["epWorldSize"])
                tp_rank = rank // int(first_ctx.attributes["epWorldSize"])
                if x_active_mask_input_list is None:
                    raise ValueError("Variable-BS input mask is missing")
                input_mask = x_active_mask_input_list[tp_rank][
                    ep_rank * per_rank_bs : (ep_rank + 1) * per_rank_bs
                ]
                real_tokens = int(input_mask.sum())
                expected = expected[:real_tokens]
                actual = actual[:real_tokens]
                logging.info(
                    "MoeDistributeCombine variable-BS compare rank=%s "
                    "real_tokens=%s actual_shape=%s expected_shape=%s",
                    did,
                    real_tokens,
                    tuple(actual.shape),
                    tuple(expected.shape),
                )
            elif active_mask_dim > 0:
                row_mask = case.get("row_mask_list")
                if row_mask is None:
                    raise ValueError("Active-mask Combine golden row mask is missing")
                if expected.shape[0] > int(row_mask[golden_rank].sum()):
                    expected = expected[row_mask[golden_rank]]
                actual = actual[: expected.shape[0]]
            if is_add_rms_norm:
                if bool(int(first_ctx.attributes.get("_moeVariableBs", 0))):
                    reference = reference[:real_tokens]
                cr = compare_module.tripartite_compare(
                    actual.cpu(),
                    expected.cpu(),
                    reference.cpu(),
                    moe=False,
                )
                cr = {
                    **cr,
                    "precision": (
                        f"y={compare_module.moe_data_compare_np(actual.cpu(), expected.cpu())['precision']},"
                        f"target_diff_avg={cr['metrics']['target']['diff_big_avg']},"
                        f"reference_diff_avg={cr['metrics']['third_party']['diff_big_avg']}"
                        if output_position == 0
                        else cr["precision"]
                    ),
                }
            else:
                cr = compare_module.moe_data_compare_np(
                    actual,
                    expected,
                    diff_thd=0.001,
                    pct_thd=0.001,
                    max_diff_hd=0.0,
            )
            output_results.append(cr)
            if not cr["pass"]:
                mismatch = torch.empty((0, actual.dim()), dtype=torch.long)
                if actual.shape == expected.shape:
                    mismatch = (actual.float() != expected.float()).nonzero(as_tuple=False)
                else:
                    logging.error(
                        "MoeDistributeCombine output%s shape mismatch rank=%s "
                        "actual=%s expected=%s",
                        output_position,
                        did,
                        tuple(actual.shape),
                        tuple(expected.shape),
                    )
                if mismatch.numel() > 0:
                    first_index = tuple(mismatch[0].tolist())
                    if is_add_rms_norm:
                        diff = (actual.float() - expected.float()).abs().reshape(-1)
                        top = torch.topk(diff, min(3, diff.numel()))
                        cr["precision"] += (
                            f";first={actual[first_index].item()};"
                            f"expected={expected[first_index].item()};"
                            f"reference={reference[first_index].item()};"
                            f"max_abs_diff={top.values[0].item():.8g}"
                        )
                    if is_add_rms_norm and reference is not None:
                        logging.error(
                            "AddRmsNorm output%s tripartite mismatch rank=%s index=%s "
                            "actual=%s expected=%s reference=%s",
                            output_position,
                            did,
                            first_index,
                            actual[first_index].item(),
                            expected[first_index].item(),
                            reference[first_index].item(),
                        )
                    else:
                        logging.error(
                            "MoeDistributeCombine output%s mismatch rank=%s index=%s "
                            "actual=%s expected=%s",
                            output_position,
                            did,
                            first_index,
                            actual[first_index].item(),
                            expected[first_index].item(),
                        )
                if is_add_rms_norm and output_position == 2:
                    logging.error(
                        "AddRmsNorm x mismatch count=%s",
                        int(mismatch.shape[0]),
                    )
                if actual.shape == expected.shape:
                    diff_index = (actual.float() != expected.float()).nonzero(as_tuple=False)
                    if diff_index.numel() > 0:
                        first_index = tuple(diff_index[0].tolist())
                        logging.error(
                            "MoeDistributeCombine mismatch detail rank=%s output=%s index=%s actual=%s expected=%s",
                            did,
                            output_position,
                            first_index,
                            actual[first_index].item(),
                            expected[first_index].item(),
                        )
                        import os
                        import tempfile

                        dump_path = os.path.join(
                            tempfile.gettempdir(),
                            f"moe_combine_mismatch_rank_{did}.pt",
                        )
                        torch.save(
                            {
                                "actual": actual.cpu(),
                                "expected": expected.cpu(),
                                "case_name": first_ctx.testcase_name,
                            },
                            dump_path,
                        )
        if all(cr["pass"] for cr in output_results):
            precision = ";".join(str(cr["precision"]) for cr in output_results)
            all_precision.append(f"rank{did}:PASS({precision})")
            logging.info(f"MoeDistributeCombine: rank dev={did} PASSED")
        else:
            reasons = ";".join(
                f"output{index}:{cr['error_info']}"
                for index, cr in enumerate(output_results)
                if not cr["pass"]
            )
            details = ";".join(
                f"output{index}:{cr['precision']}"
                for index, cr in enumerate(output_results)
                if not cr["pass"]
            )
            all_precision.append(f"rank{did}:FAIL({reasons}|{details})")
            logging.error(f"MoeDistributeCombine: rank dev={did} FAILED: {reasons}")

    return

    to_torch_f32, fmt_compare, Comparator = _load_framework()

    to_torch_f32, fmt_compare, Comparator = _load_framework()
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    world_size = len(device_ids)
    ep_ws = int(attrs.get("epWorldSize", world_size))
    moe_expert_num = int(attrs.get("moeExpertNum", 1))
    local_expert_num = moe_expert_num // ep_ws if ep_ws > 0 else moe_expert_num

    all_rank_expand_x = {}
    all_rank_send_counts = {}
    all_rank_expert_ids = {}
    all_rank_expand_idx = {}
    all_rank_expert_scales = {}

    for did in device_ids:
        tc = thread_contexts[did]
        expand_x = tc.flatten_tensors[0]
        expert_ids = tc.flatten_tensors[1]
        expand_idx = tc.flatten_tensors[2]
        ep_send_counts = tc.flatten_tensors[3]
        expert_scales = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None

        h = expand_x.shape[1]

        if ep_send_counts is not None and ep_send_counts.numel() > 0:
            esc = ep_send_counts.float()
            send_counts = []
            prev = 0.0
            for i in range(min(ep_ws, esc.numel())):
                cur = float(esc[i])
                send_counts.append(int(cur - prev))
                prev = cur
            while len(send_counts) < ep_ws:
                send_counts.append(0)
        else:
            num_tokens = expand_x.shape[0]
            per_rank = num_tokens // ep_ws
            send_counts = [per_rank] * ep_ws
            remainder = num_tokens % ep_ws
            for i in range(remainder):
                send_counts[i] += 1

        all_rank_expand_x[did] = expand_x
        all_rank_send_counts[did] = send_counts
        all_rank_expert_ids[did] = (
            expert_ids
            if expert_ids is not None
            else torch.zeros(0, 1, dtype=torch.int32)
        )
        all_rank_expand_idx[did] = (
            expand_idx if expand_idx is not None else torch.zeros(0, dtype=torch.int32)
        )
        all_rank_expert_scales[did] = (
            expert_scales if expert_scales is not None else torch.zeros(0)
        )

    alltoallv_out = _simulate_moe_alltoallv(
        all_rank_expand_x, device_ids, all_rank_send_counts
    )

    for did in device_ids:
        tc = thread_contexts[did]
        a2a_result = alltoallv_out[did]
        expert_ids = all_rank_expert_ids[did]
        expand_idx = all_rank_expand_idx[did]
        expert_scales = all_rank_expert_scales[did]

        h = a2a_result.shape[1] if a2a_result.dim() > 1 else 1
        bs = expert_ids.shape[0] if expert_ids.dim() > 0 else 0
        k = expert_ids.shape[1] if expert_ids.dim() > 1 else 1

        x_out = torch.zeros(bs, h, dtype=expand_x.dtype)
        if a2a_result.numel() > 0 and bs > 0 and expand_idx.numel() > 0:
            num_valid = min(expand_idx.shape[0], a2a_result.shape[0])
            for idx_pos in range(num_valid):
                orig_flat_idx = int(expand_idx[idx_pos])
                orig_token = orig_flat_idx // k if k > 0 else orig_flat_idx
                k_idx = orig_flat_idx % k if k > 0 else 0
                if orig_token < bs:
                    scale = 1.0
                    if expert_scales is not None and expert_scales.numel() > 0:
                        if (
                            expert_scales.dim() > 1
                            and orig_token < expert_scales.shape[0]
                            and k_idx < expert_scales.shape[1]
                        ):
                            scale = float(expert_scales[orig_token][k_idx])
                        elif (
                            expert_scales.dim() == 1
                            and orig_flat_idx < expert_scales.shape[0]
                        ):
                            scale = float(expert_scales[orig_flat_idx])
                    x_out[orig_token] = (
                        x_out[orig_token] + a2a_result[idx_pos].float() * scale
                    )

        x_out = x_out.to(expand_x.dtype)
        tc.golden_tensors = [x_out.contiguous()]
        try:
            cr = Comparator(tc).compare()
            all_precision.append(f"rank{did}:{cr.passed}({fmt_compare(cr)})")
            if cr.passed != "PASS":
                logging.error(
                    f"MoeDistributeCombine: rank dev={did} FAILED: {cr.precision}"
                )
            else:
                logging.info(f"MoeDistributeCombine: rank dev={did} PASSED")
        except Exception:
            logging.exception(f"MoeDistributeCombine: rank dev={did} compare failure")
            all_precision.append(f"rank{did}:COMPARE_EXCEPTION")
