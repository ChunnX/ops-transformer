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

"""Golden for aclnnMoeDistributeDispatch / torch_npu.npu_moe_distribute_dispatch.

MoeDistributeDispatch: sort tokens by destination expert -> all_to_allv -> output.
Pure CPU golden, no HCCL cascade (communication simulated on CPU).
Multi-output: recv_x, dynamic_scales, expand_idx, expert_token_nums,
              ep_recv_counts, tp_recv_counts, expand_scales.

E2E golden returns a list (one per rank) of the recv_x output tensor.
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
    spec = importlib.util.spec_from_file_location("mc2_moe_case_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
# E2E golden: returns list of recv_x tensors (one per rank)
# ---------------------------------------------------------------------------


def moe_distribute_dispatch_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    import torch

    world_size = len(cpu_inputs_per_rank)
    ep_ws = int(attrs.get("ep_world_size", world_size))
    moe_expert_num = int(attrs.get("moe_expert_num", 1))
    local_expert_num = moe_expert_num // ep_ws if ep_ws > 0 else moe_expert_num

    all_rank_expand_x = {}
    all_rank_send_counts = {}

    for r in range(world_size):
        inputs = cpu_inputs_per_rank[r]
        x = inputs[0].float() if inputs[0] is not None else None
        expert_ids = inputs[1] if (len(inputs) > 1 and inputs[1] is not None) else None
        if x is None or expert_ids is None:
            all_rank_expand_x[r] = torch.zeros(0, 1)
            all_rank_send_counts[r] = [0] * ep_ws
            continue

        bs = x.shape[0]
        h = x.shape[1]
        k = expert_ids.shape[1] if expert_ids.dim() > 1 else 1

        # Sort tokens by destination rank (expert_id // local_expert_num)
        send_counts = [0] * ep_ws
        token_groups = [[] for _ in range(ep_ws)]
        for i in range(bs):
            for j in range(k):
                eid = (
                    int(expert_ids[i][j])
                    if expert_ids.dim() > 1
                    else int(expert_ids[i])
                )
                dest_rank = eid // local_expert_num if local_expert_num > 0 else 0
                if dest_rank >= ep_ws:
                    dest_rank = ep_ws - 1
                token_groups[dest_rank].append(x[i])
                send_counts[dest_rank] += 1

        sorted_tokens = []
        for dest in range(ep_ws):
            sorted_tokens.extend(token_groups[dest])

        if sorted_tokens:
            expand_x_local = torch.stack(sorted_tokens, dim=0)
        else:
            expand_x_local = torch.zeros(0, h)

        all_rank_expand_x[r] = expand_x_local
        all_rank_send_counts[r] = send_counts

    alltoallv_out = _simulate_moe_alltoallv(
        all_rank_expand_x, world_size, all_rank_send_counts
    )

    # The NPU expand_x output is padded to tp_ws * A_moe per rank, where
    # A_moe = global_bs * min(local_expert_num, k) = bs * ep_ws * min(local_expert_num, k).
    # We pad the golden recv_x to match this size for isclose comparison.
    first_x = cpu_inputs_per_rank[0][0]
    bs = first_x.shape[0] if first_x is not None else 0
    h = first_x.shape[1] if first_x is not None else 1
    k = 1
    for inp in cpu_inputs_per_rank:
        if len(inp) > 1 and inp[1] is not None:
            eid = inp[1]
            k = eid.shape[1] if eid.dim() > 1 else 1
            break
    tp_ws = int(attrs.get("tp_world_size", 1))
    a_moe = bs * ep_ws * min(local_expert_num, k)
    target_len = tp_ws * a_moe

    goldens = []
    for r in range(world_size):
        recv_x = alltoallv_out.get(r, torch.zeros(0, h))
        if recv_x.shape[0] < target_len:
            pad_size = target_len - recv_x.shape[0]
            pad = torch.zeros(pad_size, h, dtype=recv_x.dtype)
            recv_x = torch.cat([recv_x, pad], dim=0)
        elif recv_x.shape[0] > target_len:
            recv_x = recv_x[:target_len]
        goldens.append(recv_x.contiguous())
    return goldens


# ---------------------------------------------------------------------------
# ACLNN multi-device golden
# ---------------------------------------------------------------------------


def moe_distribute_dispatch_multi_device_golden(
    thread_contexts, device_ids, all_precision
):
    import torch

    to_torch_f32, fmt_compare, Comparator = _load_framework()
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    case_module = _load_case_data_module()
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
        local_moe_expert_num_override=int(attrs.get("localMoeExpertNum", 0)),
        shared_expert_num=int(attrs.get("sharedExpertNum", 0)),
        shared_expert_rank_num=int(attrs.get("sharedExpertRankNum", 0)),
        const_expert_num=int(attrs.get("constExpertNum", 0)),
        zero_expert_num=int(attrs.get("zeroExpertNum", 0)),
        copy_expert_num=int(attrs.get("copyExpertNum", 0)),
        rank_num_per_shared_expert=(
            int(attrs.get("sharedExpertRankNum", 0))
            // int(attrs.get("sharedExpertNum", 0) or 1)
        ),
        input_dtype=str(first_ctx.tensor_dtypes[0]).replace("torch.", ""),
        is_mask=int(attrs.get("xActiveMaskDim", 0)) > 0,
        is_variable_bs=bool(attrs.get("_moeVariableBs", 0)),
        is_quant=int(attrs.get("quantMode", 0)) == 2,
        is_dispatch_scales=bool(attrs.get("isDispatchScales", 0)),
        is_topk_balance=attrs.get("commAlg") == "fullmesh_v2",
        is_deepep=bool(attrs.get("_moeDeepep", 0)),
        is_elastic=int(attrs.get("isElastic", 0)) == 1,
        elastic_shared_rank_num=int(attrs.get("shareBrokenCardNum", 0)),
        elastic_moe_rank_num=int(attrs.get("moeBrokenCardNum", 0)),
        comm_quant_mode=int(attrs.get("commQuantMode", 0)),
        is_expand=bool(attrs.get("isExpand", 0)),
    )
    case = case_module.load_or_generate_case(case_config)
    ep_ws = int(attrs.get("epWorldSize", len(device_ids)))
    moe_expert_num = int(attrs.get("moeExpertNum", 1))
    shared_expert_num = int(attrs.get("sharedExpertNum", 1))
    shared_expert_rank_num = int(attrs.get("sharedExpertRankNum", 0))
    shared_expert_rank_num = (
        shared_expert_num
        if shared_expert_rank_num == 0 and shared_expert_num > 0
        else shared_expert_rank_num
    )
    moe_rank_num = ep_ws - shared_expert_rank_num
    local_expert_num = moe_expert_num // max(moe_rank_num, 1)
    tp_ws = int(attrs.get("tpWorldSize", 1))
    global_bs = int(attrs.get("globalBs", 0))
    quant_mode = int(attrs.get("quantMode", 0))
    zero_copy_const_num = (
        int(attrs.get("zeroExpertNum", 0))
        + int(attrs.get("copyExpertNum", 0))
        + int(attrs.get("constExpertNum", 0))
    )

    rank_num_per_shared_expert = (
        shared_expert_rank_num // shared_expert_num if shared_expert_num > 0 else 0
    )
    max_shared_group_num = (
        (ep_ws + rank_num_per_shared_expert - 1) // rank_num_per_shared_expert
        if rank_num_per_shared_expert > 0
        else 0
    )
    first_shape = first_ctx.tensor_view_shapes
    topk = int(first_shape[1][1]) if len(first_shape) > 1 and len(first_shape[1]) > 1 else 1
    source_bs = first_ctx.flatten_tensors[0].shape[0]
    a_shared = source_bs * max_shared_group_num
    a_moe = int(attrs.get("globalBs", 0)) * min(local_expert_num, topk)

    source = []
    elastic_active_rank_ids = (
        case_config.available_ranks if case_config.is_elastic else range(ep_ws)
    )
    shared_output_rank_ids = (
        range(case_config.elastic_shared_expert_rank_num)
        if case_config.is_elastic
        else range(shared_expert_rank_num)
    )
    shared_group_stride = (
        case_config.elastic_rank_num_per_shared_expert
        if case_config.is_elastic
        else rank_num_per_shared_expert
    )
    for did in device_ids:
        tc = thread_contexts[did]
        x = tc.flatten_tensors[0]
        expert_ids = tc.attributes.get("_moeDeepepExpertIdsExpand", tc.flatten_tensors[1])
        mask = tc.flatten_tensors[3]
        bs, h = x.shape
        k = expert_ids.shape[1] if expert_ids.dim() > 1 else 1

        source_scales = thread_contexts[did].flatten_tensors[2]
        expert_total = moe_expert_num + int(attrs.get("zeroExpertNum", 0)) + int(attrs.get("copyExpertNum", 0)) + int(attrs.get("constExpertNum", 0))
        grouped_tokens = [[] for _ in range(expert_total)]
        for token_id in range(bs):
            for k_id in range(k):
                if mask is None:
                    is_valid = True
                elif mask.dim() == 2:
                    is_valid = bool(mask[token_id, k_id])
                else:
                    is_valid = bool(mask[token_id])
                if not is_valid:
                    continue
                expert_id = int(expert_ids[token_id, k_id])
                if expert_id >= moe_expert_num:
                    continue
                scale_row = shared_expert_num + expert_id
                expert_scale = (
                    source_scales[scale_row]
                    .float()
                    if source_scales is not None
                    else torch.ones(h)
                )
                grouped_tokens[expert_id].append((x[token_id].float(), expert_scale))
        source.append((x, mask, grouped_tokens))

    def cal_quant(tokens, scales):
        if tokens.numel() == 0 or scales.numel() == 0:
            return tokens, torch.zeros(tokens.shape[0], dtype=torch.float32)
        if scales.dim() == 1:
            scales = scales[:, None]
        if scales.shape[0] < tokens.shape[0]:
            scales = torch.cat(
                (
                    scales,
                    torch.ones(
                        tokens.shape[0] - scales.shape[0],
                        scales.shape[1],
                        dtype=scales.dtype,
                    ),
                )
            )
        scaled_tokens = tokens * scales[: tokens.shape[0]]
        dynamic_scales = scaled_tokens.abs().max(dim=1).values / 127.0
        dynamic_scales[dynamic_scales == 0] = 1.0
        quantized_tokens = (scaled_tokens / dynamic_scales[:, None]).round().to(torch.int8)
        return quantized_tokens, dynamic_scales

    goldens = {}
    for rank_idx, did in enumerate(device_ids):
        if rank_idx < shared_expert_rank_num:
            target_a = a_shared
            is_active_shared_rank = rank_idx in shared_output_rank_ids
            shared_group_id = (
                rank_idx % max(shared_group_stride, 1)
                if is_active_shared_rank
                else -1
            )
            generated = case_module.rank_case(case, case_config, rank_idx, tp_rank_id=0)
            send_card_ids = (
                [
                    elastic_active_rank_ids[local_rank_id]
                    for local_rank_id in range(
                        shared_group_id,
                        len(elastic_active_rank_ids),
                        max(shared_group_stride, 1),
                    )
                ]
                if is_active_shared_rank
                else []
            )
            token_parts = [
                case_module.rank_case(
                    case, case_config, card_id, tp_rank_id=0
                )["x"].float()
                for card_id in send_card_ids
            ]
            index_parts = []
            received_tokens = torch.cat(token_parts) if token_parts else torch.zeros(0, first_shape[0][1])
            expert_token_nums = torch.ones(1, dtype=torch.int64)
            dynamic_scales = torch.zeros(0)
        else:
            target_a = a_moe
            moe_rank_id = rank_idx - shared_expert_rank_num
            expert_offset = moe_rank_id * local_expert_num
            token_segments = []
            index_segments = []
            expert_token_nums = torch.zeros(local_expert_num, dtype=torch.int64)
            generated = case_module.rank_case(case, case_config, rank_idx, tp_rank_id=0)
            golden_expand_x = generated["expand_x"]
            dynamic_scales = torch.empty(0, dtype=torch.float32)
            for local_expert_id in range(local_expert_num):
                expert_id = expert_offset + local_expert_id
                tokens = []
                expert_scales = []
                for src_rank, (_, _, grouped_tokens) in enumerate(source):
                    for token in grouped_tokens[expert_id]:
                        tokens.append(token[0])
                        expert_scales.append(token[1])
                        index_segments.append((src_rank, -1, -1))
                if tokens:
                    token_segments.append(torch.stack(tokens))
                    expert_token_nums[local_expert_id] = len(tokens)
                else:
                    token_segments.append(torch.zeros(0, first_shape[0][1]))

            received_tokens = (
                torch.cat(token_segments) if token_segments else torch.zeros(0, first_shape[0][1])
            )
            rank_scale_segments = []
            for local_expert_id in range(local_expert_num):
                expert_id = expert_offset + local_expert_id
                rank_scale_segments.extend(
                    token[1]
                    for (_, _, grouped_tokens) in source
                    for token in grouped_tokens[expert_id]
                )
            rank_source_scales = (
                torch.stack(rank_scale_segments) if rank_scale_segments else torch.empty(0, first_shape[0][1])
            )

        if quant_mode == 2:
            if rank_idx < shared_expert_rank_num:
                source_scale_parts = []
                shared_expert_id = rank_idx // max(shared_group_stride, 1)
                for src_rank in send_card_ids:
                    src_scales = thread_contexts[device_ids[src_rank]].flatten_tensors[2]
                    src_mask = source[src_rank][1]
                    valid_token_num = (
                        int(src_mask.any(dim=1).sum().item())
                        if src_mask is not None and src_mask.dim() == 2
                        else int(src_mask.sum().item())
                    )
                    if src_scales is None:
                        source_scale_parts.append(torch.ones(valid_token_num, first_shape[0][1]))
                    else:
                        source_scale_parts.append(
                            src_scales[shared_expert_id]
                            .float()
                            .expand(valid_token_num, first_shape[0][1])
                        )
                source_scales = (
                    torch.cat(source_scale_parts)
                    if source_scale_parts
                    else torch.empty(0, first_shape[0][1])
                )
            else:
                source_scales = rank_source_scales
            received_tokens, dynamic_scales = cal_quant(received_tokens, source_scales)

        expand_x = torch.zeros((target_a, first_shape[0][1]), dtype=received_tokens.dtype)
        if rank_idx >= shared_expert_rank_num:
            valid_rows = min(target_a, int(generated["tokens"].sum().item()))
            compact_rows = min(golden_expand_x.shape[0], valid_rows)
            expand_x[:compact_rows] = golden_expand_x[:compact_rows]
        else:
            valid_rows = min(target_a, received_tokens.shape[0])
            expand_x[:valid_rows] = received_tokens[:valid_rows]
        tc = thread_contexts[did]
        tc.attributes["_moeValidExpandRows"] = valid_rows
        dynamic_scales = torch.cat(
            [dynamic_scales, torch.zeros(max(target_a - dynamic_scales.shape[0], 0), dtype=torch.float32)]
        )[:target_a]

        ep_recv = torch.zeros(ep_ws, dtype=torch.int32)
        tp_recv = torch.zeros(max(tp_ws, 1), dtype=torch.int32)
        expand_scales = None
        goldens[did] = (
            expand_x,
            dynamic_scales,
            torch.zeros(max(bs * topk, target_a * 128), dtype=torch.int32),
            expert_token_nums,
            ep_recv,
            tp_recv,
            expand_scales,
        )

    for did in device_ids:
        tc = thread_contexts[did]
        if getattr(tc.prof_result, "api_prof", None) == "SUPPRESSED":
            all_precision.append(f"rank{did}:PASS(SUPPRESSED)")
            continue
        goldens[did] = [
            tensor.contiguous() if tensor is not None else None
            for tensor in goldens[did]
        ]
        tc.golden_tensors = goldens[did]
        if isinstance(tc.prof_result.output_bytes, tuple):
            tc.prof_result.output_bytes = list(tc.prof_result.output_bytes)
        try:
            cr = Comparator(tc).compare()
            actual = tc.prof_result.output_bytes[0]
            expected = goldens[did][0]
            from ttk.utilities.dtypes import torch_to_numpy_tensor

            if hasattr(actual, "cpu"):
                actual = actual.cpu()
            if hasattr(actual, "numpy"):
                actual = torch_to_numpy_tensor(actual)
            if hasattr(expected, "cpu"):
                expected = expected.cpu()
            if int(did) in (0, 1):
                logging.info(
                    "MoeDistributeDispatch output detail rank=%s first64=%s",
                    did,
                    actual[: min(64, actual.shape[0])][:, 0].tolist(),
                )
            if hasattr(expected, "numpy"):
                expected = torch_to_numpy_tensor(expected)
            mismatch_rows = (
                (actual != expected).any(axis=1).nonzero()[0]
                if actual.shape == expected.shape
                else []
            )
            logging.info(
                "MoeDistributeDispatch detail rank=%s shape=%s/%s mismatch_rows=%s first=%s actual=%s expected=%s",
                did,
                getattr(actual, "shape", None),
                getattr(expected, "shape", None),
                int(len(mismatch_rows)),
                None if len(mismatch_rows) == 0 else int(mismatch_rows[0]),
                None if len(mismatch_rows) == 0 else actual[mismatch_rows[0]][:8].tolist(),
                None if len(mismatch_rows) == 0 else expected[mismatch_rows[0]][:8].tolist(),
            )
            all_precision.append(f"rank{did}:{cr.passed}({fmt_compare(cr)})")
            if cr.passed != "PASS":
                logging.error(
                    f"MoeDistributeDispatch: rank dev={did} FAILED: {cr.precision}"
                )
            else:
                logging.info(f"MoeDistributeDispatch: rank dev={did} PASSED")
        except Exception:
            logging.exception(f"MoeDistributeDispatch: rank dev={did} compare failure")
            all_precision.append(f"rank{did}:COMPARE_EXCEPTION")
